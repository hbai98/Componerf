from typing import Union
import numpy as np
import copy
import torch
import torch as th
import torch.nn as nn
from src.compo_nerf.training.losses.shape_loss import ShapeLoss
from src.compo_nerf.training.losses.sparsity_loss import sparsity_loss
from src.compo_nerf.models.render_utils import safe_normalize
from src.compo_nerf.raymarching.raymarchinglatent.raymarching import composite_rays_train
from src.compo_nerf.models.nerf_utils import MLP, trunc_exp
from src.compo_nerf.models.encoding import get_encoder
from src.compo_nerf.models.network_grid import NeRFNetwork

from src.compo_nerf.cp_nerf_utils import box_pts, world2object, get_common_rays, func_crop, normalize_bound

from einops import rearrange, repeat
from itertools import chain
import torch.nn.functional as F

from PIL import Image
from einops import rearrange


class CompoNeRF(nn.Module):
    def __init__(self, cfg, diffusion, hidden_dim=64, writer=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.bound = cfg.render.bound
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.sjc = cfg.guide.sjc
        additional_dim_size = 1
        self._last_all_obj_properties = None
        self.dict_id2classnode = {}
        self.num_classNode = 0
        self.bg_radius = 0
        self.diffusion = diffusion
        self.hidden_dim = hidden_dim
        self.num_layers = cfg.guide.num_layers
        self.use_global_albedo = cfg.guide.use_global_albedo
        self.use_global_loss = cfg.guide.use_global_loss
        self.use_global_density = cfg.guide.use_global_density
        self.use_local_loss = cfg.guide.use_local_loss
        self.box_scale = cfg.guide.box_scale
        self.with_residual = cfg.guide.with_residual
        self.with_mlp_residual = cfg.guide.with_mlp_residual
        self.train_with_crop = cfg.guide.train_with_crop
        self.prompt = cfg.guide.text
        self.num_layers = cfg.guide.num_layers
        self.use_params = cfg.guide.use_hyper_params
        self.text_z = self.calc_text_embeddings(self.prompt)
        if self.use_params:
            self.global_sigma = nn.Parameter(th.tensor(1e-3))
            self.global_color = nn.Parameter(th.tensor(1e-3))
            self.global_color_direction = nn.Parameter(th.tensor(1e-3))
        self.writer = writer

        if self.use_global_loss:
            if self.use_global_albedo or self.use_global_density:
                self.encoder_o, self.in_dim_o = get_encoder('tiledgrid', input_dim=3,
                                                            desired_resolution=2048 * self.bound)
                self.encoder_d, self.in_dim_d = get_encoder('frequency', input_dim=3)
                # (directions + positions)
                if self.use_global_albedo:
                    self.global_albedo_net = MLP(self.in_dim_d, 4, hidden_dim, self.num_layers, bias=True,
                                                 res=self.with_mlp_residual)
                if self.use_global_density:
                    self.global_density_net = MLP(self.in_dim_o, 4 + additional_dim_size, hidden_dim, self.num_layers,
                                                  bias=True, res=self.with_mlp_residual)
                if not self.use_global_density and self.use_global_albedo:
                    self.global_albedo_net = MLP(self.in_dim_d + self.in_dim_o, 4, hidden_dim, self.num_layers,
                                                 bias=True, res=self.with_mlp_residual)
        self.init_nodes(cfg.guide)

    def gaussian(self, x):
        # x: [B, N, 3]

        d = (x ** 2).sum(-1)
        g = 5 * torch.exp(-d / (2 * 0.2 ** 2))

        return g

    def global_forward(self, o, d, albedo, density, with_residual=True, step=None):
        if not self.use_global_loss:
            return albedo, density
        if not self.use_global_albedo and not self.use_global_density:
            return albedo, density

        h_o = self.encoder_o(o, bound=self.bound)
        h_d = self.encoder_d(d, bound=self.bound)

        if self.use_global_density:
            h = self.global_density_net(h_o)

            if with_residual:
                if self.use_params:
                    density_ = self.global_sigma * h[..., 0] + density  # residual
                else:
                    density_ = h[..., 0] + density  # residual
            else:
                density_ = h[..., 0]

            density = trunc_exp(density_ + self.gaussian(o))
            color = h[..., 1:]

            if self.use_global_albedo:
                h = self.global_albedo_net(h_d)  # [N, 3]
                if self.use_params:
                    color = color + self.global_color_direction * h
                else:
                    color = color + h

        if not self.use_global_density and self.use_global_albedo:
            color = self.global_albedo_net(th.cat([h_o, h_d], dim=-1))

        if with_residual:
            if self.use_params:
                albedo = albedo + self.global_color * color  # residul
            else:
                albedo = albedo + color  # residul
        else:
            albedo = color

        if self.use_params and self.training and step is not None:
            self.writer.add_scalar('orig/color', self.global_color, step)
            self.writer.add_scalar('orig/sigma', self.global_sigma, step)
            self.writer.add_scalar('orig/direction', self.global_color_direction, step)

        return albedo, density

    def calc_text_embeddings(self, ref_text=None):
        if ref_text is None:
            ref_text = self.cfg.guide.text
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
        else:
            text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                text = f"{ref_text}, {d} view"
                text_z.append(self.diffusion.get_text_embeds([text]))
        return text_z

    def init_nodes(self, cfg):
        sub_text_list = cfg.node_text_list  # ['red apple', 'yellow apple'],
        pose_list = [torch.tensor(pos) for pos in cfg.node_pos_list]
        dim_list = [torch.tensor(dim) for dim in cfg.node_dim_list]
        poses = th.stack(pose_list)
        dims = th.stack(dim_list)
        poses_ = poses.clone()
        poses[:, -1] = poses_[:, 1]
        poses[:, 1] = poses_[:, -1]
        dims_ = dims.clone()
        dims[:, -1] = dims_[:, 1]
        dims[:, 1] = dims_[:, -1]
        # normalize the scale to [-1, 1]
        pose_list, dim_list = normalize_bound(self.bound, poses, dims, box_scale=self.box_scale)
        # print(pose_list)
        # print(dim_list/2)
        # assert 0
        # hack for test
        for class_id, (sub_text, pose, dim) in enumerate(zip(sub_text_list,
                                                             pose_list,
                                                             dim_list)):
            self.addClassNode(class_id,
                              sub_text,
                               pose=pose.to(self.device),
                               dim=dim.to(self.device))

    def addClassNode(self, class_id, sub_text, pose=None, dim=None):
        class_node_cfg = copy.deepcopy(self.cfg)
        class_node_cfg.guide.text = sub_text
        if self.cfg.guide.shape_list is not None:
            class_node_cfg.guide.shape_path = self.cfg.guide.shape_list[class_id]
        else:
            class_node_cfg.guide.shape_path = None
        # hack for test:
        # if 'apple' in sub_text:
        # class_node_cfg.guide.shape_path = 'shapes/apple_mesh.obj'
        # elif 'banana' in sub_text:
        # class_node_cfg.guide.shape_path = 'shapes/banana_mesh.obj'
        # elif 'man' in sub_text or 'boy' in sub_text:
        # class_node_cfg.guide.shape_path = 'shapes/man_mesh.obj'
        # else:
        # class_node_cfg.guide.shape_path = None

        self.dict_id2classnode[class_id] = ClassNode(class_node_cfg, class_id, pose=pose,dim=dim)
        if self.diffusion is None:
            text_z = None
        else:
            text_z = self.calc_text_embeddings(sub_text)
        self.dict_id2classnode[class_id].text_z = text_z
        self.num_classNode += 1
        self._last_all_obj_properties = None

    def _get_obj_properties(self, ):
        """Get all objectt nodes properties.

        Returns:
            res: a batch of object info [N_obj, (pose, theta_y, dim, id_c, id_o)] with 'nan' representing the miss. 
        """
        if self._last_all_obj_properties is not None:
            return self._last_all_obj_properties
        r = []
        for id_c, c_node in self.dict_id2classnode.items():
            info = []
            info.append(c_node.pose) if c_node.pose is not None \
                else info.append(th.tensor(float('nan')).to(self.device))
            info.append(c_node.theta_y) if c_node.theta_y is not None \
                else info.append(th.tensor(float('nan')).to(self.device))
            info.append(c_node.dim) if c_node.dim is not None \
                else info.append(th.tensor(float('nan')).to(self.device))
            info.append(id_c)
            r.append(info)
        self._last_all_obj_properties = r
        if len(r) == 0:
            raise ValueError('It is not initalized properly.')
        return self._last_all_obj_properties

    def get_params(self, lr):
        all_params = [
            self.dict_id2classnode[c_node].nerf.get_params(lr)
            for c_node in self.dict_id2classnode.keys()
        ]
        # add params from sg
        if self.use_params:
            all_params.append([
                {'params': [self.global_sigma, self.global_color, self.global_color_direction], 'lr': lr},
            ])
        params = []
        for p in all_params:
            params += p
        return params

    def update_extra_state(self, decay=0.95, S=128):
        for id_c, c_node in self.dict_id2classnode.items():
            c_node.nerf.update_extra_state(decay=decay, S=S)

    def update_counter(self, c_node):
        nerf = c_node.nerf
        counter = nerf.step_counter[nerf.local_step % 16]
        counter.zero_()  # set to 0
        nerf.local_step += 1
        return counter

    def render(self, data,
               dt_gamma=0,
               light_d=None,
               ambient_ratio=0.1,
               shading='albedo',
               bg_color=None,
               perturb=False,
               force_all_rays=False,
               max_steps=1024,
               debug=False,
               step=None,
               ):
        """Run cuda rendering.

        Args:
            dt_gamma (int, optional): _description_. Defaults to 0.
            light_d (_type_, optional): _description_. Defaults to None.
            ambient_ratio (int, optional): _description_. Defaults to 1.
            shading (str, optional): _description_. Defaults to 'albedo'.
            bg_color (_type_, optional): _description_. Defaults to None.
            perturb (bool, optional): _description_. Defaults to False.
            force_all_rays (bool, optional): _description_. Defaults to False.
            max_steps (int, optional): _description_. Defaults to 1024.

        Returns:
            _type_: _description_
        """

        rays_o, rays_d = data['rays_o'], data['rays_d']
        prefix = rays_o.shape[:-1]
        B = rays_o.size(0)
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]
        rays_o = rays_o.contiguous().view(-1, 3)  # world frame
        rays_d = rays_d.contiguous().view(-1, 3)
        H, W = data['H'], data['W']
        N = rays_o.size(0)
        device = rays_o.device

        RAY_SIZE = N

        # get all nsg node info
        # (pose, theta_y, dim, id_c, id_o)
        all_obj_properties = self._get_obj_properties()
        poses = th.stack([p[0] for p in all_obj_properties])
        theta_y = th.stack([p[1] for p in all_obj_properties])
        dim = th.stack([p[2] for p in all_obj_properties])
        num_class = poses.size(0)
        # poses = th.cat([poses, th.zeros(1, 3).cuda()])
        # theta_y = th.cat()

        # repeat rays for each objects
        poses = repeat(poses, 'N d -> B N d', B=N).cuda()
        theta_ys = repeat(theta_y, 'N -> B N', B=N)
        theta_ys = th.nan_to_num(theta_ys, ).cuda()
        dims = repeat(dim, 'N d -> B N d', B=N).cuda()

        # ray-box intersections
        f_w, f_o, intersection_map = box_pts(
            (rays_o, rays_d), poses, dim=dims, theta_y=theta_ys)

        rays_o_o, rays_d_o, nears_o, fars_o = f_o  # local frame
        rays_o_w, rays_d_w, nears_w, fars_w = f_w  # world frame

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + th.randn(3, device=device, dtype=th.float))
            light_d = safe_normalize(light_d)

        iIdx2Ct = intersection_map[:, 1]
        iIdx2wIdx = intersection_map[:, 0]
        results = {}

        if self.training:
            local_losses = []
        sigmas_w = []
        rgbs_w = []
        normals_w = []
        deltas_w = []
        xyzs_os = []
        dirs_os = []
        IIdxes = []
        # global point and ray indexes
        rays_idx_w = []
        # ------------------------------------------------------
        # local rendering
        # use local indexes to infer each ray in the local bounding boxes
        for i, (cid, node) in enumerate(self.dict_id2classnode.items()):
            counter = self.update_counter(node)
            # the intersected world rays' indexes and their corresponding obejct categories
            nerf = node.nerf
            # scale the bounding box for each local positions
            xyzs_o, dirs_o, deltas_o, rays_idx_o = nerf.raymarching.march_rays_train(
                rays_o_o[iIdx2Ct == cid], rays_d_o[iIdx2Ct == cid], nerf.bound,
                nerf.density_bitfield, nerf.cascade,
                nerf.grid_size, nears_o[iIdx2Ct == cid], fars_o[iIdx2Ct == cid], counter,
                nerf.mean_count, perturb, 128, force_all_rays, dt_gamma,
                max_steps)

            # p = rays_o_o[iIdx2Ct==cid]+nears_o[iIdx2Ct==cid].unsqueeze(1)*rays_d_o[iIdx2Ct==cid]
            # only valid rangge
            lIdx2Ptidxes = [list(range(r[1], r[1] + r[2]))
                            for r in rays_idx_o if r[2] != 0]
            ptIdx = list(chain(*lIdx2Ptidxes))
            # select points which recorded by rays_idx_w
            xyzs_o = xyzs_o[ptIdx]
            dirs_o = dirs_o[ptIdx]
            deltas_o = deltas_o[ptIdx]
            if len(xyzs_o) < 10:
                return None
            # note: deltas for colors(0) and depth(1) are not equal
            # print(th.equal(deltas_o[:,0], deltas_o[:,1])) # False
            # revise rays index maps wihtout empty rays

            rays_idx_o = rays_idx_o[rays_idx_o[:, 2] != 0]
            # function : local index to intersected rays' index
            lIdx2IIdx = rays_idx_o[:, 0]
            IIdex = th.index_select(th.nonzero(
                iIdx2Ct == cid).squeeze(), 0, lIdx2IIdx)
            # lIdx2wIdx that directely maps current idx lIdx -> ray index in the world
            lIdx2WIdx = th.index_select(
                iIdx2wIdx[iIdx2Ct == cid], 0, lIdx2IIdx)

            if debug:
                img = np.zeros((RAY_SIZE, 3))
                img[lIdx2WIdx.tolist()] = np.array((1, 0, 0))
                full_idx = intersection_map[:,
                           0][intersection_map[:, 1] == cid]
                img_full = np.zeros((RAY_SIZE, 3))
                img_full[full_idx.tolist()] = np.array((1, 0, 0))
                pred_rgb_mask = func_crop(img, full_idx, H, W, square=True)
                im = Image.fromarray(
                    rearrange((img_full * 255).astype(np.uint8), '(H W) D -> H W D', H=H))
                im.save(f"full_test_c{cid}.png")
                im = Image.fromarray(
                    rearrange((img * 255).astype(np.uint8), '(H W) D -> H W D', H=H))
                im.save(f"sparse_test_c{cid}.png")
                im = Image.fromarray((pred_rgb_mask * 255).astype(np.uint8))
                im.save(f"sparse_crop_test_c{cid}.png")

            #
            # infer properties by NeRFs
            # TODO: use position p as prior input for each objets' rgbs
            # sigmas, rgbs, normals
            _nerf = nerf(
                xyzs_o,
                dirs_o,
                l=light_d, ratio=ambient_ratio, shading=shading)
            # collect results
            sigmas_w.append(_nerf[0])
            rgbs_w.append(_nerf[1])
            normals_w.append(_nerf[2])
            # composite the rays_idx_o in image size for CUDA implementation
            # to world index
            rays_idx_o[:, 0] = lIdx2WIdx
            # record
            rays_idx_w.append(rays_idx_o)
            xyzs_os.append(xyzs_o)
            dirs_os.append(dirs_o)
            deltas_w.append(deltas_o)
            IIdxes.append(IIdex)
            rays_shift_indx = 0
            # scale
            # padding length
            delta = H * W - len(rays_idx_o)
            # then pad the rays -> the last dim = 0 will be skipped by CUDA
            rays_idx_o = F.pad(rays_idx_o, (0, 0, 0, delta))
            weights_sum, depth, image = nerf.raymarching.composite_rays_train(
                _nerf[0],
                _nerf[1],
                deltas_o,
                rays_idx_o)  # local index in rays_idx_o[:, 0]
            # (pred_rgb, pred_ws, losses)
            _ray_train = \
                dict(ws=weights_sum,
                     depth=depth,
                     img=image,
                     rays=rays_idx_o,
                     rays_shift_indx=rays_shift_indx
                     )
            # image, depth, weights_sum, mask = self.bg_color(rays_o_o[iIdx2Ct==cid],       rays_d_o[iIdx2Ct==cid], _ray_train['ws'],
            #     _ray_train['img'], _ray_train['depth'],
            #     fars_o[iIdx2Ct==cid],  nears_o[iIdx2Ct==cid], disable_background)
            image = _ray_train['img'] + \
                    (1 - _ray_train['ws']).unsqueeze(-1) * bg_color
            intermediate_results = dict(
                xyzs=xyzs_o,
                sigmas=_nerf[0],
                img=image,
                ws=_ray_train['ws'],
                wIdxes=intersection_map[:,
                       0][intersection_map[:, 1] == cid],
                rays_shift_indx=_ray_train['rays_shift_indx'])
            if self.training:
                _, _, _losses = node.get_single_node_losses(
                    (data['dir'], B, data['H'], data['W']),
                    # intersected rays' categories -> iIdx2CT
                    self.diffusion,
                    intermediate_results,
                    crop=self.train_with_crop,
                    compact=False,
                    use_local_loss=self.use_local_loss
                )
                local_losses.append(_losses)
            image = image.reshape(
                B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            weights_sum = weights_sum.reshape(B, 1, H, W)
            depth = depth.reshape(
                B, H, W)
            # record the uncroped
            results[f'local_{cid}_image'] = image
            results[f'local_{cid}_weights_sum'] = weights_sum
            results[f'local_{cid}_depth'] = depth

        # global rendering
        sigmas_w = th.cat(sigmas_w)
        rgbs_w = th.cat(rgbs_w)
        normals_w = th.cat(normals_w) if None not in normals_w else None
        deltas_w = th.cat(deltas_w)
        pt_nums = [i.size(0) for i in xyzs_os]
        rays_idx_w = th.cat(rays_idx_w)
        IIdxes = th.cat(IIdxes)

        # correspond to local ray indexes
        rays_d_o = th.index_select(rays_d_o, 0, IIdxes)
        rays_d_w = th.index_select(rays_d_w, 0, IIdxes)
        rays_o_o = th.index_select(rays_o_o, 0, IIdxes)
        rays_o_w = th.index_select(rays_o_w, 0, IIdxes)
        nears_w = th.index_select(nears_w, 0, IIdxes)
        fars_w = th.index_select(fars_w, 0, IIdxes)
        nears_o = th.index_select(nears_o, 0, IIdxes)
        fars_o = th.index_select(fars_o, 0, IIdxes)
        # update rays
        rays_idx_w[0, 1] = 0
        for i in range(1, rays_idx_w.size(0)):
            rays_idx_w[i, 1] = rays_idx_w[i - 1, 1] + rays_idx_w[i - 1, 2]
            #
        xyzs_o = th.cat(xyzs_os)
        dirs_o = th.cat(dirs_os)
        xyzs_w, dirs_w = world2object(
            xyzs_o,
            dirs_o,
            th.cat([repeat(poses[0][cid], 'D -> B D', B=pt_nums[cid])
                    for cid in range(len(pt_nums))]),
            th.cat([theta_ys[0][cid].repeat(1, pt_nums[cid]).squeeze()
                    for cid in range(len(pt_nums))]),
            th.cat([repeat(dims[0][cid], 'D -> B D', B=pt_nums[cid])
                    for cid in range(len(pt_nums))]),
            inverse=True)
        xyzs_w = xyzs_w.squeeze()
        dirs_w = dirs_w.squeeze()
        # object to world
        # # use t = ||r-o||/||d|| to calculate the unit distance scale
        # where we use the first point sample on each ray for r
        lIdx2Ptidxes = [list(range(r[1], r[1] + r[2])) for r in rays_idx_w]
        scale_t = ((fars_w - nears_w) / \
                   th.norm(rays_d_w, dim=-1)) / \
                  ((fars_o - nears_o) / \
                   th.norm(rays_d_o, dim=-1))

        deltas_w = th.cat([deltas_w[idx] * scale_t[i]
                           for i, idx in enumerate(lIdx2Ptidxes)])
        # map the inferred properties back to the original place
        # merge the same rays(same origins and same directions) through different objects
        #
        # merging indxes of duplicating elements in IIdx2uniIdx array
        # 1. gather all duplicated intersected ray indexes
        res = get_common_rays((rays_d_w, rays_o_w))

        if len(res) != 0:
            # 2. rearranged intersected ray indexes -> concantenate(changed + unchanged)
            _ = set(chain(*res))
            # left unchanged local ray indexes
            left_idx = list(set(range(rays_d_w.size(0))) - _)
            rays_idx_w_ = th.zeros((len(res) + len(left_idx), 3),
                                   device=device, dtype=th.int)
            # 3. update the ray_index, llist2Ptidxes accordingly
            rays_idx_w_[0, 1] = 0
            lIdx2Ptidxes_ = []
            lIdxes = []
            for i in range(len(rays_idx_w_)):
                # iterate the combined rays
                if i < len(res):
                    lIdx = res[i][0]
                    # to world idxes
                    rays_idx_w_[i, 0] = rays_idx_w[lIdx, 0]
                    # rays_idx_w accessed by local ray indexes
                    rays_idx_w_[i, 2] = th.sum(
                        th.stack([rays_idx_w[j, 2] for j in res[i]]))
                    t = []
                    # gather point indexes in the original point indexes
                    for j in res[i]:
                        t.extend(lIdx2Ptidxes[j])
                    lIdx2Ptidxes_.append(t)
                    lIdxes.append(lIdx)
                else:
                    lIdx = left_idx[i - len(res)]
                    rays_idx_w_[i, 0] = rays_idx_w[lIdx, 0]
                    rays_idx_w_[i, 2] = rays_idx_w[lIdx, 2]
                    lIdx2Ptidxes_.append(lIdx2Ptidxes[lIdx])
                    lIdxes.append(lIdx)
                if i > 0:
                    rays_idx_w_[i, 1] = rays_idx_w_[
                                            i - 1, 1] + rays_idx_w_[i - 1, 2]
            #
            # 4. update the global propeties accordingly after reordering
            rays_idx_w = rays_idx_w_
            lIdx2Ptidxes = lIdx2Ptidxes_
            ptIdxes = list(chain(*lIdx2Ptidxes))

            sigmas_w = sigmas_w[ptIdxes]
            rgbs_w = rgbs_w[ptIdxes]
            deltas_w = deltas_w[ptIdxes]
            xyzs_w = xyzs_w[ptIdxes]
            dirs_w = dirs_w[ptIdxes]
            rays_o_w = rays_o_w[lIdxes]
            rays_d_w = rays_d_w[lIdxes]

            # fix the problem: lIdx2Ptidxes -> can't be used again since it has already mapped previously
            lIdx2Ptidxes = [list(range(r[1], r[1] + r[2])) for r in rays_idx_w]
            # test
            i_ray_mask = [np.full(len(list_), i)
                          for i, list_ in enumerate(lIdx2Ptidxes)]
            ptidx2lIdx = np.take(list(chain(*i_ray_mask)), ptIdxes)
            # sort sampling points (xyzs_w, dirs_w, deltas_w, rays_idx_w) based on time distances for each ray
            zvals_w = th.norm(xyzs_w - rays_o_w[ptidx2lIdx], dim=-1) / \
                      th.norm(rays_d_w[ptidx2lIdx], dim=-1)
            # sort sampling points for each ray based on zvals_w

            map_ = [th.argsort(zvals_w[idx]) for idx in lIdx2Ptidxes]

            # apply the mapping indexes to sort sampling points
            deltas_w = th.cat([th.index_select(deltas_w[idxes], 0, map_[i])
                               for i, idxes in enumerate(lIdx2Ptidxes)])
            rgbs_w = th.cat([th.index_select(rgbs_w[idxes], 0, map_[i])
                             for i, idxes in enumerate(lIdx2Ptidxes)])
            sigmas_w = th.cat([th.index_select(sigmas_w[idxes], 0, map_[i])
                               for i, idxes in enumerate(lIdx2Ptidxes)])
            xyzs_w = th.cat([th.index_select(xyzs_w[idxes], 0, map_[i])
                             for i, idxes in enumerate(lIdx2Ptidxes)])
            dirs_w = th.cat([th.index_select(dirs_w[idxes], 0, map_[i])
                             for i, idxes in enumerate(lIdx2Ptidxes)])

        if debug:
            img = np.zeros((RAY_SIZE, 3))
            img[rays_idx_w[:, 0].tolist()] = np.array((1, 0, 0))
            full_idx = intersection_map[:, 0]
            img_full = np.zeros((RAY_SIZE, 3))
            img_full[full_idx.tolist()] = np.array((1, 0, 0))
            pred_rgb_mask = func_crop(img, full_idx, H, W, square=True)
            # img[lIdx2WIdx[lIdx2Ct==1].tolist()] = np.array((0, 0, 1))
            im = Image.fromarray(
                rearrange((img_full * 255).astype(np.uint8), '(H W) D -> H W D', H=H))
            # im_ = th.tensor(rearrange(img , '(H W) D -> H W D', H=H))
            im.save(f"global_full_test.png")
            # im = Image.fromarray(
            # rearrange((img * 255).astype(np.uint8), '(H W) D -> H W D', H=H))
            # im.save(f"global_sparse_test.png")
            im = Image.fromarray((pred_rgb_mask * 255).astype(np.uint8))
            im.save(f"global_sparse_crop_test.png")

        # composite rays in the world
        # convert world indexes and pad rays
        delta = H * W - len(rays_idx_w)
        rays_idx_w = F.pad(rays_idx_w, (0, 0, 0, delta))
        rgbs_w, sigmas_w = self.global_forward(xyzs_w, dirs_w, rgbs_w, sigmas_w, with_residual=self.with_residual,
                                               step=step)
        # update the global rgbs_w based on the global ray positions
        weights_sum, depth, image = composite_rays_train(
            sigmas_w, rgbs_w, deltas_w, rays_idx_w)
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        res = dict(image=image)
        image = image.reshape(
            B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        weights_sum = weights_sum.reshape(B, 1, H, W)
        depth = depth.reshape(
            B, H, W)
        # idx -> world ray index -> to composite image
        # results['global_idx'] = im_
        results['global_weights_sum'] = weights_sum
        results['global_depth'] = depth
        results['global_image'] = image
        res['wIdxes'] = intersection_map[:, 0]
        if self.training:
            if self.use_global_loss:
                global_losses = self.get_global_losses(
                    (data['dir'], B, H, W),
                    res, crop=self.train_with_crop)
                # idx -> world ray index -> to composite image
                results['global_losses'] = global_losses
            # if self.use_local_loss:
            results['local_losses'] = th.stack(local_losses).mean()
        return results

    def local_render(self, data, **kwargs):
        results = dict()
        # [B, N, 3] # ([1, 4096, 3])
        rays_o, rays_d = data['rays_o'], data['rays_d']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
        if self.cfg.optim.start_shading_iter is None or self.train_step < self.cfg.optim.start_shading_iter:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            shading = 'lambertian'
            ambient_ratio = 0.1
        # Will be used if bg_radius <= 0
        bg_color = torch.rand((B * N, 3), device=rays_o.device)

        for i, (idx, node) in enumerate(self.dict_id2classnode.items()):
            outputs = node.render(rays_o.clone(), rays_d.clone(), staged=False, perturb=True, bg_color=bg_color,
                                  ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True)
            render_res = dict(
                xyzs=outputs['xyzs'],
                sigmas=outputs['sigmas'],
                img=outputs['image'],
                ws=outputs['weights_sum'],
                rays=None
            )
            loss = node.get_single_node_losses(
                (data['dir'], B, data['H'], data['W']),
                self.diffusion, render_res, crop=False)
            results[f'local_{i}_image'] = loss[0]
            results[f'local_{i}_ws'] = loss[1]
            results[f'local_{i}_losses'] = loss[2]
        local_losses = 0
        for key in results.keys():
            if 'loss' in key:
                local_losses += results[key]
        results['local_losses'] = local_losses
        return results

    def get_global_losses(self, data, outputs, crop=False):
        B, H, W = data[1:]
        # text embeddings
        wIdxes = outputs['wIdxes']
        img = outputs['image'].contiguous()
        if self.cfg.guide.append_direction:
            dirs = data[0]  # [B,]
            text_z = self.text_z[dirs]
        else:
            text_z = self.text_z
        pred_rgb = img.reshape(
            B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if crop:
            pred_rgb = func_crop(img, wIdxes, H, W, square=True)
            pred_rgb = rearrange(pred_rgb, 'H W D -> 1 D H W')

        # add bg_color
        if self.sjc:
            loss_guidance = self.diffusion.train_step_sjc(text_z, pred_rgb)
        else:
            loss_guidance = self.diffusion.train_step(
                text_z, pred_rgb, guidance_scale=self.cfg.optim.lambda_global)
        return loss_guidance


class ClassNode(nn.Module):
    def __init__(self, cfg, c_id, pose=None, theta_y=None, dim=None,
                 nerf=None,) -> None:
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        # Use NeRFNetwork
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.optim_cfg = cfg.optim
        self.render_cfg = cfg.render
        self.guide_cfg = cfg.guide
        self.nerf = NeRFNetwork(cfg.render).to(self.device)
        self.id = c_id
        self.sjc = cfg.guide.sjc
        if cfg.guide.shape_path is not None:
            self.shape_prior = ShapeLoss(cfg.guide)
        else:
            self.shape_prior = None
        self.sparsity_prior = sparsity_loss
        self.text_z = None
        self.obj_id = None
        self.cfg = cfg
        self.sjc = cfg.guide.sjc
        # TODO from cfg
        self.lambda_sparsity = 1.0
        self.pose = pose
        self.theta_y = theta_y
        self.dim = dim


    def transform(self, mat_trans):
        pass

    def scale(self, mat_scale):
        pass

    # TODO : Add object specifically
    def get_single_node_losses(self, data, diffusion, intermediate_results,
                               crop=False, compact=False, use_local_loss=True):
        # intermediate results: ['xyzs', 'sigmas', 'image', 'weights_sum']
        # data:['dir', 'B', 'H', 'W',] -> H, W is the orignal feature map sizes
        B, H, W = data[1:]
        xyzs = intermediate_results['xyzs']
        sigmas = intermediate_results['sigmas']
        img = intermediate_results['img']
        ws = intermediate_results['ws']
        if crop:
            wIdxes = intermediate_results['wIdxes']  # wo
            rays_shift_indx = intermediate_results['rays_shift_indx']

        full_img = img
        full_ws = ws

        pred_rgb = full_img.reshape(
            B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_ws = full_ws.reshape(B, 1, H, W)

        if crop:
            assert B == 1
            pred_rgb = func_crop(img, wIdxes, H, W, square=True)
            pred_ws = func_crop(ws[:, None], wIdxes, H, W, square=True)
            pred_rgb = rearrange(pred_rgb, 'H W D -> 1 D H W')
            pred_ws = rearrange(pred_ws, 'H W D -> 1 D H W')

        if self.cfg.guide.append_direction:
            dirs = data[0]  # [B,]
            text_z = self.text_z[dirs]
        else:
            text_z = self.text_z

        losses = 0
        if use_local_loss:
            if self.sjc:
                losses += diffusion.train_step_sjc(text_z, pred_rgb)
            else:
                losses += diffusion.train_step(text_z, pred_rgb, self.cfg.optim.lambda_local)
            if self.shape_prior is not None:
                losses += self.shape_prior(xyzs, sigmas)
        losses += self.optim_cfg.lambda_sparsity * self.sparsity_prior(pred_ws)
        return pred_rgb, pred_ws, losses

    def render(self, rays_o, rays_d, **kwargs):
        return self.nerf.render(rays_o, rays_d, **kwargs)

    def set_spatial_info(self, obj_id, pose, theta_y, dim):
        if self.obj_id is not None:
            assert self.obj_id == obj_id
        self.obj_id = obj_id
        self.pose = pose
        self.theta_y = theta_y
        self.dim = dim




