import torch
from torch import nn
from igl import read_obj


from src.latent_nerf.configs.train_config import GuideConfig
from src.latent_nerf.models.mesh_utils import MeshOBJ, MeshOBJOffset
from src.point_e.point_to_mesh import get_mesh
DELTA = 0.2

def ce_pq_loss(p, q, weight=None):
    def clamp(v, T=0.01):
        return v.clamp(T, 1 - T)

    ce = -1 * (p * torch.log(clamp(q)) + (1 - p) * torch.log(clamp(1 - q)))
    if weight is not None:
        ce *= weight
    return ce.sum()


def generate_shape_from_text(text):
    return


class ShapeLoss(nn.Module):
    def __init__(self, cfg:GuideConfig):
        super().__init__()
        self.cfg = cfg
        if self.cfg.shape_path is not None:
            v, _, _, f, _, _ = read_obj(self.cfg.shape_path, float)
        else:
            pointE_mesh = get_mesh(self.cfg.subtext)
            v, _, _, f, _, _ = read_obj(pointE_mesh, float)

        mesh = MeshOBJ(v, f)
        self.sketchshape = mesh.normalize_mesh(cfg.mesh_scale)

    def forward(self, xyzs, sigmas):
        mesh_occ = self.sketchshape.winding_number(xyzs)
        # print(mesh_occ)
        if self.cfg.proximal_surface > 0:
            weight = 1 - self.sketchshape.gaussian_weighted_distance(xyzs, self.cfg.proximal_surface)
        else:
            weight = None
        indicator = (mesh_occ > 0.5).float()
        nerf_occ = 1 - torch.exp(-DELTA * sigmas)
        nerf_occ = nerf_occ.clamp(min=0, max=1.1)
        loss = ce_pq_loss(nerf_occ, indicator, weight=weight)  # order is important for CE loss + second argument may not be optimized
        return loss


class MultiShapeLoss(nn.Module):
    def __init__(self, cfg:GuideConfig):
        super().__init__()
        self.cfg = cfg
        v, _, _, f, _, _ = read_obj(self.cfg.shape_path_a, float)
        mesh_a = MeshOBJOffset(v, f)
        self.sketchshape_a = mesh_a.normalize_mesh(0.3, offset=0.3)

        v, _, _, f, _, _ = read_obj(self.cfg.shape_path_b, float)
        mesh_b = MeshOBJOffset(v, f)
        self.sketchshape_b = mesh_b.normalize_mesh(0.5, offset=-0.2)
        self.meshocc_a = None
        self.meshocc_b = None

    def forward(self, xyzs, sigmas):
        if (self.meshocc_a is None) or (self.meshocc_b is None):
            self.init_mesh_occ(xyzs)

        mesh_occ_a = self.sketchshape_a.winding_number(xyzs)
        mesh_occ_b = self.sketchshape_b.winding_number(xyzs)
        # print(mesh_occ)
        if self.cfg.proximal_surface > 0:
            weight_a = 1 - self.sketchshape_a.gaussian_weighted_distance(xyzs, self.cfg.proximal_surface)
            weight_b = 1 - self.sketchshape_b.gaussian_weighted_distance(xyzs, self.cfg.proximal_surface)
            weight_a[weight_b > weight_a] = weight_b[weight_b > weight_a]
            weight = weight_a
        else:
            weight = None
        indicator = ((mesh_occ_a > 0.5) | (mesh_occ_b > 0.5)).float()
        nerf_occ = 1 - torch.exp(-DELTA * sigmas)
        nerf_occ = nerf_occ.clamp(min=0, max=1.1)
        loss = ce_pq_loss(nerf_occ, indicator, weight=weight)  # order is important for CE loss + second argument may not be optimized
        self.reset_mesh_occ()
        return loss

    def get_density(self, xyzs, sketchshape):
        mesh_occ_a = sketchshape.winding_number(xyzs)
        return (mesh_occ_a > 0.5).float()

    def init_mesh_occ(self, xyzs):
        self.meshocc_a = self.get_density(xyzs, self.sketchshape_a)
        self.meshocc_b = self.get_density(xyzs, self.sketchshape_b)

    def reset_mesh_occ(self):
        self.meshocc_a = None
        self.meshocc_b = None