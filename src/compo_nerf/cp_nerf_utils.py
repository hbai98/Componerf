import torch.nn as nn
import numpy as np
import torch as th
import torch

from einops import rearrange, repeat
from numpy import linalg as LA
from sklearn.metrics.pairwise import euclidean_distances

# refer to https://github.com/princeton-computational-imaging/neural-scene-graphs

# TODO: for both the background and the object nodes.

def cal_plane_ray(data, planes, id_planes, near, method='planes'):
    """ Ray-Plane intersection for given planes in the scene
    Args:
        data: the ray sample dict including 'H', 'W', 'rays_o', 'rays_d', 'dir', and 'fixed_viewpoint'.
        planes: first plane position, plane normal and distance between planes
        id_planes: ids of used planes
        near: distance between camera pose and first intersecting plane
        method: Method used
    Returns:
        pts: [N_rays, N_samples+N_importance] - intersection points of rays and selected planes
        z_vals: position of the point along each ray respectively
    """
    rays_o, rays_d = data['rays_o'], data['rays_d']  # [B, N, 3]
    N = rays_o.shape[0]  # num rays

    pass


def box_pts(rays, pose, theta_y=None, dim=None, one_intersec_per_ray=False):
    """gets ray-box intersection points in world and object frames in a sparse notation
    Args:
        rays: ray origins and directions, [[N_rays, 3], [N_rays, 3]]
        pose: object positions in world frame for each ray, [N_rays, N_obj, 3]
        theta_y: rotation of objects around world y axis, [N_rays, N_obj, 3]
        dim: object bounding box dimensions [N_rays, N_obj, 3]
        one_intersec_per_ray: If True only the first interesection along a ray will lead to an
        intersection point output
    Returns:
        the world frame:
        [
            pts_box_w: box-ray intersection points given in the world frame (in, out for rays' intersections on the boundaries)
            viewdirs_box_w: view directions of each intersection point in the world frame
            z_vals_w_in: integration step in the world frame
            z_vals_w_out
        ]
        the object frame:
        [
            `pts_box_o`: box-ray intersection points given in the respective object frame
            viewdirs_box_o: view directions of each intersection point in the respective object frame
            z_vals_o_in: integration step for scaled rays in the object frame
            z_vals_o_out
        ]
        intersection_map: mapping of points, viewdirs and z_vals to the specific rays and objects at the intersection
    """
    rays_o, rays_d = rays
    # Transform each ray into each object frame
    rays_o_o, dirs_o = world2object(rays_o, rays_d, pose, theta_y, dim)
    rays_o_o = th.squeeze(rays_o_o)  # [N obj 3]
    dirs_o = th.squeeze(dirs_o)

    # Get the intersection with each Bounding Box
    z_ray_in_o, z_ray_out_o, intersection_map = ray_box_intersection(
        rays_o_o, dirs_o)  # z : 1D intersection points
    # intersection_map-> None check
    if intersection_map is None:
        return None
    idx_im = (*intersection_map.T,)  # for indexing matrixes
    if z_ray_in_o is not None:
        # Calculate the intersection points for all box in all object frames -> object frames
        pts_box_in_o = rays_o_o[idx_im] + \
                       z_ray_in_o[:, None] * dirs_o[idx_im]  # [N_sel, 3]
        # Transform the intersection points for each box in world frame
        pts_box_in_w, _ = world2object(pts_box_in_o,
                                       None,
                                       pose[idx_im],
                                       theta_y[idx_im],
                                       dim[idx_im],
                                       inverse=True)
        pts_box_in_w = th.squeeze(pts_box_in_w)  # [N_sel, 3]
        # Get all intersecting rays in unit length and the corresponding z_vals
        # in_w means in world frame
        rays_o_in_w = repeat(rays_o, 'B N -> B O N', O=pose.shape[1])[idx_im]
        rays_d_in_w = repeat(rays_d, 'B N -> B O N',
                             O=pose.shape[1])[idx_im]  # [N_sel, 3]
        # Account for non-unit length rays direction
        z_vals_in_w = th.norm(pts_box_in_w - rays_o_in_w,
                              dim=1) / th.norm(rays_d_in_w, dim=-1)

        # TODO: implement this for the background node.
        if one_intersec_per_ray:
            raise NotImplementedError()
            # Get just nearest object point on a single ray
            z_vals_in_w, intersection_map, first_in_only = get_closest_intersections(z_vals_in_w,
                                                                                     intersection_map,
                                                                                     N_rays=rays_o.shape[0],
                                                                                     N_obj=theta_y.shape[1])
            # Get previous calculated values just for first intersections
            z_ray_in_o = tf.gather_nd(z_ray_in_o, first_in_only)
            z_ray_out_o = tf.gather_nd(z_ray_out_o, first_in_only)
            pts_box_in_o = tf.gather_nd(pts_box_in_o, first_in_only)
            pts_box_in_w = tf.gather_nd(pts_box_in_w, first_in_only)
            rays_o_in_w = tf.gather_nd(rays_o_in_w, first_in_only)
            rays_d_in_w = tf.gather_nd(rays_d_in_w, first_in_only)

        # Get the far intersection points and integration steps for each ray-box intersection in world and object frames
        pts_box_out_o = rays_o_o[idx_im] + \
                        z_ray_out_o[:, None] * dirs_o[idx_im]
        pts_box_out_w, _ = world2object(pts_box_out_o,
                                        None,
                                        pose[idx_im],
                                        theta_y[idx_im],
                                        dim[idx_im],
                                        inverse=True)
        pts_box_out_w = th.squeeze(pts_box_out_w)
        z_vals_out_w = th.norm(pts_box_out_w - rays_o_in_w,
                               dim=1) / th.norm(rays_d_in_w, dim=-1)

        # Get viewing directions for each ray-box intersection
        viewdirs_box_o = dirs_o[idx_im]
        viewdirs_box_w = 1 / th.norm(rays_d_in_w, dim=1)[:, None] * rays_d_in_w

    else:
        # In case no ray intersects with any object return empty lists
        # z_vals_in_w = z_vals_out_w = []
        pts_box_in_w = pts_box_in_o = []
        viewdirs_box_w = viewdirs_box_o = []
        # z_ray_out_o = z_ray_in_o = []
    return (rays_o_in_w, viewdirs_box_w, z_vals_in_w, z_vals_out_w), \
           (rays_o_o[idx_im], viewdirs_box_o, z_ray_in_o, z_ray_out_o), \
           intersection_map


def ray_box_intersection(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected
    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary
    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified
    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
        idx_im: the splited indexes for pytorch tensor indexing
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = th.ones_like(ray_o) * -1.  # constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = th.ones_like(ray_o)  # constant([1., 1., 1.])

    inv_d = th.reciprocal(ray_d)

    t_min = (aabb_min - ray_o) * inv_d  # element-wise multiplication
    t_max = (aabb_max - ray_o) * inv_d  # [rays, boxes, 3]

    t0 = th.min(t_min, t_max)
    t1 = th.max(t_min, t_max)

    t_near = th.max(th.max(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = th.min(th.min(t1[..., 0], t1[..., 1]), t1[..., 2])  # [rays, boxes]

    # Check if rays are inside boxes
    intersection_map = th.nonzero(t_far > t_near)
    idx_im = (*intersection_map.T,)  # for indexing matrixes
    # Check that boxes are in front of the ray origin
    positive_far = (*th.nonzero(t_far[idx_im] > 0).T,)
    intersection_map = intersection_map[positive_far]
    idx_im = (*intersection_map.T,)  # for indexing matrixes

    if not intersection_map.shape[0] == 0:
        z_ray_in = t_near[idx_im]
        z_ray_out = t_far[idx_im]
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def world2object(pts, dirs, pose, theta_y=None, dim=None, inverse=False):
    """Transform points given in world frame into N_obj object frames
    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim
    Args:
        pts: N_pts times 3D points given in world frame, [N_pts, 3]
        dirs: Corresponding 3D directions given in world frame, [N_pts, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        inverse: if true pts and dirs should be given in object frame and are transofmed back into world frame, bool
            For inverse: pts, [N_pts, N_obj, 3]; dirs, [N_pts, N_obj, 3]
    Returns:
        pts_w: 3d points transformed into object frame (world frame for inverse task)
        dir_w: unit - 3d directions transformed into object frame (world frame for inverse task)
    """
    #  Prepare args if just one sample per ray-object
    #  or world frame only

    if len(pts.shape) == 3:
        # [batch_rays, n_obj,  xyz]
        n_sample_per_ray = pts.size[1]
        pose = pose.repeat(n_sample_per_ray)
        if dim is not None:
            dim = dim.repeat(n_sample_per_ray)
        if len(dirs.shape) == 2:
            dirs = dirs.repeat(n_sample_per_ray)
        pts = rearrange(pts, 'B N D -> (B N) D')  # [N_pts, 1, 3] ->[N_pts, 3]

    # # Shift the object reference point to the middle of the bbox (vkitti2 specific)
    # if inverse:
    #     y_shift = th.tensor([0., -1., 0.])[None, :].to(device)
    # else:
    #     y_shift = th.tensor([0., -1., 0.])[None, None, :].to(device) * \
    #           (dim[..., 1] / 2)[..., None]

    # pose_w = pose + y_shift
    # Describes the origin of the world system w in the object system o
    t_w_o = rotate_yaw(-pose, theta_y)

    if not inverse:  # means world -> object
        N_obj = theta_y.shape[1]
        # [N_pts, 3] -> [N_pts, n_obj, 3]
        pts_w = repeat(pts, 'B N -> B n N', n=N_obj)
        # [N_pts, 3] -> [N_pts, n_obj, 3]
        dirs_w = repeat(dirs, 'B N -> B n N', n=N_obj)

        # Rotate coordinate axis
        # TODO: Generalize for 3d roaations
        pts_o = rotate_yaw(pts_w, theta_y) + t_w_o
        dirs_o = rotate_yaw(dirs_w, theta_y)

        # Scale rays_o_v and rays_d_v for box [[-1.,1], [-1.,1], [-1.,1]]
        if dim is not None:
            pts_o = scale_frames(pts_o, dim)
            dirs_o = scale_frames(dirs_o, dim)

        # Normalize direction
        dirs_o = dirs_o / th.norm(dirs_o, dim=3)[..., None, :]
        return [pts_o, dirs_o]

    else:  # object -> world
        pts_o = pts[None, :, None, :]
        dirs_o = dirs
        if dim is not None:
            pts_o = scale_frames(pts_o, dim[None, ...], inverse=True)
            if dirs is not None:
                dirs_o = scale_frames(dirs_o[None, :, None, :], dim[None, ...], inverse=True)

        pts_o = pts_o - t_w_o
        pts_w = rotate_yaw(pts_o, -theta_y)[0, :]

        if dirs is not None:
            dirs_w = rotate_yaw(dirs_o, -theta_y)
            # Normalize direction
            dirs_w = dirs_w / th.norm(dirs_w, dim=-1)[..., None, :]
        else:
            dirs_w = None

        return [pts_w, dirs_w]


def rotate_yaw(p, yaw):
    """Rotates p with yaw in the given coord frame with y being the relevant axis and pointing downwards
    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        (object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3])
        yaw: Rotation angle
        (Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts])
    Returns:
        p: Rotated points [N_pts, N_frames, N_samples, 3]
    """
    # p of size [batch_rays, n_obj=n_frame, samples, xyz]

    if len(p.shape) < 4:
        # [N_pts, N_frames, 3] one sample
        p = p[..., None, :]

    c_y = th.cos(yaw)[..., None]
    s_y = th.sin(yaw)[..., None]

    p_x = c_y * p[..., 0] - s_y * p[..., 2]
    p_y = p[..., 1]
    p_z = s_y * p[..., 0] + c_y * p[..., 2]

    return th.concat([p_x[..., None], p_y[..., None], p_z[..., None]], axis=-1)


def scale_frames(p, sc_factor, inverse=False):
    """Scales points given in N_frames in each dimension [xyz] for each frame or rescales for inverse==True
    Args:
        p: Points given in N_frames frames [N_points, N_frames, N_samples, 3]
        sc_factor: Scaling factor for new frame [N_points, N_frames, 3]
        inverse: Inverse scaling if true, bool
    Returns:
        p_scaled: Points given in N_frames rescaled frames [N_points, N_frames, N_samples, 3]
    """
    # Take 150% of bbox to include shadows etc.
    dim = th.tensor([1., 1., 1.]).to(p.device) * sc_factor
    # dim = tf.constant([0.1, 0.1, 0.1]) * sc_factor

    half_dim = dim / 2
    scaling_factor = (1 / (half_dim + 1e-9))[:, :, None, :]

    if not inverse:
        p_scaled = scaling_factor * p
    else:
        p_scaled = (1 / scaling_factor) * p

    return p_scaled


def get_closest_intersections(z_vals_w, intersection_map, N_rays, N_obj):
    """Reduces intersections given by z_vals and intersection_map to the first intersection along each ray
    Args:
        z_vals_w: All integration steps for all ray-box intersections in world coordinates [n_intersections,]
        intersection_map: Mapping from flat array to ray-box intersection matrix [n_intersections, 2]
        N_rays: Total number of rays
        N_obj: Total number of objects
    Returns:
        z_vals_w: Integration step for the first ray-box intersection per ray in world coordinates [N_rays,]
        intersection_map: Mapping from flat array to ray-box intersection matrix [N_rays, 2]
        id_first_intersect: Mapping from all intersection related values to first intersection only [N_rays,1]
    """
    # Flat to dense indices
    # Create matching ray-object intersectin matrix with index for all z_vals
    id_z_vals = tf.scatter_nd(intersection_map, tf.range(
        z_vals_w.shape[0]), [N_rays, N_obj])
    # Create ray-index array
    id_ray = tf.cast(tf.range(N_rays), tf.int64)

    # Flat to dense values
    # Scatter z_vals in world coordinates to ray-object intersection matrix
    z_scatterd = tf.scatter_nd(intersection_map, z_vals_w, [N_rays, N_obj])
    # Set empty intersections to 1e10
    z_scatterd_nz = tf.where(tf.equal(z_scatterd, 0),
                             tf.ones_like(z_scatterd) * 1e10, z_scatterd)

    # Get minimum values along each ray and corresponding ray-box intersection id
    id_min = tf.argmin(z_scatterd_nz, axis=1)
    id_reduced = tf.concat(
        [id_ray[:, tf.newaxis], id_min[:, tf.newaxis]], axis=1)
    z_vals_w_reduced = tf.gather_nd(z_scatterd, id_reduced)

    # Remove all rays w/o intersections (min(z_vals_reduced) == 0)
    id_non_zeros = tf.where(tf.not_equal(z_vals_w_reduced, 0))
    if len(id_non_zeros) != N_rays:
        z_vals_w_reduced = tf.gather_nd(z_vals_w_reduced, id_non_zeros)
        id_reduced = tf.gather_nd(id_reduced, id_non_zeros)

    # Get intersection map only for closest intersection to the ray origin
    intersection_map_reduced = id_reduced
    id_first_intersect = tf.gather_nd(id_z_vals, id_reduced)[:, tf.newaxis]

    return z_vals_w_reduced, intersection_map_reduced, id_first_intersect


def combine_z(z_vals_bckg, z_vals_obj_w, intersection_map, N_rays, N_samples, N_obj, N_samples_obj=1):
    """Combines and sorts background node and all object node intersections along a ray
    Args:
        z_vals_bckg: integration step along each ray [N_rays, N_samples]
        z_vals_obj_w:  integration step of ray-box intersection in the world frame [n_intersects, N_samples_obj
        intersection_map: mapping of points, viewdirs and z_vals to the specific rays and objects at ray-box intersection
        N_rays: Amount of rays
        N_samples: Amount of samples along each ray
        N_obj: Maximum number of objects
        N_samples_obj: Number of samples per object
    Returns:
        z_vals:  [N_rays, N_samples + N_samples_obj*N_obj, 4]
        id_z_vals_bckg:
        id_z_vals_obj:
    """
    if z_vals_obj_w is None:
        z_vals_obj_w_sparse = th.zeros([N_rays, N_obj * N_samples_obj])
    else:
        z_vals_obj_w_sparse = tf.scatter_nd(intersection_map, z_vals_obj_w, [
            N_rays, N_obj, N_samples_obj])
        z_vals_obj_w_sparse = tf.reshape(
            z_vals_obj_w_sparse, [N_rays, N_samples_obj * N_obj])

    sample_range = tf.range(0, N_rays)
    obj_range = tf.repeat(tf.repeat(
        sample_range[:, tf.newaxis, tf.newaxis], N_obj, axis=1), N_samples_obj, axis=2)

    # Get ids to assign z_vals to each model
    if z_vals_bckg is not None:
        if len(z_vals_bckg.shape) < 2:
            z_vals_bckg = z_vals_bckg[tf.newaxis]
        # Combine and sort z_vals along each ray
        z_vals = tf.sort(
            tf.concat([z_vals_obj_w_sparse, z_vals_bckg], axis=1), axis=1)

        bckg_range = tf.repeat(
            sample_range[:, tf.newaxis, tf.newaxis], N_samples, axis=1)
        id_z_vals_bckg = tf.concat([bckg_range, tf.searchsorted(
            z_vals, z_vals_bckg)[..., tf.newaxis]], axis=2)
    else:
        z_vals = tf.sort(z_vals_obj_w_sparse, axis=1)
        id_z_vals_bckg = None

    # id_z_vals_obj = tf.concat([obj_range, tf.searchsorted(z_vals, z_vals_obj_w_sparse)], axis=2)
    id_z_vals_obj = tf.concat([obj_range[..., tf.newaxis],
                               tf.reshape(tf.searchsorted(z_vals, z_vals_obj_w_sparse), [
                                   N_rays, N_obj, N_samples_obj])[..., tf.newaxis]
                               ], axis=-1)

    return z_vals, id_z_vals_bckg, id_z_vals_obj


def get_common_rays(rays):
    _ = th.cat(rays, dim=-1)
    _, IIdx2uniIdx = th.unique(_, return_inverse=True, dim=0)
    _ = {}  # uniIdx2lIdxes
    for lIdx in range(IIdx2uniIdx.size(0)):
        uniIdx = IIdx2uniIdx[lIdx].item()
        if uniIdx not in _:
            _[uniIdx] = [lIdx]
        else:
            # use the first ray index
            _[uniIdx].append(lIdx)
    res = list({k: v for k, v in _.items() if len(v) > 1}.values())
    return res


def func_crop(vector, full_idx, H, W, square=True):
    # calculate coordinates for all nodes
    x = full_idx // W
    y = full_idx - x * W
    x1, x2 = x.min().item(), x.max().item()
    y1, y2 = y.min().item(), y.max().item()
    if square:
        # make it a square crop -> approximate
        delta_x, delta_y = (x2 - x1), (y2 - y1)
        x_more = delta_x > delta_y
        if x_more:
            shift = (delta_x - delta_y) // 2
            y1 -= shift
            y2 += shift
            y1 = max(0, y1)
            y2 = min(y2, W - 1)
        else:
            shift = (delta_y - delta_x) // 2
            x1 -= shift
            x2 += shift
            x1 = max(0, x1)
            x2 = min(x2, H - 1)
    pred_rgb = rearrange(vector, '(H W) D -> H W D', H=H)
    pred_rgb = pred_rgb[x1:x2 + 1, y1:y2 + 1, :]
    return pred_rgb


# https://gist.github.com/EricCousineau-TRI/cc2dc27c7413ea8e5b4fd9675050b1c0
def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = repeat(indices, "N K -> N K D", D=D)
    out = th.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out


def sigmoid(x):
    return 1/(1+torch.exp(-x))

def inv_sigmoid(x):
    return torch.log(x/(1-x))

def inv_dims(x, bound):
    return inv_sigmoid(x/bound)

def inv_poses(x, bound):
    return inv_sigmoid((bound + x)/(2*bound))

def split_poses_dims(bound, nbox):
    # split the space by octree
    octree_level = torch.log2(torch.tensor(nbox))//8 + 1
    num_voxel = torch.pow(8, octree_level).int().item()
    len_voxel = torch.pow(2, octree_level).int().item()
    start = bound/(2*len_voxel)
    
    xs = torch.linspace(start, bound-start, steps=len_voxel)
    ys = xs.clone()
    zs = xs.clone()
    x, y, z = torch.meshgrid(xs, ys, zs)
    poses = torch.stack([x, y, z])
    poses = rearrange(poses, 'n x y z -> (x y z) n ')
    ps_idx = torch.randint(0, num_voxel, (nbox, ))
    poses = torch.index_select(poses, 0, ps_idx)
    
    # avoid colision, calculate the smallest size
    min_dis = euclidean_distances(poses)
    min_dis = min_dis[min_dis!=0]
    min_dis = min_dis.min()
    min_dis = torch.tensor(min_dis)
    size = torch.sqrt(torch.pow(min_dis/2, 2)/2)
    size = size*2
    dims = torch.ones(nbox, 3)*size
    poses, dims = normalize_bound(bound, poses, dims)
    return poses, dims

def get_collision_loss(poses, dims):
    min_dis = shortest_pair_path(poses)
    

def shortest_pair_path(poses):
    from sklearn.metrics.pairwise import euclidean_distances
    dis = euclidean_distances(poses, poses)
    min_dis = dis[dis!=0].min()
    return min_dis

def normalize_bound(bound, poses, dims, box_scale=0.8):
    assert box_scale > 0 and box_scale <= 1, 'Box scale is not valid.'
    # poses = th.stack(poses)
    # dims = th.stack(dims)
    # normalize poses 
    device = poses.device
    B = poses.size(1) 
    assert th.all(dims > 0), 'The dimension should be positive values'
    # move the group obj to the center 
    
   # to center
    shift_g = th.zeros(B).to(device) - th.mean(poses.T, dim=-1) 
    poses = poses + shift_g # [n,3]
    
    b_u = poses + 0.5 * dims
    b_l = poses - 0.5 * dims
    delta = b_u.T.max(dim=-1)[0] - b_l.T.min(dim=-1)[0]
    # delta = th.max(b_u, dim=) 
    # scale the max boundary of the bounding-box group   
    max_dim = th.max(delta)
    scale = 2 * bound * box_scale / max_dim
    dims = dims*scale
    # scale the relative positions
    poses = poses*scale
    # inverse y and z [Might be some bug]
    # poses_ = poses.clone()
    # poses_[:, 1] = poses[:, -1]
    # poses_[:, -1] = poses[:, 1] 

    # dims_ = dims.clone()
    # dims_[:, 1] = dims[:, -1]
    # dims_[:, -1] = dims[:, 1]
    # shift based on the relative distance [dounding box sizes/ positions]
    return poses, dims
