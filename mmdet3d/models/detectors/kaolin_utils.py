
import torch
from torchvision import utils
import torch.nn.functional as F

from kaolin.ops.spc.points import quantize_points, points_to_morton, morton_to_points, unbatched_points_to_octree
from kaolin.rep.spc import Spc
from kaolin.ops import spc
import kaolin.render.spc as spc_render



def get_rotation_matrix(tx, ty, tz):
    m_x = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_y = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_z = torch.zeros((len(tx), 3, 3)).to(tx.device)

    m_x[:, 1, 1], m_x[:, 1, 2] = tx.cos(), -tx.sin()
    m_x[:, 2, 1], m_x[:, 2, 2] = tx.sin(), tx.cos()
    m_x[:, 0, 0] = 1

    m_y[:, 0, 0], m_y[:, 0, 2] = ty.cos(), ty.sin()
    m_y[:, 2, 0], m_y[:, 2, 2] = -ty.sin(), ty.cos()
    m_y[:, 1, 1] = 1

    m_z[:, 0, 0], m_z[:, 0, 1] = tz.cos(), -tz.sin()
    m_z[:, 1, 0], m_z[:, 1, 1] = tz.sin(), tz.cos()
    m_z[:, 2, 2] = 1
    return torch.matmul(m_z, torch.matmul(m_y, m_x))



def octree_to_spc(octree):
    lengths = torch.tensor([len(octree)], dtype=torch.int32)
    _, pyramid, prefix = spc.scan_octrees(octree, lengths)
    points = spc.generate_points(octree, pyramid, prefix)
    pyramid = pyramid[0]  # remove batch
    return points, pyramid, prefix


def get_rays_p2(cam_para, curr_size):

    height, width = curr_size

    # create meshgrid to generate rays
    i, j = torch.meshgrid(torch.linspace(0.5, width - 0.5, width),
                        torch.linspace(0.5, height - 0.5, height))

    i = i.t().unsqueeze(0).to(cam_para)
    j = j.t().unsqueeze(0).to(cam_para)

    rays_d = torch.stack([(i - cam_para[0][2]) / cam_para[0][0],
                        (j - cam_para[1][2]) / cam_para[1][1],
                        torch.ones_like(i).expand(1, height, width)], -1)

    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = torch.zeros_like(rays_d)
    rays_full_dict = {'rays_o': rays_o, 'rays_d': rays_d}

    return rays_full_dict


def get_near_far(
    rays_o,
    rays_d,
    octree,
    scene_origin,
    scale,
    level,
    spc_data=None,
    visualize=False,
    ind=0,
    with_exit=False,
    return_pts=False,
    pcl_fea=None,
    mts_info=None,
):
    """
    'rays_o': ray origin in sfm coordinate system
    'rays_d': ray direction in sfm coordinate system
    'octree': spc
    'with_exit': set true to obtain accurate far. Default to false as this will perform aabb twice
    """
    # # Avoid corner cases. issuse in kaolin: https://github.com/NVIDIAGameWorks/kaolin/issues/490
    # rays_d = rays_d.clone() + 1e-7
    # rays_o = rays_o.clone() + 1e-7

    if spc_data is None:
        points, pyramid, prefix = octree_to_spc(octree)
    else:
        points, pyramid, prefix = (
            spc_data["points"],
            spc_data["pyramid"],
            spc_data["prefix"],
        )

    # transform origin from sfm to kaolin [-1, 1]
    rays_o_normalized = (rays_o - scene_origin) / scale

    rays_mask = torch.ones((rays_o.shape[0], )).to(rays_o)
    rays_fea = torch.zeros((rays_o.shape[0], 3)).to(rays_o)

    img_list = []

    fine_level = level - 3

    for level in range(level, fine_level - 1, -1):
        # print('current level:  ', level)

        rays_pid = torch.ones_like(rays_o_normalized[:, :1]) * -1
        rays_near = torch.zeros_like(rays_o_normalized[:, :1])
        rays_far = torch.zeros_like(rays_o_normalized[:, :1])

        # level = 11
        ray_index, pt_ids, depth_in_out = spc_render.unbatched_raytrace(
            octree,
            points,
            pyramid,
            prefix,
            rays_o_normalized,
            rays_d,
            level,
            return_depth=True,
            with_exit=with_exit,
        )
        # ray_index 记录了射线与所有level voxel的相交情况，值是rays的index
        # pt_ids 记录了射线与哪些voxel相交了，值是voxel的index


        ray_index = ray_index.long()
        if not with_exit:
            # if no exit, far will be the entry point of the last intersecting. This is an inaccurate far, but will be more efficient
            depth_in_out = torch.cat([depth_in_out, depth_in_out], axis=1)

        near_index, near_count = torch.unique_consecutive(ray_index, return_counts=True)
        # torch.unique_consecutive: 去重，得到发生碰撞的ray的编号和数量
        
        if ray_index.size()[0] == 0:
            print("[WARNING] batch has 0 intersections!!")
            if return_pts:
                return rays_near, rays_far, rays_pid
            else:
                return rays_near, rays_far

        near_inv = torch.roll(torch.cumsum(near_count, dim=0), shifts=1)
        # 先累加
        # torch.roll 这里是向量的数值往后滚动了1个长度，最后的值拿到第一位
        near_inv[0] = 0

        far_index, far_count = torch.unique_consecutive(torch.flip(ray_index, [0]), return_counts=True)
        far_inv = torch.roll(torch.cumsum(far_count, dim=0), shifts=1)
        far_inv[0] = 0
        far_inv = ((ray_index.size()[0] - 1) - far_inv).long()

        rays_pid[near_index] = pt_ids[near_inv].reshape(-1, 1).float()
        rays_near[near_index] = depth_in_out[near_inv, :1]
        rays_far[far_index] = depth_in_out[far_inv, 1:]

        valid = (rays_near) > 1e-4
        rays_near[~valid] = 0
        rays_far[~valid] = 0
        rays_pid[~valid] = -1

        near_points_sfm = rays_o + rays_d * rays_near * scale
        far_points_sfm = rays_o + rays_d * rays_far * scale

        # import ipdb; ipdb.set_trace()
        

        feature_index = rays_pid[..., 0] - pyramid[1, level]  # index from the latest level 
        feature_index[rays_mask == 0] = -1

        flag_1 = feature_index >= 0 
        flag_rays = feature_index[feature_index >= 0].long()

        rays_fea[flag_1] = mts_info[1][level][flag_rays]

        rays_mask[feature_index >= 0] = 0   # hit under current level

        # utils.save_image(img_recon, 'octree_img_folder/output_%06d.jpg' % level)

        if level == fine_level:
            img_recon = rays_fea.reshape((352, 640, 3)).permute(2,0,1)
            img_list.append(img_recon.clone())

    return img_list




def unbatched_pointcloud_to_spc_multiscale(pointcloud, level, features=None):
    r"""This function takes as input a single point-cloud - a set of continuous coordinates in 3D,
    and coverts it into a :ref:`Structured Point Cloud (SPC)<spc>`, a compressed octree representation where
    the point cloud coordinates are quantized to integer coordinates.
    Point coordinates are expected to be normalized to the range :math:`[-1, 1]`.
    If a point is out of the range :math:`[-1, 1]` it will be clipped to it.
    If ``features`` are specified, the current implementation will average features
    of points that inhabit the same quantized bucket.
    Args:
        pointclouds (torch.Tensor):
            An unbatched pointcloud, of shape :math:`(\text{num_points}, 3)`.
            Coordinates are expected to be normalized to the range :math:`[-1, 1]`.
        level (int):
            Maximum number of levels to use in octree hierarchy.
        features (optional, torch.Tensor):
            Feature vector containing information per point, of shape
            :math:`(\text{num_points}, \text{feat_dim})`.
    Returns:
        (kaolin.rep.Spc):
        A Structured Point Cloud (SPC) object, holding a single-item batch.
    """
    
    mtc_points = {}
    mtc_feat = {}

    points = quantize_points(pointcloud.contiguous(), level)
    mtc_points[level] = points

    # Avoid duplications if cells occupy more than one point
    unique, unique_keys, unique_counts = torch.unique(points.contiguous(), dim=0,
                                                      return_inverse=True, return_counts=True)
    # Create octree hierarchy
    morton, keys = torch.sort(points_to_morton(unique.contiguous()).contiguous())
    points = morton_to_points(morton.contiguous())
    octree = unbatched_points_to_octree(points, level, sorted=True)

    # Organize features for octree leaf nodes
    feat = None
    if features is not None:
        # Feature collision of multiple points sharing the same cell is consolidated here.
        # Assumes mean averaging
        feat_dtype = features.dtype
        is_fp = features.is_floating_point()

        # Promote to double precision dtype to avoid rounding errors
        feat = torch.zeros(unique.shape[0], features.shape[1], device=features.device).double()
        feat = feat.index_add_(0, unique_keys, features.double()) / unique_counts[..., None].double()
        if not is_fp:
            feat = torch.round(feat)
        feat = feat.to(feat_dtype)
        feat = feat[keys]

    # A full SPC requires octree hierarchy + auxilary data structures
    lengths = torch.tensor([len(octree)], dtype=torch.int32)   # Single entry batch
    spc_highest_level = Spc(octrees=octree, lengths=lengths, features=feat)


    mtc_feat[level] = feat

    for curr_level in range(level - 4, level):
        mtc_points[curr_level] = quantize_points(pointcloud.contiguous(), curr_level)

        unique, unique_keys, unique_counts = torch.unique(mtc_points[curr_level].contiguous(), dim=0,
                                                        return_inverse=True, return_counts=True)
        morton, keys = torch.sort(points_to_morton(unique.contiguous()).contiguous())

        feat_dtype = features.dtype
        is_fp = features.is_floating_point()

        # Promote to double precision dtype to avoid rounding errors
        feat = torch.zeros(unique.shape[0], features.shape[1], device=features.device).double()
        feat = feat.index_add_(0, unique_keys, features.double()) / unique_counts[..., None].double()
        if not is_fp:
            feat = torch.round(feat)
        mtc_feat[curr_level] = feat.to(feat_dtype)[keys]

    return spc_highest_level, [mtc_points, mtc_feat]