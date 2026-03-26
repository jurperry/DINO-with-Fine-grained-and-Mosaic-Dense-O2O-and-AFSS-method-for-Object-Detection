import torch
from utils import box_xyxy_to_cxcywh

# 权重方程(概率函数), 生成33个权重(偏移分布所对应的权重分布)
# 传入的reg_max是一个数值, up与reg_scale是一个张量, 张量梯度截断了
def weighting_function(reg_max, up, reg_scale, deploy=False):
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        reg_max (int): Max number of the discrete bins. reg_max = 32, total_bins = 33
        up (Tensor): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W. up = a = 0.5, a = 1/2
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(reg_max/2)=0
                           and steeper weights at both ends. reg_scale = 4.0 = 1/c, c = 1/4
        deploy (bool): If True, uses deployment mode settings.

    Returns:
        Tensor: Sequence of Weighting Function.
    """
    if deploy:
        # 部署阶段全部都是数值, 不是张量, 最后再统一把数值转换为张量tensor
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    else:
        # 训练阶段全部都是张量
        upper_bound1 = abs(up[0]) * abs(reg_scale) # a/c, 是一个张量
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2 # 2a/c, 是一个张量
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2)) # (a/c + 1)**(2/(32-2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)] # 左侧15个值
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)] # 右侧15个值
        values = (
            [-upper_bound2] # 左侧最小值
            + left_values # 左侧15个值
            + [torch.zeros_like(up[0][None])] # 中值权重为0
            + right_values # 右侧15个值
            + [upper_bound2] # 右侧最大值
        ) # 一共 1+15+1+15+1 = 1+31+1=33个权重值
        return torch.cat(values, 0) # 拼接得到 (33, )的权重, 其值为W(n)/c = 4.0 * W(n)

# gt真实框转换, 不单独使用, 是bbox2distance的子函数
def translate_gt(gt, reg_max, reg_scale, up):
    """
    Decodes bounding box ground truth (GT) values into distribution-based GT representations.

    This function maps continuous GT values into discrete distribution bins, which can be used
    for regression tasks in object detection models. It calculates the indices of the closest
    bins to each GT value and assigns interpolation weights to these bins based on their proximity
    to the GT value.

    Args:
        gt (Tensor): Ground truth bounding box values, shape (n, 4) ->(N, ), N=n*4.
        reg_max (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Tensor): Controls the upper bounds of the Weighting Function.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - indices (Tensor): Index of the left bin closest to each GT value, shape (N, ).
            - weight_right (Tensor): Weight assigned to the right bin, shape (N, ).
            - weight_left (Tensor): Weight assigned to the left bin, shape (N, ).
    """
    gt = gt.reshape(-1) # 真实框相对于ref_points_initial的l,t,r,b偏移 (n, 4) ->(N, )
    # 相减原理: 真实框与ref_points_initial差值是确定的, 属于狄拉克delta分布P(ix)=1, ∑W(i)*P(i) = W(ix)*1 = tmp(l,t,r,b)
    function_values = weighting_function(reg_max, up, reg_scale) # (33, ) 一共33个权重值

    # Find the closest left-side indices for each value
    # W(i) - tmp(l, t, r, b) = W(i) - W(ix) 中小于0的就是权重函数中位于ix左侧的索引数, 从而得到ix左右两侧临近的整数
    # (1, 33) - (N, 1) ->(N, 33)
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1) # (N, 33)

    mask = diffs <= 0
    # 得到所有真实偏移分布点左侧的临近整数值(可能无效)
    closest_left_indices = torch.sum(mask, dim=1) - 1 # (N, )

    # Calculate the weights for the interpolation
    indices = closest_left_indices.float() # (N, )

    weight_right = torch.zeros_like(indices) # (N, )
    weight_left = torch.zeros_like(indices) # (N, )

    valid_idx_mask = (indices >= 0) & (indices < reg_max) # (N, ) 是一个True, False组成的一维张量
    valid_indices = indices[valid_idx_mask].long() # 把有效的l,t,r,b分布给拿到 (M, ), 有效分布的横坐标上, 真实偏移点左右两侧临近整数的左侧整数

    # Obtain distances
    left_values = function_values[valid_indices] # (M, )得到的是有效真实偏移点的左侧临近整数点对应的权重值
    right_values = function_values[valid_indices + 1] # (M, )得到的是有效真实偏移点的右侧临近整数点对应的权重值

    left_diffs = torch.abs(gt[valid_idx_mask] - left_values) # 有效偏移点对应权重与左侧临近点对应权重差值 (M, )
    right_diffs = torch.abs(right_values - gt[valid_idx_mask]) # 左侧临近点对应权重与有效偏移点对应权重差值 (M, )

    # Valid weights
    # 通过权重值对横坐标进行插值得到真实偏移点的近似横坐标ix, 对权重进行有效区域赋值
    # 相似三角原理 weight_left + weight_right = 1.0 
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    # Invalid weights (out of range)
    # 无效分布的处理
    # 左侧点小于0的处理, 全力监督bin0
    invalid_idx_mask_neg = indices < 0
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    # 左侧点大于32, 全力监督bin32
    invalid_idx_mask_pos = indices >= reg_max
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    # 返回得到的indices是每一个真实偏移在权重函数横坐标上的左侧最临近点的位置
    # 连续分布中的bin(k), bin(k+1)下, weight_right指的是bin(k+1)的权重, weight_left指的是bin(k)的权重
    # indices, weight_right, weight_left 维度均为(N, ), N = bs*num_gt*4 = n*4
    return indices, weight_right, weight_left

# 偏移距离分布distance转换为边界框bbox
# iter_ref_bbox的生成, 用于decoder输出的cxcywh训练值, 把ref_points_initial 与 pred_corners加和得到输出cxcywh值
def distance2bbox(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height. 
                         And points are normlized.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.(l, t, r, b)

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale) # 由于之前的权重值是W(n)/c, 因此要变成权重, 则需要对最后的distance * c, c=1/reg_scale = 1/4
    # ref_x1 = cx - 1/2*w, cx = points[...,0], 1/2*w = 0.5*points[...,2], ref_x1 - ref_left = x1, 
    # ref_x1的偏移量为ref_left = distance[...,0] * points[...,2]/reg_scale = left*w *c
    # distance量要经过c乘积变换后再乘以初始参考点ref_points_initial的 w,h值, w = points[..., 2], h = points[..., 3]
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale) # x1取值范围为ref_x1 ± 1.0
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale) # y1取值范围为ref_y1 ± 1.0
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale) # x2取值范围为ref_x2 ± 1.0
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale) # y2取值范围为ref_y2 ± 1.0

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return box_xyxy_to_cxcywh(bboxes)

# bbox变偏移分布distance, 损失函数使用, 基于真实框gt_bbox生成偏移距离分布distance
def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """
    Converts bounding box coordinates to distances from a reference point.

    Args:
        points (Tensor): (matched_gt, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Tensor): (matched_gt, 4) bounding boxes in "xyxy" format.
        reg_max (float): Maximum bin value. 32
        reg_scale (float): Controling curvarture of W(n). 4.0
        up (Tensor): Controling upper bounds of W(n). 0.5
        eps (float): Small value to ensure target < reg_max. 0.1

    Returns:
        Tensor: Decoded distances.
    """
    reg_scale = abs(reg_scale) 
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale # (matched_gt, )
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale # (matched_gt, )
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale # (matched_gt, )
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale # (matched_gt, )
    four_lens = torch.stack([left, top, right, bottom], -1) # 真实框相对于ref_points_inital的l,t,r,b偏移量, (matched_gt, ltrb=4)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()
