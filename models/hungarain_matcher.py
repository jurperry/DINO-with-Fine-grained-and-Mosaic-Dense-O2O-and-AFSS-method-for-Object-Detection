import torch
import torchvision.ops.boxes as box_ops
from scipy.optimize import linear_sum_assignment
from utils import box_cxcywh_to_xyxy

# 匈牙利匹配函数, 找到一种排序使得总成本最小
# 每一个txt包含多个框的框类别和框的4个坐标信息数据, 读取后real_id一般是只有几个框
def HungarianMatch(real_id:tuple, real_bbox:tuple, pred_id:torch.Tensor, pred_bbox:torch.Tensor):
    '''
    real_id: 列表存储每张图的所有box对应的id: [tensor([1,0,1,2]), tensor([0,1,2])...]
    real_bbox: 列表存储每张图的所有box对应坐标: [(a1, 4), (a2, 4)...], 其中a1~an: batch_size=n, ai代表的是一张图中的gt框数量
    pred_id: (bs, num_queries, num_classes)
    pred_bbox: (bs, num_queries, 4)
    C: (bs, num_queries, ground_truth)
    '''
    batch_size, num_queries = pred_id.shape[0:2]
    
    # 逐元素映射sigmoid(), 不像softmax需要指定维度, 这里是deformable-detr的改进
    out_pro = pred_id.sigmoid().flatten(0, 1) # (bs*num_queries, num_calsses)
    out_bbox = pred_bbox.flatten(0, 1) #(bs*num_queries, 4)
    
    # real_id: 列表存储每张图的所有box对应的id: [tensor(1,0,1,2), tensor(0,1,2)...]
    # real_bbox:每张图的所有box对应的bbox: [(a1, 4), (a2, 4)...]
    target_id = torch.cat(real_id) # (num_id,)
    target_bbox = torch.cat(real_bbox, dim=0) # (num_bbox, bbox_xyxy=4)
    
    # cls_cost
    # 0~num_classes个概率值（num_classes个类别）, num_id = num_bbox = gt_num_in_batch
    alpha = 0.25
    gamma = 2.0
    neg_cost_class = (1 - alpha) * (out_pro ** gamma) * (-(1 - out_pro + 1e-8).log())
    pos_cost_class = alpha * ((1 - out_pro) ** gamma) * (-(out_pro + 1e-8).log())
    cls_cost = pos_cost_class[:, target_id] - neg_cost_class[:, target_id] # (bs*300, gt_num_in_batch)

    # l1_cost, 直接使用cxcywh进行匹配
    l1_cost = torch.cdist(out_bbox, target_bbox, p=1) # (bs*300, gt_num_in_batch)
    # giou_cost, 使用cxcywh转换为xyxy格式进行匹配
    giou_cost = -box_ops.generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(target_bbox)) # (bs*300, gt_num_in_batch)

    C = 2 * cls_cost + 5 * l1_cost + 2 * giou_cost
    C = C.view(batch_size, num_queries, -1) # C = (bs, num_queries=300, gt_num_in_batch)

    # sizes是一个列表，里面每个元素代表每张图的gt框个数
    sizes = [len(i) for i in real_id]
    
    # C.split(sizes, dim=-1)返回一个元组, 为batch_size长度的元组, 里面是被分割开的tensor
    # size就是每一张图片真实框个数, dim=-1就是对最后一维度进行分割
    # C.split(sizes, dim=-1) = (tensor((300, size_i), (300, size_j)...)  , tensor((300, size_i), (300, size_j)...) ...)
    indices = [linear_sum_assignment(c[i].detach().cpu().numpy()) for i, c in enumerate(C.split(sizes, dim=-1))]
    # indices：输出batch_size个最优化元组, 存储在列表中: [(row_id1, col_id1), (row_id2, col_id2),...]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] # 转换为tensor形式
