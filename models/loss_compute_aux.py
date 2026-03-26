import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops
from .hungarain_matcher import HungarianMatch
from utils import box_cxcywh_to_xyxy, box_iou
from .fine_grained import bbox2distance

# 总损失计算
def compute_train_loss(class_out, bbox_out, dn_outs, dec_out_pred_corners, dec_out_refs, 
                 traditional_logits, traditional_bboxes, fine_grained_out,
                 real_id:tuple, real_bbox:tuple, device, alpha=1.0, gamma=2.0, class_loss_methed='mal'):
    '''
    class_out: list[(bs, num_queries=300, num_classes) x (6 + 2)], 6xdecoder + 2xencoder
    bbox_out: list[(bs, num_queries=300, cxcywh) x (6 +2 )], 6xdecoder + 2xencoder

    dn_outs: 
        dn_outs['indices'] = dn_pos_idx # 损失计算用匹配索引indices (query_indices, target_indices)
        dn_outs['num_dn_groups'] = num_dn_groups
        dn_outs['dn_traditional_logits'] = dn_traditional_logits # (bs, dnq, num_classes)
        dn_outs['dn_traditional_bboxes'] = dn_traditional_bboxes # (bs, dnq, cxcywh=4)
        dn_outs['dn_dec_logits'] = dn_dec_logits   # (6, bs, dnq, num_classes)
        dn_outs['dn_dec_bboxes'] = dn_dec_bboxes   # (6, bs, dnq, cxcywh=4)
        dn_outs['dn_dec_corners'] = dn_dec_corners # (6, bs, dnq, 4*33)
        dn_outs['dn_dec_refs'] = dn_dec_refs       # (6, bs, dnq, cxcywh=4)

    dec_out_pred_corners: list[(bs, num_queries, 132) x 6]
    dec_out_refs: list[(bs, num_queries, 4) x 6]
    traditional_logits: (bs, num_queries, num_classes)
    traditional_bboxes: (bs, num_queries, cxcywh=4)
    fine_grained_out: 
            fine_grained_out['reg_scale'] = 4.0 无梯度
            fine_grained_out['up'] = 0.5 无梯度
    real_id: 一个批次内的真实id, tuple内存储tensor x bs, 其tensor.shape = (num_gti, )
    real_bbox: 一个批次内的真实框boxes, tuple内存储tensor x bs, 其tensor.shape = (num_gti, 4) 
    '''
    # 对真实标注进行list转换
    real_id = [id_tensor.to(device) for id_tensor in real_id]
    real_bbox = [bbox_tensor.to(device) for bbox_tensor in real_bbox]

    # encoder部分
    # tmp+ref的topk
    enc_class_logits0 = class_out[-2]
    enc_bbox_pred0 = bbox_out[-2]
    # init_box_proposals
    enc_class_logits1 = class_out[-1]
    enc_bbox_pred1 = bbox_out[-1]

    # 去噪部分所需预测值
    dn_dec_logits = dn_outs['dn_dec_logits']    # (6, bs, dnq, num_classes)
    dn_dec_bboxes = dn_outs['dn_dec_bboxes']    # (6, bs, dnq, cxcywh=4)
    dn_traditional_logits = dn_outs['dn_traditional_logits'] # (bs, dnq, num_classes)
    dn_traditional_bboxes = dn_outs['dn_traditional_bboxes'] # (bs, dnq, cxcywh=4)
    dn_dec_corners = dn_outs['dn_dec_corners']  # (6, bs, dnq, 4*33)
    dn_dec_refs = dn_outs['dn_dec_refs']        # (6, bs, dnq, cxcywh=4)

    # 解码器的层数的计算, 总输出层数就是6x decoder + 2x encoder
    num_decoder_layers = len(class_out) - 2
    
    indices_aux_list = [] # 5x decoder + 2x encoder + 1 traditonal decoder, 均为本层匹配值
    # 计算所有decoder层的辅助匹配与主匹配
    for i in range(num_decoder_layers):
        class_out_i = class_out[i] # (bs, num_queries=300, num_classes)
        bbox_out_i = bbox_out[i]  # (bs, num_queries=300, bbox=4)
        if i == num_decoder_layers -1 :
            # HungarianMatch函数返回的是batch_size大小的一个列表，其中包含元组 (query_indices, target_indices), 存放匹配数据
            main_indices = HungarianMatch(real_id, real_bbox, class_out_i, bbox_out_i)
        else:
            indices_dec_aux = HungarianMatch(real_id, real_bbox, class_out_i, bbox_out_i)
            indices_aux_list.append(indices_dec_aux)
    
    # 计算encoder层的辅助匹配
    indices_enc_aux0 = HungarianMatch(real_id, real_bbox, enc_class_logits0, enc_bbox_pred0)
    indices_enc_aux1 = HungarianMatch(real_id, real_bbox, enc_class_logits1, enc_bbox_pred1)
    indices_aux_list.append(indices_enc_aux0)
    indices_aux_list.append(indices_enc_aux1)

    # 计算traditional_decoder层的辅助匹配
    indices_traditional_dec_aux = HungarianMatch(real_id, real_bbox, traditional_logits, traditional_bboxes)
    indices_aux_list.append(indices_traditional_dec_aux)

    # 获得匹配的全局索引, 用于box_loss与local_loss计算, 对于5xdecoder + 2xencoder + 1xtraditional decoder 与main_decoder进行全局匹配
    global_indices = get_global_indices(main_indices, indices_aux_list)

    # 去噪部分索引, 既是本层索引, 也是全局索引, 因为去噪分支仅仅针对decoder设计, 且索引全部相同
    # 用于vfl_loss, bbox_loss, local_loss计算
    global_dn_indices = dn_outs['indices'] 

    # cls_loss, 使用自己本层索引
    if class_loss_methed == 'vfl':
        get_cls_loss = get_vfl_loss
    else:
        get_cls_loss = get_mal_loss
    # 非去噪部分
    #     # 主匹配(最后一个decoder)
    main_cls_loss = get_cls_loss(main_indices, class_out[num_decoder_layers -1], bbox_out[num_decoder_layers -1], 
                             real_id, real_bbox, device, alpha, gamma)
    
    # 辅助匹配 5x decoder + 2x encoder + 1 traditional decoder
    # decoder部分(前5层)
    dec_cls_loss_list = []
    for j in range(num_decoder_layers):
        if j == num_decoder_layers - 1:
            continue
        dec_cls_loss = get_cls_loss(indices_aux_list[j], class_out[j], bbox_out[j], real_id, real_bbox, device, alpha, gamma)
        dec_cls_loss_list.append(dec_cls_loss)
    # encoder部分
    enc_cls_loss0 = get_cls_loss(indices_aux_list[-3], enc_class_logits0, enc_bbox_pred0, real_id, real_bbox, device, alpha, gamma)
    enc_cls_loss1 = get_cls_loss(indices_aux_list[-2], enc_class_logits1, enc_bbox_pred1, real_id, real_bbox, device, alpha, gamma)
    # traditional_decoder部分
    tradition_dec_cls_loss = get_cls_loss(indices_aux_list[-1], traditional_logits, traditional_bboxes, 
                                      real_id, real_bbox, device, alpha, gamma)
    # 1 main + 5 dec + 2 enc + 1 traditional
    non_dn_cls_loss = [main_cls_loss] + dec_cls_loss_list + [enc_cls_loss0, enc_cls_loss1] + [tradition_dec_cls_loss]

    # 去噪部分, 使用本层索引, 但是去噪分支全部的索引相同, 也为全局索引
    # 主匹配(最后一个decoder)
    dn_main_cls_loss = get_cls_loss(global_dn_indices, dn_dec_logits[num_decoder_layers -1], dn_dec_bboxes[num_decoder_layers -1], 
                                real_id, real_bbox, device, alpha, gamma)
    # 辅助匹配 5x decoder + 1 traditional decoder
    # decoder部分
    dn_dec_cls_loss_list = []
    for k in range(num_decoder_layers):
        if k == num_decoder_layers - 1:
            continue
        dn_dec_cls_loss = get_cls_loss(global_dn_indices, dn_dec_logits[k], dn_dec_bboxes[k], real_id, real_bbox, device, alpha, gamma)
        dn_dec_cls_loss_list.append(dn_dec_cls_loss)
    # traditional_decoder部分
    dn_traditional_dec_cls_loss = get_cls_loss(global_dn_indices, dn_traditional_logits, dn_traditional_bboxes, 
                                           real_id, real_bbox, device, alpha, gamma)
    # 1 main_dn + 5 dn_dec + 1 dn traditional dec
    dn_cls_loss = [dn_main_cls_loss] + dn_dec_cls_loss_list + [dn_traditional_dec_cls_loss]

    # 全部的cls_loss损失, list存储数值, 共16个loss
    cls_loss = dn_cls_loss + non_dn_cls_loss

    # 计算bbox_loss, 使用全局索引
    # 非去噪部分
    # 主匹配(最后一个decoder)
    main_box_loss = get_box_loss(global_indices, bbox_out[num_decoder_layers -1], real_bbox)
    # 辅助匹配 5x decoder + 2x encoder + 1 traditional decoder
    # decoder部分(前5层)
    dec_box_loss_list = []
    for l in range(num_decoder_layers):
        if l == num_decoder_layers - 1:
            continue
        dec_box_loss = get_box_loss(global_indices, bbox_out[l], real_bbox)
        dec_box_loss_list.append(dec_box_loss)
    # encoder部分
    enc_box_loss0 = get_box_loss(global_indices, enc_bbox_pred0, real_bbox)
    enc_box_loss1 = get_box_loss(global_indices, enc_bbox_pred1, real_bbox)
    # traditional_decoder部分
    tradition_dec_box_loss = get_box_loss(global_indices, traditional_bboxes, real_bbox)
    # 1 main + 5 dec + 2 enc + 1 traditional
    non_dn_box_loss = [main_box_loss] + dec_box_loss_list + [enc_box_loss0, enc_box_loss1] + [tradition_dec_box_loss]

    # 去噪部分
    # 主匹配(最后一个decoder)
    dn_main_box_loss = get_box_loss(global_dn_indices, dn_dec_bboxes[num_decoder_layers - 1], real_bbox)
    # 辅助匹配 5x decoder + 1 traditional decoder
    # decoder部分
    dn_dec_box_loss_list = []
    for m in range(num_decoder_layers):
        if m == num_decoder_layers - 1:
            continue
        dn_dec_box_loss = get_box_loss(global_dn_indices, dn_dec_bboxes[m], real_bbox)
        dn_dec_box_loss_list.append(dn_dec_box_loss)
    # traditional_decoder部分
    dn_traditional_dec_box_loss = get_box_loss(global_dn_indices, dn_traditional_bboxes, real_bbox)
    # 1 main_dn + 5 dn_dec + 1 dn traditional dec
    dn_box_loss = [dn_main_box_loss] + dn_dec_box_loss_list + [dn_traditional_dec_box_loss]
    # 全部的box_loss损失, list存储元组, 元组形式(l1_loss, giou_loss), 共16组损失
    box_loss = dn_box_loss + non_dn_box_loss

    # local_loss, 定位损失, 使用全局索引, 仅仅针对FDR_decoder层使用
    # 获取每一层decoder输出用于local计算的值
    non_dn_outputs = []
    dn_outputs = []

    # 记录非去噪分支全局idx, teacher值, 以及去噪分支的teacher值
    non_dn_global_idx = get_permutation_idx(global_indices) # (batch_idx, query_idx)
    non_dn_teacher_logits = class_out[num_decoder_layers-1] # (bs, nq, num_classes)
    non_dn_teacher_corners = dec_out_pred_corners[num_decoder_layers-1] # (bs, nq, 132)
    dn_teacher_logits = dn_dec_logits[num_decoder_layers-1] # (bs, dnq, num_classes)
    dn_teacher_corners = dn_dec_corners[num_decoder_layers-1] # (bs, dnq, 132)
    
    for layer_id in range(0, num_decoder_layers):
        # 对于最后一个decoder输出部分进行处理, 非去噪与去噪部分均没有teacher部分
        if layer_id == num_decoder_layers - 1:
            non_dn_outputs.append({"pred_bboxes": bbox_out[layer_id], "pred_corners": dec_out_pred_corners[layer_id],
                                   "ref_points": dec_out_refs[layer_id], "reg_scale": fine_grained_out['reg_scale'],
                                   "up": fine_grained_out['up']})
            dn_outputs.append({"pred_bboxes": dn_dec_bboxes[layer_id], "pred_corners": dn_dec_corners[layer_id],
                                   "ref_points": dn_dec_refs[layer_id], "reg_scale": fine_grained_out['reg_scale'],
                                   "up": fine_grained_out['up']})
        else:
            # 对于前5层decoder的输出进行处理, 去噪与非去噪部分均有teacher部分
            non_dn_outputs.append({"pred_bboxes": bbox_out[layer_id], "pred_corners": dec_out_pred_corners[layer_id],
                                   "ref_points": dec_out_refs[layer_id], 
                                   "teacher_logits": non_dn_teacher_logits, "teacher_corners": non_dn_teacher_corners,
                                   "reg_scale": fine_grained_out['reg_scale'], "up": fine_grained_out['up']})
            
            dn_outputs.append({"pred_bboxes": dn_dec_bboxes[layer_id], "pred_corners": dn_dec_corners[layer_id],
                                   "ref_points": dn_dec_refs[layer_id], 
                                   "teacher_logits": dn_teacher_logits, "teacher_corners": dn_teacher_corners, 
                                   "reg_scale": fine_grained_out['reg_scale'], "up": fine_grained_out['up']})

    # 非去噪部分
    # 主匹配(最后一个decoder)
    main_local_loss = get_local_loss(non_dn_outputs[-1], real_bbox, global_indices, non_dn_global_idx, non_dn_teacher_logits, T=5, reg_max=32)
    # 辅助匹配
    # 5x decoder
    dec_local_loss_list = []
    for n in range(0, num_decoder_layers):
        if n == num_decoder_layers - 1:
            continue
        dec_local_loss = get_local_loss(non_dn_outputs[n], real_bbox, global_indices, non_dn_global_idx, non_dn_teacher_logits, T=5, reg_max=32)
        dec_local_loss_list.append(dec_local_loss)
    # 6x decoder
    non_dn_local_loss = [main_local_loss] + dec_local_loss_list

    # 去噪部分
    # 主匹配(最后一个decoder)
    dn_main_local_loss = get_local_loss(dn_outputs[-1], real_bbox, global_dn_indices, non_dn_global_idx, non_dn_teacher_logits, T=5, reg_max=32)
    # 辅助匹配
    # 5x decoder
    dn_dec_local_loss_list = []
    for n1 in range(0, num_decoder_layers):
        if n1 == num_decoder_layers - 1:
            continue
        dn_dec_local_loss = get_local_loss(dn_outputs[n1], real_bbox, global_dn_indices, non_dn_global_idx, non_dn_teacher_logits, T=5, reg_max=32)
        dn_dec_local_loss_list.append(dn_dec_local_loss)
    # 6x decoder
    dn_loacl_loss = [dn_main_local_loss] + dn_dec_local_loss_list
    # 全部的local_loss损失, list存储元组, 元组形式(lfgl_loss, ddf_loss), 共12组损失, 其中ddf只有10个值有效, 其中2个为最后的decoder-->ddf=0
    local_loss = dn_loacl_loss + non_dn_local_loss

    # 损失权重 loss_vfl: 1, loss_l1: 5, loss_giou: 2, loss_fgl: 0.15, loss_ddf: 1.5
    total_cls_loss = sum(cls_loss) # 16 layer
    total_l1_loss = sum([t[0] for t in box_loss]) # 16 layer
    total_giou_loss = sum([t[1] for t in box_loss]) # 16 layer
    total_fgl_loss = sum([t[0] for t in local_loss]) # 12 layer
    total_ddf_loss = sum([t[1] for t in local_loss]) # 10 layer

    # 单层级损失
    layer_cls_loss = total_cls_loss/16
    layer_box_loss = (5 * total_l1_loss + 2 * total_giou_loss)/16
    layer_local_loss = 0.15 * total_fgl_loss/12 + 1.5 * total_ddf_loss/10
    # 以下部分用于监视模型回归的状态
    loss_dict = {
        'cls_loss': layer_cls_loss.item(),
        'box_loss': layer_box_loss.item(),
        'loc_loss': layer_local_loss.item(),
    }

    total_loss = 1 * total_cls_loss + 5 * total_l1_loss + 2 * total_giou_loss + 0.15 * total_fgl_loss + 1.5 * total_ddf_loss
    main_box_loss = 5 * main_box_loss[0] + 2 * main_box_loss[1]
    main_local_loss = 0.15 * main_local_loss[0] + 1.5 * main_local_loss[1]
    return total_loss, loss_dict, main_cls_loss, main_box_loss, main_local_loss


def compute_val_loss(class_out, bbox_out, dec_out_pred_corners, dec_out_refs, 
                 traditional_logits, traditional_bboxes, fine_grained_out,
                 real_id:tuple, real_bbox:tuple, device, alpha=1.0, gamma=2.0, class_loss_methed='mal'):
    '''
    class_out: list[(bs, num_queries=300, num_classes) x (6 + 2)], 6xdecoder + 2xencoder
    bbox_out: list[(bs, num_queries=300, cxcywh) x (6 +2 )], 6xdecoder + 2xencoder

    dec_out_pred_corners: list[(bs, num_queries, 132) x 6]
    dec_out_refs: list[(bs, num_queries, 4) x 6]
    traditional_logits: (bs, num_queries, num_classes)
    traditional_bboxes: (bs, num_queries, cxcywh=4)
    fine_grained_out: 
            fine_grained_out['reg_scale'] = 4.0 无梯度
            fine_grained_out['up'] = 0.5 无梯度
    real_id: 一个批次内的真实id, tuple内存储tensor x bs, 其tensor.shape = (num_gti, )
    real_bbox: 一个批次内的真实框boxes, tuple内存储tensor x bs, 其tensor.shape = (num_gti, 4) 
    '''
    # 对真实标注进行list转换
    real_id = [id_tensor.to(device) for id_tensor in real_id]
    real_bbox = [bbox_tensor.to(device) for bbox_tensor in real_bbox]

    # encoder部分
    # tmp+ref的topk
    enc_class_logits0 = class_out[-2]
    enc_bbox_pred0 = bbox_out[-2]
    # init_box_proposals
    enc_class_logits1 = class_out[-1]
    enc_bbox_pred1 = bbox_out[-1]

    # 解码器的层数的计算, 总输出层数就是6x decoder + 2x encoder
    num_decoder_layers = len(class_out) - 2
    
    indices_aux_list = [] # 5x decoder + 2x encoder + 1 traditonal decoder, 均为本层匹配值
    # 计算所有decoder层的辅助匹配与主匹配
    for i in range(num_decoder_layers):
        class_out_i = class_out[i] # (bs, num_queries=300, num_classes)
        bbox_out_i = bbox_out[i]  # (bs, num_queries=300, bbox=4)
        if i == num_decoder_layers -1 :
            # HungarianMatch函数返回的是batch_size大小的一个列表，其中包含元组 (query_indices, target_indices), 存放匹配数据
            main_indices = HungarianMatch(real_id, real_bbox, class_out_i, bbox_out_i)
        else:
            indices_dec_aux = HungarianMatch(real_id, real_bbox, class_out_i, bbox_out_i)
            indices_aux_list.append(indices_dec_aux)
    
    # 计算encoder层的辅助匹配
    indices_enc_aux0 = HungarianMatch(real_id, real_bbox, enc_class_logits0, enc_bbox_pred0)
    indices_enc_aux1 = HungarianMatch(real_id, real_bbox, enc_class_logits1, enc_bbox_pred1)
    indices_aux_list.append(indices_enc_aux0)
    indices_aux_list.append(indices_enc_aux1)

    # 计算traditional_decoder层的辅助匹配
    indices_traditional_dec_aux = HungarianMatch(real_id, real_bbox, traditional_logits, traditional_bboxes)
    indices_aux_list.append(indices_traditional_dec_aux)

    # 获得匹配的全局索引, 用于box_loss与local_loss计算, 对于5xdecoder + 2xencoder + 1xtraditional decoder 与main_decoder进行全局匹配
    global_indices = get_global_indices(main_indices, indices_aux_list)

    # cls_loss, 使用自己本层索引
    if class_loss_methed == 'vfl':
        get_cls_loss = get_vfl_loss
    else:
        get_cls_loss = get_mal_loss
    # 非去噪部分
    ## 主匹配(最后一个decoder)
    main_cls_loss = get_cls_loss(main_indices, class_out[num_decoder_layers -1], bbox_out[num_decoder_layers -1], 
                             real_id, real_bbox, device, alpha, gamma)
    
    # 辅助匹配 5x decoder + 2x encoder + 1 traditional decoder
    # decoder部分(前5层)
    dec_cls_loss_list = []
    for j in range(num_decoder_layers):
        if j == num_decoder_layers - 1:
            continue
        dec_cls_loss = get_cls_loss(indices_aux_list[j], class_out[j], bbox_out[j], real_id, real_bbox, device, alpha, gamma)
        dec_cls_loss_list.append(dec_cls_loss)
    # encoder部分
    enc_cls_loss0 = get_cls_loss(indices_aux_list[-3], enc_class_logits0, enc_bbox_pred0, real_id, real_bbox, device, alpha, gamma)
    enc_cls_loss1 = get_cls_loss(indices_aux_list[-2], enc_class_logits1, enc_bbox_pred1, real_id, real_bbox, device, alpha, gamma)
    # traditional_decoder部分
    tradition_dec_cls_loss = get_cls_loss(indices_aux_list[-1], traditional_logits, traditional_bboxes, 
                                      real_id, real_bbox, device, alpha, gamma)
    # 1 main + 5 dec + 2 enc + 1 traditional
    non_dn_cls_loss = [main_cls_loss] + dec_cls_loss_list + [enc_cls_loss0, enc_cls_loss1] + [tradition_dec_cls_loss]
    # 全部的cls_loss损失, list存储数值, 共9个loss
    cls_loss = non_dn_cls_loss

    # 计算bbox_loss, 使用全局索引
    # 非去噪部分
    # 主匹配(最后一个decoder)
    main_box_loss = get_box_loss(global_indices, bbox_out[num_decoder_layers -1], real_bbox)
    # 辅助匹配 5x decoder + 2x encoder + 1 traditional decoder
    # decoder部分(前5层)
    dec_box_loss_list = []
    for l in range(num_decoder_layers):
        if l == num_decoder_layers - 1:
            continue
        dec_box_loss = get_box_loss(global_indices, bbox_out[l], real_bbox)
        dec_box_loss_list.append(dec_box_loss)
    # encoder部分
    enc_box_loss0 = get_box_loss(global_indices, enc_bbox_pred0, real_bbox)
    enc_box_loss1 = get_box_loss(global_indices, enc_bbox_pred1, real_bbox)
    # traditional_decoder部分
    tradition_dec_box_loss = get_box_loss(global_indices, traditional_bboxes, real_bbox)
    # 1 main + 5 dec + 2 enc + 1 traditional
    non_dn_box_loss = [main_box_loss] + dec_box_loss_list + [enc_box_loss0, enc_box_loss1] + [tradition_dec_box_loss]
    # 全部的box_loss损失, list存储元组, 元组形式(l1_loss, giou_loss), 共9组损失
    box_loss = non_dn_box_loss

    # local_loss, 定位损失, 使用全局索引, 仅仅针对FDR_decoder层使用
    # 获取每一层decoder输出用于local计算的值
    non_dn_outputs = []

    # 记录非去噪分支全局idx, teacher值, 以及去噪分支的teacher值
    non_dn_global_idx = get_permutation_idx(global_indices) # (batch_idx, query_idx)
    non_dn_teacher_logits = class_out[num_decoder_layers-1] # (bs, nq, num_classes)
    non_dn_teacher_corners = dec_out_pred_corners[num_decoder_layers-1] # (bs, nq, 132)
    
    for layer_id in range(0, num_decoder_layers):
        # 对于最后一个decoder输出部分进行处理, 非去噪与去噪部分均没有teacher部分
        if layer_id == num_decoder_layers - 1:
            non_dn_outputs.append({"pred_bboxes": bbox_out[layer_id], "pred_corners": dec_out_pred_corners[layer_id],
                                   "ref_points": dec_out_refs[layer_id], "reg_scale": fine_grained_out['reg_scale'],
                                   "up": fine_grained_out['up']})
        else:
            # 对于前5层decoder的输出进行处理, 去噪与非去噪部分均有teacher部分
            non_dn_outputs.append({"pred_bboxes": bbox_out[layer_id], "pred_corners": dec_out_pred_corners[layer_id],
                                   "ref_points": dec_out_refs[layer_id], 
                                   "teacher_logits": non_dn_teacher_logits, "teacher_corners": non_dn_teacher_corners,
                                   "reg_scale": fine_grained_out['reg_scale'], "up": fine_grained_out['up']})

    # 非去噪部分
    # 主匹配(最后一个decoder)
    main_local_loss = get_local_loss(non_dn_outputs[-1], real_bbox, global_indices, non_dn_global_idx, non_dn_teacher_logits, T=5, reg_max=32)
    # 辅助匹配
    # 5x decoder
    dec_local_loss_list = []
    for n in range(0, num_decoder_layers):
        if n == num_decoder_layers - 1:
            continue
        dec_local_loss = get_local_loss(non_dn_outputs[n], real_bbox, global_indices, non_dn_global_idx, non_dn_teacher_logits, T=5, reg_max=32)
        dec_local_loss_list.append(dec_local_loss)
    # 6x decoder
    non_dn_local_loss = [main_local_loss] + dec_local_loss_list
    # 全部的local_loss损失, list存储元组, 元组形式(lfgl_loss, ddf_loss), 共6组损失, 其中ddf只有5个值有效, 其中1个为最后的decoder-->ddf=0
    local_loss = non_dn_local_loss

    # 损失权重 loss_vfl: 1, loss_l1: 5, loss_giou: 2, loss_fgl: 0.15, loss_ddf: 1.5
    total_cls_loss = sum(cls_loss) # 9 layer
    total_l1_loss = sum([t[0] for t in box_loss]) # 9 layer
    total_giou_loss = sum([t[1] for t in box_loss]) # 9 layer
    total_fgl_loss = sum([t[0] for t in local_loss]) # 6 layer
    total_ddf_loss = sum([t[1] for t in local_loss]) # 5 layer

    # 单层级损失
    layer_cls_loss = total_cls_loss/9
    layer_box_loss = (5 * total_l1_loss + 2 * total_giou_loss)/9
    layer_local_loss = 0.15 * total_fgl_loss/6 + 1.5 * total_ddf_loss/5
    # 以下部分用于监视模型回归的状态
    loss_dict = {
        'cls_loss': layer_cls_loss.item(),
        'box_loss': layer_box_loss.item(),
        'loc_loss': layer_local_loss.item(),
    }

    total_loss = 1 * total_cls_loss + 5 * total_l1_loss + 2 * total_giou_loss + 0.15 * total_fgl_loss + 1.5 * total_ddf_loss
    main_box_loss = 5 * main_box_loss[0] + 2 * main_box_loss[1]
    main_local_loss = 0.15 * main_local_loss[0] + 1.5 * main_local_loss[1]
    return total_loss, loss_dict, main_cls_loss, main_box_loss, main_local_loss


# 分类损失计算, indices是本层的匹配结果
# # 所有decoder和encoder都计算vfl分类损失
def get_vfl_loss(indices, class_pred, bbox_pred, real_id, real_bbox, device, alpha=0.75, gamma=2.0):
    '''
    indices: 本层匈牙利匹配或者去噪匹配的索引, (batch_idx, query_idx), 长度为total_matches = total_gt
    class_pred: 本层分类预测logits (bs, nq, num_classes) or (bs, dnq, num_classes)
    bbox_pred: 本层预测边界框 (bs, nq, 4) or (bs, dnq, 4)
    real_id: 一个批次内的真实id, tuple内存储tensor x bs, 其tensor.shape = (num_gti, )
    real_bbox: 一个批次内的真实框boxes, tuple内存储tensor x bs, 其tensor.shape = (num_gti, 4) 
    '''
    idx = get_permutation_idx(indices)
    # 真实框数量, 对去噪和非去噪均有用
    num_total_gt = len(idx[0])
    bs, num_queries, num_classes = class_pred.shape
    # 预分配目标张量, matched_id=(bs, num_queries=300), 内部全填num_classes
    matched_id = torch.full((len(indices), num_queries), num_classes, dtype=torch.long).to(device)
    # j表示批次内的图像索引, (query_indices, target_indices)则表示query与target的对应索引
    # query_indices是从小到大排序, target_indices则是按照query_indices对应排好序的目标类别(不是从小到大的, 而是安排好的)
    batch_gt_id = []
    batch_gt_bbox = []
    for j, (query_indices, target_indices) in enumerate(indices):
        # 使用 real_id 直接填充
        arranged_id = real_id[j][target_indices] # 找出安排好的目标类别
        batch_gt_id.append(arranged_id)

        # 对真实框进行aranged 索引匹配
        arranged_bbox = real_bbox[j][target_indices]
        batch_gt_bbox.append(arranged_bbox)
    
    # 一个批次的预测框与对应真实框, 用于计算ious
    batch_pred_bbox = bbox_pred[idx]
    batch_gt_bbox = torch.cat(batch_gt_bbox, dim=0)

    # 把排序好的类别 按指定batch, query索引填入
    matched_id[idx] = torch.cat(batch_gt_id, dim=0)

    # 得到样本对应的ious
    ious, _ = box_iou(box_cxcywh_to_xyxy(batch_pred_bbox), box_cxcywh_to_xyxy(batch_gt_bbox)) # (total_matches, total_matches)
    ious = torch.diag(ious).detach() # (total_matches,)

    # 计算分类损失, 每一个query框的cls_loss
    src_logits = class_pred  # (bs, num_queries, num_classes)
    
    # target_classes_onehot dim=(bs, num_queries, num_classes+1)全0张量
    target_classes_onehot = torch.zeros([bs, num_queries, num_classes + 1],
                                    dtype=src_logits.dtype, device=device)
    target_classes_onehot.scatter_(2, matched_id.unsqueeze(-1), 1)
    target_classes_onehot = target_classes_onehot[:, :, :-1] # 是一个只包含0或1的张量, 维度为(bs, num_queries, num_classes)

    # 创建iou-aware分数计算模版: target_score_o=(bs, num_queries), 全零张量(全部是负样本的分数, 均为0)
    target_score_o = torch.zeros((bs, num_queries), dtype=src_logits.dtype, device=device)
    # iou赋值给匹配的queries, (bs, num_queries) 每一个query都对应一个iou值, 正样本为1 x IoU, 负样本为0(没有赋值改变)
    target_score_o[idx] = ious.to(target_score_o.dtype) # (bs, num_queries)
    target_score = target_score_o.unsqueeze(-1) * target_classes_onehot # (bs, num_queries, num_classes)

    # vfl
    # target_score = IoU
    # 对于正样本 权重为alpha * pred_score.pow(gamma) * (1 - target_classes_onehot) + target_score(值=IoU)
    # 对于负样本 权重为alpha * pred_score.pow(gamma) * (1 - target_classes_onehot)
    pred_score = F.sigmoid(src_logits).detach() # (bs, dn_num_queries, num_classes)
    weight = alpha * pred_score.pow(gamma) * (1 - target_classes_onehot) + target_score
    
    cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
    cls_loss = cls_loss.mean(1).sum() * src_logits.shape[1] / num_total_gt
    return cls_loss

# mal_loss是对vfl_loss的低质量匹配的改进(改变iou低, score高时, vfl损失loss低的问题, vfl与mal在iou高, score低, loss都很大)
# 其中的alpha默认为1.0
def get_mal_loss(indices, class_pred, bbox_pred, real_id, real_bbox, device, alpha=1.0, gamma=1.5):
    '''
    indices: 本层匈牙利匹配或者去噪匹配的索引, (batch_idx, query_idx), 长度为total_matches = total_gt
    class_pred: 本层分类预测logits (bs, nq, num_classes) or (bs, dnq, num_classes)
    bbox_pred: 本层预测边界框 (bs, nq, 4) or (bs, dnq, 4)
    real_id: 一个批次内的真实id, tuple内存储tensor x bs, 其tensor.shape = (num_gti, )
    real_bbox: 一个批次内的真实框boxes, tuple内存储tensor x bs, 其tensor.shape = (num_gti, 4) 
    '''
    idx = get_permutation_idx(indices)
    # 真实框数量, 对去噪和非去噪均有用
    num_total_gt = len(idx[0])
    bs, num_queries, num_classes = class_pred.shape
    # 预分配目标张量, matched_id=(bs, num_queries=300), 内部全填num_classes
    matched_id = torch.full((len(indices), num_queries), num_classes, dtype=torch.long).to(device)
    # j表示批次内的图像索引, (query_indices, target_indices)则表示query与target的对应索引
    # query_indices是从小到大排序, target_indices则是按照query_indices对应排好序的目标类别(不是从小到大的, 而是安排好的)
    batch_gt_id = []
    batch_gt_bbox = []
    for j, (query_indices, target_indices) in enumerate(indices):
        # 使用 real_id 直接填充
        arranged_id = real_id[j][target_indices] # 找出安排好的目标类别
        batch_gt_id.append(arranged_id)

        # 对真实框进行aranged 索引匹配
        arranged_bbox = real_bbox[j][target_indices]
        batch_gt_bbox.append(arranged_bbox)
    
    # 一个批次的预测框与对应真实框, 用于计算ious
    batch_pred_bbox = bbox_pred[idx]
    batch_gt_bbox = torch.cat(batch_gt_bbox, dim=0)

    # 把排序好的类别 按指定batch, query索引填入
    matched_id[idx] = torch.cat(batch_gt_id, dim=0)

    # 得到样本对应的ious
    ious, _ = box_iou(box_cxcywh_to_xyxy(batch_pred_bbox), box_cxcywh_to_xyxy(batch_gt_bbox)) # (total_matches, total_matches)
    ious = torch.diag(ious).detach() # (total_matches,)

    # 计算分类损失, 每一个query框的cls_loss
    src_logits = class_pred  # (bs, num_queries, num_classes)
    
    # target_classes_onehot dim=(bs, num_queries, num_classes+1)全0张量
    target_classes_onehot = torch.zeros([bs, num_queries, num_classes + 1],
                                    dtype=src_logits.dtype, device=device)
    target_classes_onehot.scatter_(2, matched_id.unsqueeze(-1), 1)
    target_classes_onehot = target_classes_onehot[:, :, :-1] # 是一个只包含0或1的张量, 维度为(bs, num_queries, num_classes)

    # 创建iou-aware分数计算模版: target_score_o=(bs, num_queries), 全零张量(全部是负样本的分数, 均为0)
    target_score_o = torch.zeros((bs, num_queries), dtype=src_logits.dtype, device=device)
    # iou赋值给匹配的queries, (bs, num_queries) 每一个query都对应一个iou值, 正样本为1 x IoU, 负样本为0(没有赋值改变)
    target_score_o[idx] = ious.to(target_score_o.dtype) # (bs, num_queries)
    target_score = target_score_o.unsqueeze(-1) * target_classes_onehot # (bs, num_queries, num_classes)

    # mal_loss改进
    # target_score = IoU.pow(gamma)
    # 对于正样本 权重为pred_score.pow(gamma) * (1 - target_classes_onehot) + target_classes_onehot(值=1)
    # 对于负样本 权重为pred_score.pow(gamma) * (1 - target_classes_onehot)
    pred_score = F.sigmoid(src_logits).detach() # (bs, dn_num_queries, num_classes)
    target_score = target_score.pow(gamma)
    weight = alpha * pred_score.pow(gamma) * (1 - target_classes_onehot) + target_classes_onehot
    
    cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
    cls_loss = cls_loss.mean(1).sum() * src_logits.shape[1] / num_total_gt
    return cls_loss

# 边界框损失计算, 使用全局索引
# 所有decoder和encoder都计算box_loss
def get_box_loss(global_indices, bbox_pred, real_bbox):
    '''
    global_indices: 全局匹配索引, (batch_idx, query_idx), 长度为total_matches = total_gt
    bbox_pred: 本层预测边界框 (bs, nq, 4) or (bs, dnq, 4)
    real_id: 一个批次内的真实id, tuple内存储tensor x bs, 其tensor.shape = (num_gti, )
    real_bbox: 一个批次内的真实框boxes, tuple内存储tensor x bs, 其tensor.shape = (num_gti, 4) 
    '''
    idx = get_permutation_idx(global_indices)
    # 真实框数量, 对去噪和非去噪均有用
    num_total_gt = len(idx[0]) 
    batch_gt_bbox = []
    for j, (query_indices, target_indices) in enumerate(global_indices):
        arranged_bbox = real_bbox[j][target_indices]
        batch_gt_bbox.append(arranged_bbox)
    
    batch_pred_bbox = bbox_pred[idx]
    batch_gt_bbox = torch.cat(batch_gt_bbox, dim=0)

    l1_loss_batch = F.l1_loss(batch_gt_bbox, batch_pred_bbox, reduction='none')
    l1_loss = l1_loss_batch.sum() / num_total_gt

    giou_loss_batch = 1 - torch.diag(box_ops.generalized_box_iou(
                                        box_cxcywh_to_xyxy(batch_gt_bbox),
                                        box_cxcywh_to_xyxy(batch_pred_bbox)))
    giou_loss = giou_loss_batch.sum() / num_total_gt

    return (l1_loss, giou_loss)

# 定位损失计算, 使用全局索引
# 只有6个decoder计算fgl, 只有前5个decoder计算ddf
def get_local_loss(outputs, real_bbox, global_indices, 
               non_dn_global_idx, non_dn_teacher_logits, T=5, reg_max=32):
    """
    Fine-Grained Localization (FGL) Loss 细粒度定位 损失
    Decoupled Distillation Focal (DDF) Loss 解耦蒸馏交点 损失

    outputs是单一decoder的预测结果
    包含pred_bboxes, pred_corners, ref_points, reg_scale, up, teacher_corners, teacher_logits
    pred_bboxes: (bs, dnq, cxcywh=4) or (bs, nq, cxcywh=4)
    pred_corners: (bs, dnq, 132) or (bs, nq, 132)
    ref_points: (bs, dnq, cxcywh=4) or (bs, nq, cxcywh=4), 每个decoder的ref_points都是同一个值
    reg_scale: 4.0
    up: 0.5
    teacher_corners: 最后一个decoder输出的pred_corners, (bs, dnq, 132) or (bs, nq, 132)
    teacher_logits: 最后一个decoder输出的pred_logits, (bs, dnq, num_classes) or (bs, nq, num_classes)
    
    notice: teacher是最后一层decoder的输出部分, 只计算前5个decoder的ddf损失, 最后一层decoder输出不进行蒸馏ddf
            当outputs是最后一个decoder的输出值时候, 没有teacher存在
    
    global_indices是全局匹配索引, 去噪分支与非去噪分支均使用
    non_dn_global_idx: 是非去噪分支的全局匹配idx, 用于计算去噪分支ddf的num_pos和num_neg
    non_dn_teacher_logits: 非去噪分支的teacher_logits, 用于计算去噪分支ddf的num_pos和num_neg
    """
    # 获取全局的匹配索引, 去噪分支与非去噪分支均使用
    # idx = (batch_idx, query_idx), 长度为total_matches = total_gt
    idx = get_permutation_idx(global_indices)
    # 真实框数量, 对去噪和非去噪均有用
    num_total_gt = len(idx[0])
    # 匹配的真实框进行索引排序
    batch_gt_bbox = []
    for j, (query_indices, target_indices) in enumerate(global_indices):
        # l1_loss与giou_loss, 只考虑匹配上的框计算
        arranged_bbox = real_bbox[j][target_indices] # (num_matches, 4)
        batch_gt_bbox.append(arranged_bbox)
    batch_gt_bbox = torch.cat(batch_gt_bbox, dim=0) # (matched_gt, 4)
    # (bs, q, 132) -> (matched_gt, 132) -> (matched_gt*4, 33)
    pred_corners = outputs["pred_corners"][idx].reshape(-1, (reg_max + 1)) 
    ref_points = outputs["ref_points"][idx].detach() # 无梯度构造

    # gt框偏移分布的构造, 因为gt是无梯度的
    with torch.no_grad():
        fgl_gt = bbox2distance(
            ref_points,
            box_cxcywh_to_xyxy(batch_gt_bbox),
            reg_max,
            outputs["reg_scale"],
            outputs["up"],
        )
    # gt_left_bin, weight_right, weight_left均为(matched_gt*4, )
    # 其中gt_left_bin是真实框的狄拉克delta分布的左侧bin索引
    # weight_right, weight_left是真实框的狄拉克delta分布的bin坐标离左右临近bin坐标的距离求出的权重值
    gt_left_bin, weight_right, weight_left = (fgl_gt)

    # iou计算与fgl权重平衡系数
    # outputs["pred_bboxes"][idx] = (matched_gt, 4)
    # batch_gt_bbox = (matched_gt, 4)
    # box_iou[0]返回的是(matched_gt, matched_gt)的iou矩阵, 取主对角线就是匹配上的框的iou值
    # torch.diag()返回的是主对角线的元素, 形状为(matched_gt,), ious = (matched_gt,)
    ious = torch.diag(box_iou( 
                                box_cxcywh_to_xyxy(outputs["pred_bboxes"][idx]), 
                                box_cxcywh_to_xyxy(batch_gt_bbox) 
                            )[0])
    # (matched_gt,) -> (matched_gt, 1) -> (1, matched_gt, 4) -> (matched_gt*4,)
    weight = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach() # 匹配上的框的权重值
    fgl_loss = unimodal_distribution_focal_loss(
        pred_corners,
        gt_left_bin,
        weight_right,
        weight_left,
        weight,
        avg_factor=num_total_gt,
    )

    # 这里是为了判断是否进行ddf计算的部分, 如果输入层有teacher就计算ddf损失, 没有teacher就不计算ddf损失
    # 其实就是说对于最后一个decoder不输入teacher_logits和teacher_corners, 其余decoder输入teacher_logits和teacher_corners
    if "teacher_corners" not in outputs:
        ddf_loss = 0
    else:
        # (bs, q, 132) -> (bs*q*4, 33)
        pred_corners = outputs["pred_corners"].reshape(-1, (reg_max + 1))
        # (bs, q, 132) -> (bs*q*4, 33)
        gt_corners = outputs["teacher_corners"].reshape(-1, (reg_max + 1))
        # 如果pred_corners与target_corners相等, 那损失就是0
        if torch.equal(pred_corners, gt_corners):
            ddf_loss = pred_corners.sum() * 0
        else:
            # (bs, q, num_classes).max(dim=-1) -> values and indices
            # values是每个预测框的最大logit.sigmoid值, indices是每个预测框最大logit值对应的类别索引
            # weight_gt_local = (bs, q)
            weight_gt_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]
            # mask = (bs, q)
            mask = torch.zeros_like(weight_gt_local, dtype=torch.bool)
            # mask[idx]位置的值被替换为True, mask的形状依旧为(bs, q)
            mask[idx] = True
            # 得到对应的正样本与负样本掩码
            # (bs, q) -> (bs, q, 1) -> (bs, q, 4) -> (bs*q*4,)
            mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)
            # (bs, q) -> (bs, q)不变形状, 把匹配上的位置替换为ious值
            # weight_gt_local[idx] = (matched_gt,), ious = (matched_gt,) 直接赋值即可
            # 匹配上的是ious值, 未匹配上的是原始的weight_gt_local分数值
            weight_gt_local[idx] = ious.to(weight_gt_local.dtype)
            # (bs, q) -> (bs, q, 1) -> (bs, q, 4) -> (bs*q*4,).detach()
            weight_gt_local = weight_gt_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
            # 注意KL散度损失的计算, 公式为:
            # L = sum( p * (log(p) - log(q)) )
            # 其中p是真实分布, q是预测分布, 这里的p是gt_corners, q是pred_corners
            # kl_div_loss.dim = (bs*q*4, 33)
            kl_div_loss = nn.KLDivLoss(reduction="none")(
                            F.log_softmax(pred_corners / T, dim=1),
                            F.softmax(gt_corners.detach() / T, dim=1))
            # kl_scales.dim = (bs*q*4, )
            kl_scales = weight_gt_local * (T**2)
            # kl_div_loss.sum(-1)是每个样本边的kl散度损失, (bs*q*4, 33) -> (bs*q*4, )
            # loss_match_local.dim = (bs*q*4, ), 每个样本的kl散度损失都被加权了
            loss_match_local = kl_scales * kl_div_loss.sum(-1)
            
            # output['pred_boxes'].shape[0]表示的是批次大小batch_size
            # ddf加权平衡参数计算: 对于去噪与非去噪分支而言, 均根据非去噪部分全局匹配的num_pos, num_neg
            num_pos, num_neg = ddf_num_pos_neg_compute(non_dn_global_idx, non_dn_teacher_logits)

            # 用于计算匹配到的预测框的根号比例值
            loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
            # 用于计算未匹配到的预测框的根号比例值
            loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
            ddf_loss = (
                loss_match_local1 * num_pos + loss_match_local2 * num_neg
            ) / (num_pos + num_neg)

    return (fgl_loss, ddf_loss)


# 把匈牙利匹配或者去噪训练得到的结果, 转换为(batch_idx, query_idx)
def get_permutation_idx(matched_indices):
    # 构建全局索引 idx
    batch_idx_list = []   # 存每个正样本属于哪个 batch 图像
    query_idx_list = []   # 存每个正样本对应的 query 索引
    for j, (query_indices, target_indices) in enumerate(matched_indices):
        # 展开 batch id 和 query indices
        batch_idx_list.append(torch.full_like(query_indices, j))  # shape: (num_matches_in_img_j, )
        query_idx_list.append(query_indices) # shape: (num_matches_in_img_j, )

    batch_idx = torch.cat(batch_idx_list)   # (total_matches,)
    query_idx = torch.cat(query_idx_list)   # (total_matches,)
    idx = (batch_idx, query_idx)           # (total_matches, total_matches)
    return idx

# 用于获取全局的匹配索引, 用到enc的indices, dec的非去噪分支的indices, traditional_head 的indices
# indices是最后一个decoder的匹配结果(主匹配), indices_aux_list是其余所有部分每一次匈牙利匹配的结果
# 这里只处理不是去噪训练的每层匹配结果, 得到一个全局的索引
def get_global_indices(indices, indices_aux_list):
    """Get a matching union set across all decoder layers."""
    # 同一个query会匹配到不同的目标, 选择query匹配到相同目标最多的那一个作为最终匹配对
    results = []
    for indices_aux in indices_aux_list:
        indices = [
            (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
            for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
        ]

    for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
        unique, counts = torch.unique(ind, return_counts=True, dim=0)
        count_sort_indices = torch.argsort(counts, descending=True)
        unique_sorted = unique[count_sort_indices]
        column_to_row = {}
        for idx in unique_sorted:
            row_idx, col_idx = idx[0].item(), idx[1].item()
            if row_idx not in column_to_row:
                column_to_row[row_idx] = col_idx
        final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
        final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
        results.append((final_rows.long(), final_cols.long()))
    return results

# 获取ddf所需要的num_pos与num_neg
def ddf_num_pos_neg_compute(non_dn_global_idx, non_dn_teacher_logits):
    '''
    non_dn_global_idx: 非去噪部分全局idx
    non_dn_teacher_logits: 非去噪分支的teacher_logits
    '''
    weight_gt_local = non_dn_teacher_logits.sigmoid().max(dim=-1)[0]
    batch_size = non_dn_teacher_logits.shape[0]

    mask = torch.zeros_like(weight_gt_local, dtype=torch.bool)
    mask[non_dn_global_idx] = True
    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

    batch_scale = (8 / batch_size)
    num_pos, num_neg = (
        (mask.sum() * batch_scale) ** 0.5,
        ((~mask).sum() * batch_scale) ** 0.5,
    )
    return num_pos, num_neg


# 分布交点损失计算dfl
def unimodal_distribution_focal_loss(pred, bin, weight_right, weight_left, 
                                     weight=None, reduction="sum", avg_factor=None):
    # N为一个批次内所有匹配到的全局匹配真实框
    # 最大索引越界处理 bin = 32 -0.1 = 31.9, dis_left=bin.long() = 31
    dis_left = bin.long() # (matched_gt*4, ) 即gt_corners, 真值左侧bin索引(33个bin的序号的其中一个)
    dis_right = dis_left + 1 # (matched_gt*4, )

    # pred是一个分布, dis_left/dis_right, weight_left/weight_right均视为标签(标签是没有梯度的)
    # pred: (matched_gt*4, 33)
    # loss越小, pred分布就越接近真实分布( 就两个值, 左bin(k)和右bin(k+1) )
    # cross_entropy:
    # (1) 对pred做log_softmax, 得到每个bin的概率的对数
    # (2) 按dis_left/dis_right取出pred中对应bin的概率对数, 然后取负数
    # (3) reduction="none"对所有负对数计算后保留不做任何处理, 返回dis_left/dis_right维度的值
    # 类比分类问题, 每个样本有多个类别(33个bin), 每个样本只有一个正确类别(gt_left_bin或gt_right_bin), 其他类别概率为0
    loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
        -1
    ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)

    # 这里的weight是weight_target(iou值) (N, )
    if weight is not None:
        weight = weight.float()
        loss = loss * weight

    if avg_factor is not None:
        loss = loss.sum() / avg_factor
    elif reduction == "mean":
        loss = loss.mean()

    # 默认执行这个损失
    elif reduction == "sum":
        loss = loss.sum()

    return loss