import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import random

from models import GatedAttention_FineGrained_DINO_Swin, compute_val_loss
from datasets import ValDataset_for_DETR
from utils import box_cxcywh_to_xyxy, collate_fn
from tools import compute_confusion_matrix, plot_confusion_matrix, gt_count_func, calculate_metrics

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 验证模型
def validate_model(val_imgpath, 
                   val_txtpath, 
                   seed_worker, 
                   model_path=None, 
                   model=None, 
                   prefetch_factor=None, 
                   seed=42, 
                   num_classes=80, 
                   num_queries=300, 
                   batch_size=2, 
                   workers=0, 
                   gate_attn=False, 
                   pin_memory=False, 
                   scores_threshold=0.001,
                   alpha=1.0, 
                   gamma=1.5, 
                   cls_loss_method='mal', 
                   compute_loss=False, 
                   drawing=False, 
                   max_size=640, 
                   val_size=640, 
                   use_ema=True):
    if drawing :
        draw_dir = os.path.join(os.path.dirname(model_path), "draws")
        os.makedirs(draw_dir, exist_ok=True)
    else:
        draw_dir = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建数据集和数据加载器
    my_dataset = ValDataset_for_DETR(val_imgpath, 
                                     val_txtpath, 
                                     image_set='val', 
                                     max_size=max_size, 
                                     val_size=val_size)
    g = torch.Generator()
    g.manual_seed(seed)
    mydataloader = DataLoader(my_dataset, 
                                   batch_size=batch_size, 
                                   shuffle=False, 
                                   collate_fn=collate_fn, 
                                   num_workers=workers, 
                                   prefetch_factor=prefetch_factor,
                                   pin_memory=pin_memory,
                                   worker_init_fn=seed_worker, 
                                   generator=g)
    
    # 训练结束后加载权重的验证
    if model_path != None and model == None:
        cur_model = GatedAttention_FineGrained_DINO_Swin(num_queries=num_queries, 
                                                        num_classes=num_classes, 
                                                        num_encoder_layer=6, 
                                                        num_decoder_layer=6, 
                                                        gate_attn=gate_attn,
                                                        ).to(device)
        # 注意权重的加载, 从保存的字典中读取module权重
        checkpoint = torch.load(model_path, map_location=device)
        if use_ema:
            # 由于ema保存的是一个字典, 因此还需要module获取权重
            model_weights = checkpoint['ema_state_dict']['module']
        else:
            # 原始权重保存仅有module
            model_weights = checkpoint['model_state_dict']
        # 加载model_weights
        cur_model.load_state_dict(model_weights)
        cur_model.eval()
    # 训练过程中的验证
    elif model_path == None and model != None:
        cur_model = model
        cur_model:GatedAttention_FineGrained_DINO_Swin
        cur_model.eval()
    # 否则报错
    else:
        raise ValueError('Running Error!')

    # 存储真实标注和预测结果
    gt_data_all = defaultdict(lambda: defaultdict(list))
    pred_data_all = defaultdict(lambda: defaultdict(list))

    # 验证损失累积
    val_losses = {
        'total': 0.0, 'cls': 0.0, 'box': 0.0, 'loc': 0.0,
        'main_cls': 0.0, 'main_box': 0.0, 'main_loc': 0.0
    } if compute_loss else None
    val_batch_num = 0
    
    # 显式的顺序加载
    img_name_lst = sorted(os.listdir(val_imgpath))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(mydataloader, desc="Verification Progress")):
            nested_tensor, real_id, real_norm_bboxes, _, _ = batch  # 忽略不需要的real_id, real_norm_bboxes
            imgs, masks = nested_tensor.tensors, nested_tensor.mask
    
            # 计算验证损失
            if compute_loss:
                # 返回所有层级的值, return_all_layers=True
                all_cls, all_bbox, _, \
                dec_out_corners, dec_out_ref_initials, \
                traditional_logits, traditional_bboxes, \
                fine_grained_out = cur_model(imgs.to(device), masks.to(device), 
                                             return_all_layers=True, draw_heatmap=False)
                
                total_loss, loss_dict, main_cls_loss, \
                    main_box_loss, main_loc_loss = compute_val_loss(
                    all_cls, all_bbox, dec_out_corners, dec_out_ref_initials,
                    traditional_logits, traditional_bboxes, fine_grained_out,
                    real_id, real_norm_bboxes, device, alpha=alpha, gamma=gamma,
                    class_loss_methed=cls_loss_method
                )
                
                val_losses['total'] += total_loss.item()
                val_losses['cls'] += loss_dict['cls_loss']
                val_losses['box'] += loss_dict['box_loss']
                val_losses['loc'] += loss_dict['loc_loss']
                val_losses['main_cls'] += main_cls_loss.item()
                val_losses['main_box'] += main_box_loss.item()
                val_losses['main_loc'] += main_loc_loss.item()
                val_batch_num += 1
            
            # 模型推理, 不绘制heatmap热力图
            last_cls, last_bbox, heatmap = cur_model(imgs.to(device), masks.to(device), 
                                                     return_all_layers=False, draw_heatmap=False)

            last_cls: torch.Tensor
            last_bbox:torch.Tensor
            pred_cls_probability = last_cls.sigmoid() # (bs, num_queries, num_classes)
            pred_bbox = box_cxcywh_to_xyxy(last_bbox) # (bs, num_queries, cxcywh=4) ->(bs, num_queries, xyxy)
            
            # 处理批次中的每个图像
            for i in range(imgs.size(0)):
                img_idx = batch_idx * mydataloader.batch_size + i
                img_name = img_name_lst[img_idx]
                img_id = os.path.splitext(img_name)[0]
                
                # 处理真实标注
                txt_path = os.path.join(val_txtpath, f"{img_id}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            class_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                            # gt_data_all: {img_id0: {class_id0: [[x1, y1, x2, y2], [x1, y1, x2, y2]],
                            #                        class_id1: [[x1, y1, x2, y2]]},
                            #               img_id1: {class_id0: [[x1, y1, x2, y2]], 
                            #                        class_id2: [[x1, y1, x2, y2]]} }
                            gt_data_all[img_id][class_id].append([x1, y1, x2, y2])
                
                # 处理预测结果
                each_img_cls_pre = pred_cls_probability[i] # (bs, num_queries, num_classes) -> (num_queries, num_classes)
                max_prob, max_cls_idx = torch.max(each_img_cls_pre, dim=-1)
                
                pred_score_i = max_prob # (num_queries,)
                pred_cls_i = max_cls_idx # (num_queries,)
                pred_bbox_i = pred_bbox[i] # (num_queries, 4)

                valid_mask = pred_score_i >= scores_threshold # 直接判断其中的最大概率是否大于scores_threshold

                # 取 top-100
                # 保留置信度最高的前100个预测框 (maxDets=100), 如果预测框不足100个, 则保留所有预测框, 这是切片的特性
                topk = min(100, valid_mask.sum().item())
                top_scores, top_idx = torch.topk(pred_score_i[valid_mask], topk, sorted=True)
                top_labels = pred_cls_i[valid_mask][top_idx] # (valid_queries,)
                top_bboxes = pred_bbox_i[valid_mask][top_idx] # (valid_queries, 4)

                # 转为numpy(高效)
                topk_labels = top_labels.cpu().numpy()
                topk_bboxes = top_bboxes.cpu().numpy()
                topk_scores = top_scores.cpu().numpy()
                
                for cls_id, bbox, score in zip(topk_labels, topk_bboxes, topk_scores):
                    # pred_data_all: {img_id0: {class_id0: [([x1, y1, x2, y2], score), 
                    #                                       ([x1, y1, x2, y2], score],
                    #                           class_id1: [ ([x1, y1, x2, y2], score) ]},
                    #                 img_id1: {class_id0: [([x1, y1, x2, y2], score)], 
                    #                          class_id2: [([x1, y1, x2, y2], score)]} }
                    # numpy数组也支持tolist(),item()            
                    pred_data_all[img_id][cls_id.item()].append((bbox.tolist(), score.item()))
    # 只计算一次全体值
    class_gt_counts, all_gt_classes, all_img_names = gt_count_func(gt_data_all)
    
    # 计算评估指标
    iou_list = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    class_metrics_list = []
    map_list = []
    all_class_p = []
    all_class_r = []
    all_class_score = []

    for iou in tqdm(iou_list, desc="Caculating Validation Metrics"):
        gt_class_metrics, all_p_iou, all_r_iou, all_score_iou, map_iou = calculate_metrics(gt_data_all, pred_data_all, class_gt_counts,
                                                                                        all_gt_classes, all_img_names, iou_threshold=iou,
                                                                                        drawing=drawing, draw_dir=draw_dir)
        
        class_metrics_list.append(gt_class_metrics)
        all_class_p.append(all_p_iou)
        all_class_r.append(all_r_iou)
        all_class_score.append(all_score_iou)
        map_list.append(map_iou)
        
    map50 = map_list[0]
    map75 = map_list[5]
    map50t95 = np.mean(np.array(map_list))

    class_metrics50 = class_metrics_list[0]
    class_metrics50:dict
    id_lst = list(class_metrics50.keys())

    all_class_p50 = all_class_p[0]
    all_class_r50 = all_class_r[0]
    all_class_score50 = all_class_score[0]

    # 统计该类别的真实框数量, 是一个列表, total_gt = sum(gt_num), gt_num中每一个元素是对应类别gt个数
    gt_num = [class_metrics50[idx]['gt'] for idx in id_lst]

    # 获取不同iou阈值下各个类别的计算结果
    p_iou, r_iou, score_iou, ap_iou = [], [], [], []
    for metrics in class_metrics_list:
        # 用于存储当前iou阈值下metrics指标下每一个类别的结果
        idx_p, idx_r, idx_score, idx_ap = [], [], [], []
        for idx in id_lst:
            class_idx_p = metrics[idx]['precision']
            class_idx_r = metrics[idx]['recall']
            class_idx_score = metrics[idx]['precision_recall_score']
            class_idx_ap = metrics[idx]['ap']
            idx_p.append(class_idx_p)
            idx_r.append(class_idx_r)
            idx_score.append(class_idx_score)
            idx_ap.append(class_idx_ap)
        p_iou.append(idx_p)
        r_iou.append(idx_r)
        score_iou.append(idx_score)
        ap_iou.append(idx_ap)
    
    # 每一行是不同iou, 每一列是不同idx类别, 按列内元素求和取平均就是该类别在50~95上的P50~95, R50~95, AP50~95
    p_iou = np.array(p_iou)
    r_iou = np.array(r_iou)
    score_iou = np.array(score_iou)
    ap_iou = np.array(ap_iou)

    p50 = p_iou[0, :] # 一行
    r50 = r_iou[0, :] # 一行
    
    score50 = score_iou[0, :] # 一行
    ap50 = ap_iou[0, :] # 一行
 
    p50t95 = np.mean(p_iou, axis=0) # 一行
    r50t95 = np.mean(r_iou, axis=0) # 一行
    ap50t95 = np.mean(ap_iou, axis=0) # 一行

    val_metrics = {'id':id_lst, 'p50':p50, 'r50': r50, 'score50': score50, 
                   'ap50': ap50, 'p50t95': p50t95, 'r50t95': r50t95, 'ap50t95': ap50t95, 'gt': gt_num}
    # 计算验证损失平均值
    if compute_loss and val_batch_num > 0:
        avg_val_total_loss = val_losses['total'] / val_batch_num
        avg_val_cls_loss = val_losses['cls'] / val_batch_num
        avg_val_box_loss = val_losses['box'] / val_batch_num
        avg_val_loc_loss = val_losses['loc'] / val_batch_num
        avg_val_main_cls_loss = val_losses['main_cls'] / val_batch_num
        avg_val_main_box_loss = val_losses['main_box'] / val_batch_num
        avg_val_main_loc_loss = val_losses['main_loc'] / val_batch_num
    else:
        avg_val_total_loss = None
        avg_val_cls_loss = None
        avg_val_box_loss = None
        avg_val_loc_loss = None
        avg_val_main_cls_loss = None
        avg_val_main_box_loss = None
        avg_val_main_loc_loss = None
    # 绘制混淆矩阵
    if drawing:
        # 计算混淆矩阵(通常使用 IoU=0.5)
        conf_mat = compute_confusion_matrix(gt_data_all, pred_data_all, num_classes, iou_threshold=0.5)
        
        # 生成类别名称(若无可从数据集获取, 这里用数字代替)
        class_names = [str(i) for i in range(num_classes)]  # 前景类别名
        # 注意：plot_confusion_matrix 内部会自动加上 'background'
        save_path = os.path.join(draw_dir, f"confusion_matrix_iou_0.5")
        plot_confusion_matrix(conf_mat, class_names=class_names, save_path=save_path)

    return val_metrics, all_class_p50, all_class_r50, all_class_score50, map50, map75, map50t95, \
           avg_val_total_loss, avg_val_cls_loss, avg_val_box_loss, avg_val_loc_loss, \
           avg_val_main_cls_loss, avg_val_main_box_loss, avg_val_main_loc_loss

# 打印结果
def printer_eval(val_metrics, all_p50, all_r50, all_score50, map50, map75, map50t95):
    class_id = val_metrics['id']
    class_gt = val_metrics['gt']
    all_gt = sum(val_metrics['gt'])
    class_num = len(class_id)

    # 修改全局F1计算, 添加零值检查
    if all_p50 + all_r50 == 0:
        all_f1_max50 = 0
    else:
        all_f1_max50 = 2*all_p50*all_r50/(all_p50+all_r50)
    
    print("=" * 80)
    print("YOLO All Classes Metrics")
    print("=" * 80)
    print("Class\tGroundTruth\tPrecision@IoU=0.5\tRecall@IoU=0.5\t\tmAP@IoU=0.5\t\tmAP@IoU=0.5~0.95\tF1(max)-score@IoU=0.5")
    print(f"All\t{all_gt}\t\t{all_p50:.3f}\t\t\t{all_r50:.3f}\t\t\t{map50:.3f}\t\t\t{map50t95:.3f}\t\t\t{all_f1_max50:.3f}-{all_score50:.3f}")
    
    if class_num <= 10:
        print("\n" + "=" * 80)
        print("YOLO Per-Class Metrics")
        print("=" * 80)
        print("Class\tGroundTruth\tPrecision@IoU=0.5\tRecall@IoU=0.5\t\tAP@IoU=0.5\t\tAP@IoU=0.5~0.95\t\tF1(max)-score@IoU=0.5")
        for i in range(class_num):
            # 修改每个类别的 F1 计算，添加零值检查
            if val_metrics['p50'][i] + val_metrics['r50'][i] == 0:
                f1_max50 = 0
            else:
                f1_max50 = 2*val_metrics['p50'][i] * val_metrics['r50'][i]/(val_metrics['p50'][i] + val_metrics['r50'][i])
            print(f"{class_id[i]}\t{class_gt[i]}\t\t{val_metrics['p50'][i]:.3f}\t\t\t"
                f"{val_metrics['r50'][i]:.3f}\t\t\t{val_metrics['ap50'][i]:.3f}\t\t\t"
                f"{val_metrics['ap50t95'][i]:.3f}\t\t\t{round(f1_max50, 3)}-{round(val_metrics['score50'][i],3)}")
    
    print("\n" + "=" * 80)
    print(f"COCO Metrics")
    print("=" * 80)
    print(f"Accumulating evaluation results...")
    print(f"Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {map50t95:.3f}")
    print(f"Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {map50:.3f}")
    print(f"Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {map75:.3f}")
    return None
