import os
import numpy as np
from collections import defaultdict

from utils import calculate_iou

import matplotlib as mpl
import matplotlib.pyplot as plt

def gt_count_func(gt_data:dict):
    # 统计每个类别的全局真实框数量
    class_gt_counts = defaultdict(int)
    for _, gt_image in gt_data.items():
        for class_id, bboxes in gt_image.items():
            class_gt_counts[class_id] += len(bboxes)
    # 统计验证集的所有类别
    all_gt_classes = set(class_gt_counts.keys())
    all_img_names = set(gt_data.keys())
    return class_gt_counts, all_gt_classes, all_img_names

def calculate_metrics(gt_data:dict, 
                      pred_data:dict, 
                      class_gt_counts:dict, 
                      all_gt_classes:set, 
                      all_img_names:set, 
                      iou_threshold=0.5, 
                      drawing=False, 
                      draw_dir=None):
    if drawing and draw_dir is not None:
        os.makedirs(draw_dir, exist_ok=True)
    elif drawing and draw_dir is None:
        raise ValueError("If you want draw, the draw_dir must be provided when drawing is True!")
    
    class_results = defaultdict(list)
    for image_id in all_img_names:
        gt_image:dict
        pred_image:dict
        gt_image = gt_data[image_id]
        pred_image = pred_data.get(image_id, {})
        
        # 遍历验证集所有类别
        for class_id in all_gt_classes:
            gt_boxes:list
            pred_boxes:list
            
            # 情况1：该图片有该类别的真实框
            if class_id in gt_image:
                gt_boxes = [bbox for bbox in gt_image[class_id]] # voc-xyxy
                pred_boxes = pred_image.get(class_id, [])
                # x[1] = score, pred_boxes = [([x,y,x,y], score), ([x,y,x,y], score), ([x,y,x,y], score)....]
                # 按置信度排序, 高置信度的优先匹配, 只要通过IoU阈值就被认为是TP, 即使低置信度的IoU更高也被认为是FP。
                pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[1], reverse=True)
                gt_matched = [False] * len(gt_boxes)
                
                # (1) pred_boxes_sorted = [], for循环不执行, 该真实类别的预测框不存在, 或者出现验证集类别以外的类别
                # (2) pred_boxes_sorted != [], for循环执行, 判断预测框是tp, fp
                for pred_box, score in pred_boxes_sorted:
                    best_iou = 0.0
                    best_idx = -1
                    
                    # this one pred_box matches all gt_boxes one by one
                    for i, gt_box in enumerate(gt_boxes):
                        if gt_matched[i]:
                            continue
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = i
                    
                    # (1) best_iou = 0 or best_iou < iou_threshold, is_tp = False
                    # (2) best_iou >= iou_threshold, is_tp = True
                    is_tp = best_iou >= iou_threshold
                    class_results[class_id].append({
                        'score': score,
                        'tp': is_tp,
                    })
                    # 当is_tp为True且best_idx有效时, 才将该gt_box标记为已匹配
                    if is_tp and best_idx >= 0:
                        gt_matched[best_idx] = True
            
            # 情况2：该图片没有该类别class_id的真实框gt
            else:
                pred_boxes = pred_image.get(class_id, [])
                # (1) pred_boxes = [], for循环不执行, 该真实类别的预测框不存在, 或者出现验证集类别以外的类别
                # (2) pred_boxes != [], 所有预测框都是误检fp
                for pred_box, score in pred_boxes:
                    class_results[class_id].append({
                        'score': score,
                        'tp': False  # 明确标记为FP
                    })
        # 以上代码处理逻辑
        # (1) 该类别下: pred_box, gt_box均存在, 正常计算tp, fp
        # (2) 该类别下: pred_box=[], gt_box存在, 漏检, 通过total_gt计算fn
        # (3) 该类别下: pred_box存在, gt_box=[], 误检, 直接标记为fp
        # (4) 该类别下: pred_box=[], gt_box=[], 说明检测正常无需担忧(不做标记)
    
    gt_class_metrics = {}
    gt_class_ap_list = [] # 仅存储验证集类别的AP, 方便用于计算验证集类别下的ap
    all_classes_results = [] # all_classes_results用于存储所有验证集类别结果, 方便进行全局tp, fp统计
    
    # 对于验证集中所存在的类别的真实框计算mAP所需, 计算每一个类别的mAP值
    if drawing:
        # 统一设置 rcParams
        plt.rcParams.update({
            'font.family': ['Times New Roman', 'SimSun'],
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'axes.unicode_minus': False,
            'figure.figsize': (12, 8),
            'figure.dpi': 300
        })
        plt.ioff()   # 关闭交互模式
        curves_data = []   # 存储类别 PR 曲线数据
    
    for class_id in all_gt_classes:
        # 用真实gt框类别, 映射预测框类别
        results = class_results[class_id]
        # x={'score':..., 'tp':...}字典, result=[{},{},{}...]是一个可迭代列表
        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # 使用全局真实框数量(tp+fn, 仅验证集中存在的类别)
        class_gt = class_gt_counts[class_id] # 总是>0
        
        tp_cum, fp_cum = 0, 0
        precisions, recalls, confs = [], [], []

        # 求解所有预测框计算出来的P、R值---->制作(P, R)散点图
        # score越大, 过滤掉的pred越多, P分母越小, tp越大, 最终达到1, R的分母是总真实框, 当pred为1, R会很低
        # results_sorted = [] 不运行for循环
        for result in results_sorted:
            # if True: tp就计数+1
            if result['tp']:
                tp_cum = tp_cum + 1
            # if False: fp就计数+1
            else:
                fp_cum = fp_cum + 1
            # 获取每一个置信度下的p, r, conf(score)
            precision_i = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 0
            recall_i = tp_cum / class_gt if class_gt > 0 else 0
            conf_i = result["score"] # 获取此预测框的置信度score

            precisions.append(precision_i)
            recalls.append(recall_i)
            confs.append(conf_i)
        all_classes_results = all_classes_results + results
        
        # iou_thres=iou下, 根据每一个真实存在的类别的(P, R), 依次计算F1值, 当F1最大值时, 返回(P, R), 以及其所对应的score值(conf值)
        f1_lst = []
        for i in range(len(confs)):
            if precisions[i] + recalls[i] == 0:
                f1 = 0
                f1_lst.append(f1)
            else:
                f1 = 2*precisions[i]*recalls[i]/(precisions[i]+recalls[i])
                f1_lst.append(f1)
        
        # 处理验证集中某类别没有预测框, 也就是 pred_box = [], 导致results = [], 这一类的p, r, score 都为 0
        if f1_lst != [] :
            f1_max = max(f1_lst)
            index_max_f1 = f1_lst.index(f1_max)

            f1_max_precision = precisions[index_max_f1]
            f1_max_recall = recalls[index_max_f1]
            f1_max_score = confs[index_max_f1]
        else:
            f1_max_precision = 0
            f1_max_recall = 0
            f1_max_score = 0

        # ap值计算, 使用101点插值法
        if recalls and precisions:
            interp_precisions = [] # interp_precisions存储101个点下的p值
            for t in np.arange(0, 1.01, 0.01):
                prec_at_recall = [p for r, p in zip(recalls, precisions) if r >= t]
                if prec_at_recall:
                    interp_precisions.append(max(prec_at_recall)) # 保证非升曲线, 从大到小排列
                else:
                    interp_precisions.append(0)
        else:
            interp_precisions = [0] * 101
        # 计算ap值
        ap = np.mean(interp_precisions) # 对101个点下的p值求平均, 得到ap值
        gt_class_ap_list.append(ap) # 存储验证集类别下的ap值, 方便后续计算验证集mAP
        
        # Classes PR Curve 数据存贮:
        if drawing:
            curves_data.append((f'Class {class_id} (AP={ap:.3f})', interp_precisions))

        tp_total = sum(1 for r in results_sorted if r['tp'])
        fp_total = sum(1 for r in results_sorted if not r['tp'])

        # 构建每一个类别的p, r, ap值, 以及对应的tp, fp, gt, 其中p, r是根据最大f1动态得出的, ap是所有P-R Curve的积分
        gt_class_metrics[class_id] = {
            'precision': f1_max_precision,
            'recall': f1_max_recall,
            'precision_recall_score': f1_max_score,
            'ap': ap,
            'tp': tp_total,
            'fp': fp_total,
            'gt': class_gt,
        }
    # 循环结束后统一设置并保存, 绘制类别 PR 曲线
    if drawing and curves_data:
        plt.figure()
        for label, precisions in curves_data:
            plt.plot(np.arange(0, 1.01, 0.01), precisions, label=label)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"P-R Curves @ IoU={iou_threshold}")
        plt.legend()
        plt.savefig(os.path.join(draw_dir, f"Classes PR Curve @ IoU={iou_threshold}.png"))
        plt.close()

    # 求解全体class下的P, R计算出的F1=2*P*R/(P+R), F1最大值所对应的(P, R)值与其对应的confs
    all_classes_results_sorted = sorted(all_classes_results, key=lambda x: x['score'], reverse=True)

    total_gt = sum([class_gt_counts[cls_gt] for cls_gt in all_gt_classes])
    all_tp_cum, all_fp_cum = 0, 0
    all_precisions, all_recalls, all_confs = [], [], []
    # 求解全局预测框计算出来的P、R值, score越大, 过滤掉的pred越多, P分母越小, tp越大, 最终达到1, R的分母是总真实框, 当pred为1, R会很低 
    for all_class_result in all_classes_results_sorted:
        # if True: tp就计数+1
        if all_class_result['tp']:
            all_tp_cum = all_tp_cum + 1
        # if False: fp就计数+1
        else:
            all_fp_cum = all_fp_cum + 1
        # 获取每一个置信度下的p, r, conf(score)
        all_precision_i = all_tp_cum / (all_tp_cum + all_fp_cum) if (all_tp_cum + all_fp_cum) > 0 else 0
        all_recall_i = all_tp_cum / total_gt if total_gt > 0 else 0
        all_conf_i = all_class_result["score"] # 获取此预测框的置信度score

        all_precisions.append(all_precision_i)
        all_recalls.append(all_recall_i)
        all_confs.append(all_conf_i)
    
    # iou_thres=iou下, 根据全体数据集的(P, R), 依次计算F1值, 当F1最大值时, 返回(P, R), 以及其所对应的score值(conf值)
    all_f1_lst =[]
    for j in range(len(all_confs)):
        if all_precisions[j] + all_recalls[j] == 0:
            all_f1 = 0
            all_f1_lst.append(all_f1)
        else:
            all_f1 = 2*all_precisions[j]*all_recalls[j]/(all_precisions[j]+all_recalls[j])
            all_f1_lst.append(all_f1)
    # 处理某真实类别没有预测框的情况
    if all_f1_lst != []:
        all_f1_max = max(all_f1_lst)
        all_index_max_f1 = all_f1_lst.index(all_f1_max)

        all_f1_max_precision = all_precisions[all_index_max_f1]
        all_f1_max_recall = all_recalls[all_index_max_f1]
        all_f1_max_score = all_confs[all_index_max_f1]
    else:
        all_f1_max = 0
        all_f1_max_precision = 0
        all_f1_max_recall = 0
        all_f1_max_score = 0
    
    # 求解map_iou值, 这是正确求解方法, 而不是用全局的P, R曲线计算全局AP作为mAP
    # 因为每个类别同等重要, 如果用全局PR曲线, 会导致优势类别主导, 分数较高, 没有意义
    map_iou = np.mean(gt_class_ap_list) if gt_class_ap_list else 0
    
    # 全局指标绘图部分
    # 使用101点插值法P-R曲线, recall >= t
    if drawing:
        if all_recalls and all_precisions:
            all_interp_precisions = []
            for t in np.arange(0, 1.01, 0.01):
                all_prec_at_recall = [p for r, p in zip(all_recalls, all_precisions) if r >= t]
                if all_prec_at_recall:
                    all_interp_precisions.append(max(all_prec_at_recall))
                else:
                    all_interp_precisions.append(0)
        else:
            all_interp_precisions = [0] * 101 # 处理all_recalls与all_precisions为空的情况
        
        # 使用 rc_context 临时设置字体、尺寸和 DPI
        with mpl.rc_context(rc={
            'font.family': ['Times New Roman', 'SimSun'],  # 英文 Times New Roman，中文宋体
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'axes.unicode_minus': False,                   # 正常显示负号
            'figure.figsize': (12, 8),                     # 图像尺寸（宽12英寸，高8英寸）
            'figure.dpi': 300                              # 图像分辨率
        }):
            # 经过101插值法, 实现P-R非降曲线, 全体类别
            plt.figure()
            plt.plot(np.arange(0, 1.01, 0.01), all_interp_precisions)
            plt.xlabel("Recalls")
            plt.ylabel("Precisions")
            plt.title(f"P-R Curve@IoU={iou_threshold}")
            plt.legend([f"mAP@{iou_threshold:.2f}={map_iou:.4f}"])
            plt.savefig(os.path.join(draw_dir, f"P-R Curve@IoU={iou_threshold}.png"))
            plt.close()

            # F1-score 绘制:
            plt.figure()
            all_confs = [1] + all_confs + [0] # 103点, 兼顾首尾
            all_f1_lst = [0] + all_f1_lst + [0] # 103点, 兼顾首尾
            plt.plot(all_confs, all_f1_lst)
            plt.xlabel("Confs")
            plt.ylabel("F1 Score")
            plt.title(f"F1-Confidence Curve@IoU={iou_threshold}")
            plt.legend([f"F1_max={all_f1_max:.4f}@Score={all_f1_max_score:.3f}"])
            plt.savefig(os.path.join(draw_dir, f"F1-Confidence Curve@IoU={iou_threshold}.png"))
            plt.close()
    
    return gt_class_metrics, all_f1_max_precision, all_f1_max_recall, all_f1_max_score, map_iou