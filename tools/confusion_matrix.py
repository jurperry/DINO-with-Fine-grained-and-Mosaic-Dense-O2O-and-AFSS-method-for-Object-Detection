import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils import calculate_iou

# 混淆矩阵计算得到的P, R值是置信度为0.001, IoU=0.5情况下的所有预测框的P, R值
# 这个P,R值与验证计算AP时使用的P, R值不同, 验证计算mAP时使用的f1_max下对应的P, R值
# 计算混淆矩阵的时, 训练集与验证集的类别数应该相同, 逻辑会处理验证集类别少于训练集的情况

def compute_confusion_matrix(gt_data, pred_data, num_classes, iou_threshold=0.5):
    """
    计算混淆矩阵
    参数:
        gt_data: {img_id: {class_id: [[x1,y1,x2,y2], ...]}}
        pred_data: {img_id: {class_id: [(bbox, score), ...]}}
        num_classes: 前景类别数
        iou_threshold: 匹配所用的 IoU 阈值
    返回:
        conf_mat: 形状 (num_classes+1, num_classes+1) 的整型矩阵，最后一行/列对应背景
    """
    conf_mat = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    bg_idx = num_classes  # 背景索引

    # 遍历所有图像（仅处理有真实标注的图像）
    for img_id in tqdm(gt_data.keys(), desc="Compute Confusion Matrix"):
        # 收集当前图像的所有真实框
        gt_boxes = []  # 元素: (class_id, [x1,y1,x2,y2])
        for cls_id, bboxes in gt_data[img_id].items():
            for bbox in bboxes:
                gt_boxes.append((cls_id, bbox))

        # 收集当前图像的所有预测框（已按置信度排序）
        pred_boxes = []  # 元素: (class_id, [x1,y1,x2,y2], score)
        if img_id in pred_data:
            for cls_id, preds in pred_data[img_id].items():
                for bbox, score in preds:
                    pred_boxes.append((cls_id, bbox, score))

        # 预测框按置信度降序排序（保证高置信度优先匹配）
        pred_boxes.sort(key=lambda x: x[2], reverse=True)

        gt_matched = [False] * len(gt_boxes)  # 真实框是否已被匹配

        # 遍历预测框, 寻找最佳匹配的真实框
        for pred_cls, pred_bbox, score in pred_boxes:
            best_iou = iou_threshold  # 只接受 ≥ 阈值的匹配
            best_gt_idx = -1
            for i, (gt_cls, gt_bbox) in enumerate(gt_boxes):
                if gt_matched[i]:
                    continue
                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou >= best_iou:
                    best_iou = iou
                    best_gt_idx = i
            # best_iou大于iou阈值, 定位正确
            if best_gt_idx >= 0: 
                gt_cls, _ = gt_boxes[best_gt_idx]
                # if gt_cls == pred_cls:
                #     # (1) 正确检测(TP): IoU ≥ 阈值, 真实类别就是预测类别
                #     conf_mat[gt_cls, pred_cls] += 1
                # else:
                #     # (2) 分类错误(FP): IoU ≥ 阈值, 真实类别被误检为预测类别
                #     conf_mat[gt_cls, pred_cls] += 1
                conf_mat[gt_cls, pred_cls] += 1
                # 目的是记录该真实框已被匹配, 避免重复匹配
                gt_matched[best_gt_idx] = True
            # best_iou小于iou阈值, 定位错误
            else:
                # (3) 定位错误(FP): IoU < 阈值, 预测框属于背景, 但pred_cls不是背景
                # 处理剩余的预测框无匹配的情况
                conf_mat[bg_idx, pred_cls] += 1

        # 处理真实框没有被任何预测框匹配的漏检FN情况, 分属于背景类预测
        # 处理剩余的真实框无匹配的情况
        for i, (gt_cls, _) in enumerate(gt_boxes):
            # gt_matched[i] = False, 说明该真实框未被匹配到预测框
            if not gt_matched[i]: 
                conf_mat[gt_cls, bg_idx] += 1

    return conf_mat

def plot_confusion_matrix(conf_mat, iou_threshold=0.5, class_names=None, save_path=None):
    """
    绘图函数
    """
    if class_names is None:
        # 默认用数字表示前景，最后加一个 'background'
        class_names = [str(i) for i in range(conf_mat.shape[0] - 1)] + ['background']
    else:
        class_names = class_names + ['background']
    
    n_classes = conf_mat.shape[0]
    # 归一化混淆矩阵（按行归一化）
    conf_mat_normalized = conf_mat / conf_mat.sum(axis=1, keepdims=True)

    # 调整图形大小和字体大小以提高可读性
    fig_size = max(16, n_classes * 0.5), max(12, n_classes * 0.5)
    annot_font_size = min(max(8, 200//n_classes), 16)  # 动态调整注解字体大小

    # 原始混淆矩阵绘图
    plt.figure(figsize=fig_size)
    sns.heatmap(
        conf_mat,
        annot=True, 
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues',
        cbar=True,
        square=True,
        annot_kws={"size": annot_font_size}  # 设置注解字体大小
    )
    plt.title(f'Confusion Matrix (Original, IoU={iou_threshold})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=0, fontsize=annot_font_size-2)
    plt.yticks(rotation=0, fontsize=annot_font_size-2)
    if save_path:
        plt.savefig(f"{save_path}_original.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 归一化后的混淆矩阵绘图
    plt.figure(figsize=fig_size)
    sns.heatmap(
        conf_mat_normalized,
        annot=True, 
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues',
        cbar=True,
        square=True,
        annot_kws={"size": annot_font_size}  # 设置注解字体大小
    )
    plt.title(f'Confusion Matrix (Normalized, IoU={iou_threshold})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=0, fontsize=annot_font_size-2)
    plt.yticks(rotation=0, fontsize=annot_font_size-2)
    if save_path:
        plt.savefig(f"{save_path}_normalized.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
