import os
import cv2
from tqdm import tqdm

import math
import numpy as np
from pathlib import Path

import torch
from torchvision.ops import nms
from torch.utils.data import DataLoader

from models import GatedAttention_FineGrained_DINO_Swin
from datasets import PredictDataset_for_DETR
from utils import box_cxcywh_to_xyxy, collate_fn

# 生成n个类别的独特颜色（HSV空间均匀分布）
def generate_distinct_colors(num_colors):
    colors = []
    for i in range(num_colors):
        # 在HSV空间均匀分布色调
        hue = i * (180 / num_colors)  # OpenCV的H范围是0-180
        # 固定饱和度和亮度为高值，确保颜色鲜艳
        color_hsv = (hue, 255, 255)
        # 转换为BGR格式
        color_bgr = cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color_bgr))
    return colors

# 一张图像的后处理
def postprocess(per_img_cls_pred, per_img_box_pred, conf_thres=0.5):
    # 对每个查询框, 取类别概率最大的类别作为预测类别
    query_scores, query_classes = torch.max(per_img_cls_pred, dim=-1)
    # 分别赋值
    pred_score_i = query_scores # (num_queries,)
    pred_cls_i = query_classes # (num_queries,)
    pred_bbox_i = per_img_box_pred # (num_queries, 4)
    # 筛选概率大于conf_thres的预测框
    valid_mask = pred_score_i >= conf_thres
    # 取 top-100
    # 保留置信度最高的前100个预测框 (maxDets=100), 如果预测框不足100个, 则保留所有预测框, 这是切片的特性
    topk = min(100, valid_mask.sum().item())
    pred_topk_scores, top_idx = torch.topk(pred_score_i[valid_mask], topk, sorted=True)
    pred_topk_labels = pred_cls_i[valid_mask][top_idx] # (valid_queries,)
    pred_topk_bboxes = pred_bbox_i[valid_mask][top_idx] # (valid_queries, 4)
    return pred_topk_labels, pred_topk_bboxes, pred_topk_scores

# nms函数, 用于非极大值抑制检测, 防止检测框重复
def apply_nms(pred_labels, pred_boxes, pred_scores, nms_tuple):
    """
    nms应用函数, 可以对指定部分框进行nms
    """
    if nms_tuple[0] == True and len(pred_boxes) > 0:
        # 创建空列表, 用于存储每个类别的过滤结果
        filtered_labels_list = []
        filtered_boxes_list = []
        filtered_score_list = []
        # 从nms_tuple中获取iou_threshold
        iou_threshold = nms_tuple[1]
        # 获取所有存在的类别
        unique_classes = torch.unique(pred_labels)
        # 对每个类别单独进行NMS
        for cls in unique_classes:
            # 创建当前类别的掩码
            cls_mask = (pred_labels == cls)
            # 获取当前类别的框、分数和类别
            cls_labels = pred_labels[cls_mask]
            cls_boxes = pred_boxes[cls_mask]
            cls_scores = pred_scores[cls_mask]
            # 对当前类别应用NMS, 一定有框
            keep = nms(cls_boxes, cls_scores, iou_threshold)
            # 添加到结果列表
            filtered_labels_list.append(cls_labels[keep])
            filtered_boxes_list.append(cls_boxes[keep])
            filtered_score_list.append(cls_scores[keep])
        # 如果存在有效检测结果, 合并所有类别
        if filtered_boxes_list:
            filtered_labels = torch.cat(filtered_labels_list)
            filtered_boxes = torch.cat(filtered_boxes_list)
            filtered_score = torch.cat(filtered_score_list)
        return filtered_labels, filtered_boxes, filtered_score
    else:
        return pred_labels, pred_boxes, pred_scores

# 用于GatedAttention_FineGrained_DINO_Swin模型的预测
def predict_func(imgdir, 
                 checkpoint_path, 
                 device, 
                 class_file=None, 
                 output_dir=None, 
                 predict_plot_dir=None, 
                 conf_thres=0.5, 
                 num_classes=80, 
                 num_queries=300, 
                 num_encoder_layer=6, 
                 num_decoder_layer=6, 
                 batch_size=4, 
                 num_workers=0, 
                 gate_attn=True, 
                 pin_memory=True, 
                 NMS=(False, 0.5),
                 max_size=640, 
                 val_size=640,
                 use_ema=True,):
    # 如果提供了绘图目录，确保它存在
    if predict_plot_dir is not None:
        os.makedirs(predict_plot_dir, exist_ok=True)
        # 读取类别文件
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        # 为所有类别生成独特颜色
        num_classes_actual = len(class_names)
        class_colors = generate_distinct_colors(num_classes_actual)
    
    if output_dir != None:
        print(f"Write predictions into {output_dir} folder, save as txt file.")
    print(f"Predicting and plotting in real-time...\n")
    
    predict_dataset = PredictDataset_for_DETR(imgdir, 
                                              image_set='val',
                                              max_size=max_size,
                                              val_size=val_size)
    predict_dataloder = DataLoader(predict_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=False, 
                                        collate_fn=collate_fn,
                                        num_workers=num_workers, 
                                        pin_memory=pin_memory)
    
    # 预测模型
    predict_model = GatedAttention_FineGrained_DINO_Swin(num_queries=num_queries, 
                                                         num_classes=num_classes,
                                                         num_encoder_layer=num_encoder_layer, 
                                                         num_decoder_layer=num_decoder_layer, 
                                                         gate_attn=gate_attn,).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if use_ema:
        # 由于ema保存的是一个字典, 因此还需要module获取权重
        model_weights = checkpoint['ema_state_dict']['module']
    else:
        # 原始权重保存仅有module
        model_weights = checkpoint['model_state_dict']
    predict_model.load_state_dict(model_weights)
    predict_model.eval()
    
    # 获取目标目录中所有图片文件, 显式的加入加载顺序(一定要进行sorted筛选, 保证数据加载与PredictDataset_for_DETR的顺序一致)
    # 因为windows是默认从大到小顺序加载, 相当于使用了sorted, 而linux系统不是顺序加载, 会出错
    # 使用pathlib确保Windows下中文文件名编码正确
    img_name_lst = sorted([f.name for f in Path(imgdir).iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')])
    
    with torch.no_grad():
        all_predictions = []
        # 添加推理进度条
        for batch_idx, batch in enumerate(tqdm(predict_dataloder, desc="Inference Progress", total=len(predict_dataloder))):
            nested_tensor, _ = batch
            imgs, masks = nested_tensor.tensors, nested_tensor.mask

            # 进行预测推理
            last_layer_cls, last_layer_bbox, heatmap \
                = predict_model(imgs.to(device), masks.to(device), return_all_layers=False, draw_heatmap=False)
            # cls_logits与box_sigmoid值转化
            batch_cls_pred = last_layer_cls.sigmoid()             # shape: (batch_size, num_queries, num_classes)
            batch_box_pred = box_cxcywh_to_xyxy(last_layer_bbox)  # shape: (batch_size, num_queries, 4)
            
            for i in range(batch_cls_pred.size(0)):
                img_idx = batch_idx * batch_size + i
                if img_idx >= len(img_name_lst):
                    break
                
                imgname_str = img_name_lst[img_idx]
                img_path = os.path.join(imgdir, imgname_str)
                # 单一图片的topk筛选的postprocess
                pred_topk_labels, pred_topk_bboxes, pred_topk_scores \
                    = postprocess(batch_cls_pred[i], batch_box_pred[i], conf_thres)
                # 是否使用nms
                single_img_preds = apply_nms(pred_topk_labels, pred_topk_bboxes, pred_topk_scores, NMS)
                
                # 反归一化坐标并绘制图像
                # 使用np.fromfile + cv2.imdecode解决Windows中文文件名问题
                img_data = np.fromfile(img_path, dtype=np.uint8)
                img_real = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img_real is None:
                    print(f"Error: Could not read image {img_path}")
                    continue
                h, w = img_real.shape[:2]
                abs_wh = torch.tensor([w, h, w, h], device=device)
                # 存储反归一化后的结果
                denorm_results = []
                txt_lines = []

                # xyxy_norm坐标反归一化, 转化为voc绝对坐标
                per_img_labels, per_img_boxes, per_img_scores = single_img_preds
                per_img_boxes_abs = per_img_boxes * abs_wh
                
                # 绘制检测框
                for cls_id, bbox_norm, bbox_abs, score in zip(per_img_labels, per_img_boxes, per_img_boxes_abs, per_img_scores):
                    # 归一化坐标, 没有必要使用, 但是保留接口
                    x1, y1, x2, y2 = bbox_norm.tolist()
                    # 反归一化坐标
                    x1_abs, y1_abs, x2_abs, y2_abs = bbox_abs.tolist()

                    # 转换为整数坐标, 用于图像绘制
                    xmin_i = int(round(x1_abs))
                    ymin_i = int(round(y1_abs))
                    xmax_i = int(round(x2_abs))
                    ymax_i = int(round(y2_abs))
                    
                    # 存储结果, 需要保存高精度的坐标
                    denorm_results.append([
                        cls_id.item(), 
                        round(x1_abs, 6), 
                        round(y1_abs, 6), 
                        round(x2_abs, 6), 
                        round(y2_abs, 6),
                        round(score.item(), 2)
                    ])
                    
                    # 准备写入txt文件的内容
                    if output_dir is not None:
                        txt_lines.append(f"{cls_id.item()} {x1_abs:.6f} {y1_abs:.6f} {x2_abs:.6f} {y2_abs:.6f} {score.item():.2f}")
                    
                    # 绘制检测框
                    if predict_plot_dir is not None:
                        # 获取类别名称和颜色
                        class_id_int = cls_id.item()
                        class_name = class_names[class_id_int] if class_id_int < num_classes_actual else f"Class_{class_id_int}"
                        color = class_colors[class_id_int] if class_id_int < num_classes_actual else (0, 0, 0)
                        
                        # 计算基础缩放因子（使用对角线长度作为基准）
                        diag = math.sqrt(h**2 + w**2)
                        base_scale = diag / 1200  # 1200 是经验值，可根据需求调整

                        # 计算自适应矩形框线宽
                        line_thickness = max(1, min(8, int(round(base_scale * 3))))
                        
                        # 绘制矩形框
                        cv2.rectangle(img_real, (xmin_i, ymin_i), (xmax_i, ymax_i), color, line_thickness)
                        
                        # 准备标签文本 (类别名称 + 置信度)
                        label_text = f"{class_name}: {score.item():.2f}"
                        
                        # 动态计算字体参数
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_thickness = max(1, min(8, int(round(base_scale * 3))))
                        font_scale = max(0.4, min(1.5, base_scale * 1.2))
                        
                        # 计算文本尺寸
                        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
                        
                        # 创建文本背景矩形
                        bg_rect_top_left = (xmin_i, ymin_i - text_height - 5)
                        bg_rect_bottom_right = (xmin_i + text_width, ymin_i)
                        
                        # 绘制文本背景
                        cv2.rectangle(img_real, bg_rect_top_left, bg_rect_bottom_right, color, -1)  # 填充矩形
                        cv2.rectangle(img_real, bg_rect_top_left, bg_rect_bottom_right, color, 2)  # 边框
                        
                        # 绘制文本
                        text_origin = (xmin_i, ymin_i - 5)
                        cv2.putText(img_real, label_text, text_origin, font, font_scale, (255, 255, 255), text_thickness)
                
                # 保存绘制后的图像
                if predict_plot_dir is not None:
                    output_img_path = os.path.join(predict_plot_dir, imgname_str)
                    ext = os.path.splitext(imgname_str)[1]
                    _, img_encoded = cv2.imencode(ext, img_real)
                    img_encoded.tofile(output_img_path)
                
                # 保存预测结果到txt文件
                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    txt_filename = os.path.splitext(imgname_str)[0] + ".txt"
                    with open(os.path.join(output_dir, txt_filename), 'w') as f:
                        for line in txt_lines:
                            f.write(line + '\n')
                
                # 将结果添加到预测字典
                txt_filename = os.path.splitext(imgname_str)[0] + ".txt"
                all_predictions.append((txt_filename, denorm_results))
            
        # 将预测结果转换为字典, 方便后续调用
        denorm_dict = {filename: results for filename, results in all_predictions}
        return denorm_dict

if __name__ =="__main__":
    # 文件读取区
    # sign3_small
    # class_file = r"E:\python_project\machine_learning02\DEIM-main\SignTest\classes.txt"
    # test_imgdir = r"E:\python_project\machine_learning02\DEIM-main\SignTest\images"
    # predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\signtest"
    # model_path = r"E:\python_project\paper_project\AFSS_DINO\results\sign_small\best_checkpoint.pth"

    # class_file = r"E:\python_project\paper_project\deformable_detr\signtest\classes.txt"
    # test_imgdir = r"E:\python_project\machine_learning02\DEIM-main\SignVal\images"
    # predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\signval"
    # model_path = r"E:\python_project\paper_project\AFSS_DINO\results\sign_big\best_checkpoint.pth"

    # class_file = r"E:\python_project\paper_project\deformable_detr_yolo\yolo_sign3\classes.txt"
    # test_imgdir = r"E:\python_project\paper_project\deformable_detr_yolo\yolo_sign3\images\val"
    # predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\sign_val"
    # model_path = r"E:\python_project\paper_project\AFSS_DINO\results\sign_small\best_checkpoint.pth"

    # class_file = r'E:\python_project\machine_learning01\DetectionTransformer\classes_coco.txt'
    # test_imgdir = r"E:\python_project\machine_learning01\DetectionTransformer\assets"
    # predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\assets_box"
    # model_path = r"E:\python_project\paper_project\AFSS_DINO\results\coco_small\best_checkpoint.pth"

    # class_file = r'E:\python_project\machine_learning01\DetectionTransformer\classes_coco.txt'
    # test_imgdir = r"E:\python_project\machine_learning01\DetectionTransformer\assets"
    # predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\assets_box"
    # model_path = r"E:\python_project\paper_project\AFSS_DINO\results\pretrained\best_checkpoint.pth"

    # class_file = r'E:\python_project\machine_learning01\DetectionTransformer\classes_coco.txt'
    # test_imgdir = r"E:\python_project\machine_learning01\DetectionTransformer\images\test"
    # predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\coco_small"
    # model_path = r"E:\python_project\paper_project\AFSS_DINO\results\pretrained\best_checkpoint.pth"

    # china road damage motobike, num_classes=5
    # class_file = r'E:\python_project\paper_project\gated_fine_dino\china_road_motobike_damage\classes.txt'
    # test_imgdir = r"E:\python_project\paper_project\gated_fine_dino\china_road_motobike_damage\images\test"
    # predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\road_damage_china_motobike"
    # model_path = r'E:\python_project\paper_project\AFSS_DINO\results\road_damage_china_motobike\best_checkpoint.pth'

    # china road damage drone, num_classes=6
    # class_file = r'E:\python_project\paper_project\gated_fine_dino\china_road_drone_damage\classes.txt'
    # test_imgdir = r"E:\python_project\paper_project\gated_fine_dino\china_road_drone_damage\images\test"
    # predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\road_damage_china_drone"
    # model_path = r"E:\python_project\paper_project\AFSS_DINO\results\road_damage_drone_china\best_checkpoint.pth"

    # Noway road damage, num_classes=4
    # class_file = r"E:\python_project\machine_learning02\DEIM-main\DamageTrain\classes.txt"
    # test_imgdir =  r"E:\python_project\machine_learning02\DEIM-main\DamageVal\images"
    # predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\road_damage_Noway"
    # model_path = r"E:\python_project\paper_project\AFSS_DINO\results\road_damage_Noway\best_checkpoint.pth"

    # road damage 2classes, num_classes=2
    # class_file = r"E:\python_project\paper_project\gated_fine_dino\road_damage_2class\classes.txt"
    # test_imgdir =  r"E:\python_project\paper_project\gated_fine_dino\road_damage_2class\images\test"
    # predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\road_damage_2class"
    # model_path = r"E:\python_project\paper_project\AFSS_DINO\results\road_damage_2class\best_checkpoint.pth"
    
    # road damage potholes, num_classes=2 
    # class_file = r"E:\python_project\paper_project\pothole_imgs\classes.txt"
    # test_imgdir = r"E:\python_project\paper_project\AFSS_DINO\pothole_damage\images\test"
    # predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\damage_potholes"
    # model_path = r"E:\python_project\paper_project\AFSS_DINO\results\road_damage_pothole\best_checkpoint.pth"

    # class_file = r"E:\python_project\paper_project\pothole_imgs\classes.txt"
    # test_imgdir = r"E:\python_project\paper_project\gated_fine_dino\test_pothole"
    # predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\test_potholes"
    # model_path = r"E:\python_project\paper_project\AFSS_DINO\results\road_damage_pothole\best_checkpoint.pth"

    class_file = r'E:\python_project\machine_learning01\DetectionTransformer\classes_coco.txt'
    test_imgdir = r"E:\python_project\machine_learning01\DetectionTransformer\assets"
    predict_pic_dir = r"E:\python_project\paper_project\AFSS_DINO\predict\assets_box"
    model_path = r"E:\python_project\paper_project\AFSS_DINO\results\coco_tiny\best_checkpoint.pth"

    # 模型参数配置
    num_classes = 80
    num_queries = 300
    gate_attn = True # 是否使用门控自注意力机制
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dict_unormalize_file = predict_func(
        imgdir=test_imgdir, 
        checkpoint_path=model_path, 
        device=device, 
        conf_thres=0.40,
        num_classes=num_classes, 
        num_queries=num_queries, 
        num_encoder_layer=6, 
        num_decoder_layer=6,
        num_workers=0, 
        gate_attn=gate_attn,
        pin_memory=True, 
        batch_size=2, 
        NMS=(True, 0.5), 
        output_dir=None,
        class_file=class_file,
        predict_plot_dir=predict_pic_dir,
        max_size=640,
        val_size=640,
        use_ema=True,
    )