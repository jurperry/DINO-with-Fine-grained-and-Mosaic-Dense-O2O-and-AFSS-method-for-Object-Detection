import os
import cv2
import glob
from tqdm import tqdm  # 导入进度条库

def draw_yolo_boxes(image_dir, label_dir, output_dir, class_names=None):
    """
    将YOLO格式的检测框绘制到图片上并保存到新文件夹
    
    参数:
        image_dir: 原始图片文件夹路径
        label_dir: YOLO标签文件夹路径
        output_dir: 输出图片文件夹路径
        class_names: 类别名称列表 (可选)
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    img_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
    image_paths = []
    for fmt in img_formats:
        image_paths.extend(glob.glob(os.path.join(image_dir, fmt)))
    
    print(f"找到 {len(image_paths)} 张图片")
    
    # 如果没有图片则退出
    if len(image_paths) == 0:
        print("错误: 未找到任何图片，请检查路径和图片格式")
        return
    
    # 颜色列表 (BGR格式)
    colors = [
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 红色
        (255, 0, 0),    # 蓝色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 紫色
        (255, 255, 0),  # 青色
        (0, 165, 255),  # 橙色
        (128, 0, 128),  # 深紫色
        (0, 128, 128),  # 橄榄色
        (255, 192, 203), # 粉色
        (165, 42, 42),   # 棕色
        (255, 165, 0)    # 橙黄色
    ]
    
    # 初始化进度条
    progress_bar = tqdm(total=len(image_paths), desc="处理图片", unit="张")
    
    # 统计信息
    processed_count = 0
    skipped_images = 0
    skipped_labels = 0
    boxes_drawn = 0
    
    for img_path in image_paths:
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"\n警告: 无法读取图片 {img_path}")
            skipped_images += 1
            progress_bar.update(1)
            continue
            
        img_h, img_w = img.shape[:2]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base_name + ".txt")
        
        # 检查标签文件是否存在
        if not os.path.exists(label_path):
            # 保存没有标注的原始图片
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, img)
            skipped_labels += 1
            processed_count += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                "已处理": processed_count, 
                "无标签": skipped_labels,
                "跳过图片": skipped_images,
                "检测框": boxes_drawn
            })
            continue
        
        # 读取并处理标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 处理每个检测框
        for line in lines:
            data = line.strip().split()
            if len(data) < 5:
                continue
                
            try:
                # 解析YOLO格式数据
                class_id = int(data[0])
                x_center = float(data[1]) * img_w
                y_center = float(data[2]) * img_h
                width = float(data[3]) * img_w
                height = float(data[4]) * img_h
                
                # 计算矩形框坐标
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # 确保坐标在图片范围内
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))
                
                # 选择颜色 (循环使用颜色列表)
                color = colors[class_id % len(colors)]
                
                # 绘制矩形框
                thickness = 2
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                boxes_drawn += 1
                
                # 准备标签文本
                label = str(class_id)
                if class_names and class_id < len(class_names):
                    label = class_names[class_id]
                
                # 添加类别标签
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                text_thickness = 2
                
                # 计算文本大小
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
                
                # 绘制文本背景
                bg_y1 = max(0, y1 - text_h - 10)
                bg_y2 = max(0, y1)
                if bg_y1 < bg_y2:  # 确保背景框高度有效
                    cv2.rectangle(
                        img, 
                        (x1, bg_y1), 
                        (x1 + text_w, bg_y2), 
                        color, 
                        -1  # 填充矩形
                    )
                
                # 绘制文本
                text_y = max(5, y1 - 5)  # 确保文本位置在图像范围内
                cv2.putText(
                    img, 
                    label, 
                    (x1, text_y), 
                    font, 
                    font_scale, 
                    (255, 255, 255),  # 白色文本
                    text_thickness
                )
                
            except Exception as e:
                print(f"\n处理 {label_path} 时出错: {e}")
        
        # 保存结果图片
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, img)
        processed_count += 1
        
        # 更新进度条信息
        progress_bar.update(1)
        progress_bar.set_postfix({
            "已处理": processed_count, 
            "无标签": skipped_labels,
            "跳过图片": skipped_images,
            "检测框": boxes_drawn
        })
    
    # 关闭进度条
    progress_bar.close()
    
    # 打印最终统计信息
    print(f"\n处理完成!")
    print(f"总图片数: {len(image_paths)}")
    print(f"成功处理: {processed_count}")
    print(f"跳过图片: {skipped_images} (无法读取)")
    print(f"无标签文件: {skipped_labels}")
    print(f"绘制的检测框总数: {boxes_drawn}")
    print(f"结果已保存到: {output_dir}")

if __name__ == "__main__":
    # 设置您的路径
    imgdir_path = "your/image/path"
    txtdir_path = "your/yolo_like/txt/path"
    output_dir = "true_boxes_plot/output/path"
    
    # 如果有类别名称, 可以在这里提供name list
    class_names = ['class 0 ', 'class 1', ...]
    # class_names = None  # 如果没有类别名称，设为None
    
    # 调用函数
    draw_yolo_boxes(
        image_dir=imgdir_path,
        label_dir=txtdir_path,
        output_dir=output_dir,
        class_names=class_names
    )
