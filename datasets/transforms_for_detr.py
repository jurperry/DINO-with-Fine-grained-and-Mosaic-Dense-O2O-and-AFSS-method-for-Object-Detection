# 本脚本名:transforms_for_detr.py
# mosaic是字节在原图上做较大裁切, 因此其mosaic一张图里面实例数衡多是吗密集监督
import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

# 绝对坐标转化, 把yolo归一化的标注框cxcywh数据转化为绝对坐标xyxy
# 处理的是一张图片内的标注框
def yolo_to_xyxy(boxes, img_size):
    """
    将YOLO格式[class,cx,cy,w,h]转换为[class,x1,y1,x2,y2]
    boxes: [N, 5], 每一行[class_id, cx, cy, w, h]
    img_size: tuple or list, (orign_w, orign_h)
    """
    if len(boxes) == 0:
        # 标注框数量为0, 则返回一个空张量, 但是其维度为dim=(0, 5)
        return torch.zeros((0, 5))
    
    boxes = boxes.clone()
    cx, cy, w, h = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    img_w, img_h = img_size
    
    # x1~classes, 均为dim=(N,)
    x1 = (cx - w/2) * img_w 
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    classes = boxes[:, 0] 
    # torch.stack给所有张量添加一个维度dim=1, 然后再第一维度上拼接, dim->(N, 5)
    return torch.stack([classes, x1, y1, x2, y2], dim=1)

# 绝对坐标xyxy转化为归一化yolo坐标cxcywh
# 处理的是一张图片内的标注
def xyxy_to_yolo(boxes, img_size):
    """
    将[class,x1,y1,x2,y2]转换为YOLO格式[class,cx,cy,w,h]
    boxes: [N, 5], 每一行abs_bbox: [class_id, x1, y1, x2, y2]
    img_size: tuple or list, (orign_w, orign_h)
    """
    if len(boxes) == 0:
        # 标注框数量为0, 则返回一个空张量, 但是其维度为dim=(0, 5)
        return torch.zeros((0, 5))
    
    x1, y1, x2, y2 = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    img_w, img_h = img_size
    
    # cx~classes, 均为dim=(N,)
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    classes = boxes[:, 0] # 处理类别信息
    # torch.stack给所有张量添加一个维度dim=1, 然后再第一维度上拼接, dim->(N, 5)
    return torch.stack([classes, cx, cy, w, h], dim=1)

def get_crop_params(img: torch.Tensor, target: torch.Tensor, output_size: tuple[int, int], trails: int = 40) -> tuple[int, int, int, int]:
    """
    Get parameters for ``crop`` for a smart random crop that ensures at least one bounding box is included.

    Args:
        img (Tensor): Image to be cropped.
        target (Tensor): Target boxes in format [N, 5] where each row is [class_id, cx, cy, w, h].
        output_size (tuple): Expected output size of the crop.
        max_attempts (int): Maximum number of attempts to find a crop that includes at least one bounding box.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a smart random crop.
    """
    _, h_img, w_img = F.get_dimensions(img)
    th, tw = output_size

    # Convert YOLO format targets to absolute coordinates
    boxes_abs = yolo_to_xyxy(target, (w_img, h_img))
    
    for _ in range(trails):
        i = torch.randint(0, h_img - th + 1, size=(1,)).item() # y 坐标
        j = torch.randint(0, w_img - tw + 1, size=(1,)).item() # x 坐标

        # Calculate the crop region
        x_min, y_min, x_max, y_max = j, i, j + tw, i + th
        
        # Check if any bounding box is within this crop region
        valid_boxes = ((boxes_abs[:, 1] >= x_min) & (boxes_abs[:, 3] <= x_max) &
                       (boxes_abs[:, 2] >= y_min) & (boxes_abs[:, 4] <= y_max))
        
        if valid_boxes.any():
            return i, j, th, tw
    # 如果尝试40次依旧没有框, 则直接返回左上角裁切
    return 0, 0, th, tw

# 裁切函数
def crop(image, target, region, eps=1e-3):
    """
    DETR裁剪
    image:PIL处理的图片或torch.Tensor
    target: yolo标注, dim=(N, 5)
    region: 裁切区块的左上角y, 左上角x, 高h和宽w, 且均为绝对坐标
    """
    cropped_image = F.crop(image, *region)
    # i, j, h, w分别代表裁切区块的左上角y, 左上角x, 高h和宽w, 且均为绝对坐标
    i, j, h, w = region
    crop_area = h * w

    # DETR训练必须保证有标签
    assert target is not None and len(target) > 0, "DETR needs labels in a single image"

    # 转换标签格式
    # 将YOLO格式转换为绝对坐标boxes_abs=(x1,y1,x2,y2)
    if isinstance(image, torch.Tensor):
        img_h, img_w = image.shape[-2:]
    else:
        img_w, img_h = image.size
    # boxes_abs->dim=(N, 5)
    boxes_abs = yolo_to_xyxy(target, (img_w, img_h))
    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    
    # 应用裁剪变换, 确保边界框坐标处于裁切区域坐标系内(像素坐标, 依旧是绝对坐标)
    # 但是处于裁切区外的边界框会成为裁切区域边界上的一条线或者点(用边界框面积过滤这些线或者点)
    cropped_boxes = boxes_abs[:, 1:] - torch.tensor([j, i, j, i]) # (N, x1y1x2y2=4)
    cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size) # (N, x1y1=2, x2y2=2)
    cropped_boxes = cropped_boxes.clamp(min=0)
    area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1) # (N, )

    # 过滤无效框(只保留面积大于eps*裁切区域面积的框)
    valid = area > eps * crop_area
    # DETR必须保证至少有一个有效目标
    if valid.sum() == 0:
        # 如果裁剪后没有有效目标，返回原始图像和标签
        return image, target
    
    # 处理有效目标
    cropped_boxes = cropped_boxes[valid].reshape(-1, 4) # (M,2,2) ->(M, 4)
    classes = boxes_abs[valid, 0].unsqueeze(-1) # (M,)->(M, 1)
    cropped_bbox_with_class = torch.cat([classes, cropped_boxes], dim=1) # (M, 5), (class, x1, y1, x2, y2)
    new_target = xyxy_to_yolo(cropped_bbox_with_class, (w, h)) #  (class, x1, y1, x2, y2) ->(class, cx, cy, w, h)转换回YOLO格式
    return cropped_image, new_target

def crop_for_mosaic(image, target, region, eps=1e-3):
    """
    DETR裁剪
    image: PIL处理的图片或torch.Tensor
    target: yolo标注, dim=(N, 5)
    region: 裁切区块的左上角y, 左上角x, 高h和宽w, 且均为绝对坐标
    """
    cropped_image = F.crop(image, *region)
    # i, j, h, w分别代表裁切区块的左上角y, 左上角x, 高h和宽w, 且均为绝对坐标
    i, j, h, w = region
    crop_area = h * w
    # 处理无标签图片
    if target is None or len(target) == 0:
        # 如果输入就没有标签, 则裁剪后也没有标签
        return cropped_image, torch.zeros((0, 5), dtype=target.dtype if target is not None else torch.float32) # 返回空标签

    if isinstance(image, torch.Tensor):
        img_h, img_w = image.shape[-2:]
    else:
        img_w, img_h = image.size
    boxes_abs = yolo_to_xyxy(target, (img_w, img_h))
    max_size = torch.as_tensor([w, h], dtype=torch.float32)

    # 应用裁剪变换
    cropped_boxes = boxes_abs[:, 1:] - torch.tensor([j, i, j, i]) # (N, x1y1x2y2=4)
    cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size) # (N, x1y1=2, x2y2=2)
    cropped_boxes = cropped_boxes.clamp(min=0)
    area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1) # (N, )

    # 过滤无效框, 处理有效目标
    valid = area > eps * crop_area
    if valid.sum() > 0: # 确保至少有一个有效框
        cropped_boxes = cropped_boxes[valid].reshape(-1, 4) # (M,2,2) ->(M, 4)
        classes = boxes_abs[valid, 0].unsqueeze(-1) # (M,)->(M, 1)
        cropped_bbox_with_class = torch.cat([classes, cropped_boxes], dim=1) # (M, 5), (class, x1, y1, x2, y2)
        new_target = xyxy_to_yolo(cropped_bbox_with_class, (w, h)) #  (class, x1, y1, x2, y2) ->(class, cx, cy, w, h)转换回YOLO格式
    else:
        # 如果裁剪后没有有效目标，则返回一个空的标签张量
        new_target = torch.zeros((0, 5), dtype=target.dtype)

    return cropped_image, new_target # 总是返回裁剪后的图像和对应的（可能为空的）标签


# mosaic函数, 如果4张图拼接后都没有边界框, 则直接返回第一张图像
def mosaic(images, labels):
    '''
    images: 4张图片, list[tensor], tensor(c, hi, wi)
    labels: 4张图片的标签, 尺寸为(img_i_num_gt, id_cx_cy_w_h)
    '''
    # 马赛克拼接4张图片, 图片顺序左上->右上->左下->右下
    img_top_left, img_top_right,\
        img_bottom_left, img_bottom_right = images[0], images[1], images[2], images[3]
    # 均为一张图内(img_i_num_gt, id_cx_cy_w_h)的标签
    labels_top_left, labels_top_right,\
        labels_bottom_left, labels_bottom_right = labels[0], labels[1], labels[2], labels[3]
    
    # 获取mosaic拼图的参数
    channel, h1, w1 = img_top_left.shape
    _, h2, w2 = img_top_right.shape
    _, h3, w3 = img_bottom_left.shape
    _, h4, w4 = img_bottom_right.shape

    # 获取马赛克拼图的尺寸大小
    top_h, bottom_h = min(h1, h2), min(h3, h4)
    left_w, right_w = min(w1, w3), min(w2, w4)

    mosaic_h = top_h + bottom_h
    mosaic_w = left_w + right_w
    # 创建目标尺寸的空白图像
    mosaic_image = torch.empty((channel, mosaic_h, mosaic_w))

    # 获取裁切区域, region: 左上角y, 左上角x, 高h和宽w, 且均为绝对坐标
    region1 = get_crop_params(img_top_left, labels_top_left, [top_h, left_w])
    region2 = get_crop_params(img_top_right, labels_top_right, [top_h, right_w])
    region3 = get_crop_params(img_bottom_left, labels_bottom_left, [bottom_h, left_w])
    region4 = get_crop_params(img_bottom_right, labels_bottom_right, [bottom_h, right_w])

    # 裁切4张图
    crop_img_top_left, crop_label_top_left = crop_for_mosaic(img_top_left, labels_top_left, region1)
    crop_img_top_right, crop_label_top_right = crop_for_mosaic(img_top_right, labels_top_right, region2)
    crop_img_bottom_left, crop_label_bottom_left = crop_for_mosaic(img_bottom_left, labels_bottom_left, region3)
    crop_img_bottom_right, crop_label_bottom_right = crop_for_mosaic(img_bottom_right, labels_bottom_right, region4)

    # 将每张图像放置到适当的位置(马赛克拼图)
    mosaic_image[:, 0:top_h, 0:left_w] = crop_img_top_left  # 左上
    mosaic_image[:, 0:top_h, left_w:mosaic_w] = crop_img_top_right  # 右上
    mosaic_image[:, top_h:mosaic_h, 0:left_w] = crop_img_bottom_left  # 左下
    mosaic_image[:, top_h:mosaic_h, left_w:mosaic_w] = crop_img_bottom_right  # 右下

    # 形成class_ids标签
    label_ids = torch.cat([
        crop_label_top_left[:, 0],
        crop_label_top_right[:, 0],
        crop_label_bottom_left[:, 0],
        crop_label_bottom_right[:, 0]
    ], dim=0) # (4*img_i_num_gt,)
    # 形成bboxes标签
    bbox_top_left = torch.as_tensor(crop_label_top_left[:, 1:], dtype=torch.float32)
    bbox_top_right = torch.as_tensor(crop_label_top_right[:, 1:], dtype=torch.float32)
    bbox_bottom_left = torch.as_tensor(crop_label_bottom_left[:, 1:], dtype=torch.float32)
    bbox_bottom_right = torch.as_tensor(crop_label_bottom_right[:, 1:], dtype=torch.float32)

    # 左上角图片标签转换
    bbox_top_left[:, 0] = bbox_top_left[:, 0] * left_w/mosaic_w
    bbox_top_left[:, 1] = bbox_top_left[:, 1] * top_h/mosaic_h
    bbox_top_left[:, 2] = bbox_top_left[:, 2] * left_w/mosaic_w
    bbox_top_left[:, 3] = bbox_top_left[:, 3] * top_h/mosaic_h

    # 右上角图片标签转换
    bbox_top_right[:, 0] = (bbox_top_right[:, 0] * right_w + left_w) / mosaic_w
    bbox_top_right[:, 1] = bbox_top_right[:, 1] * top_h/mosaic_h
    bbox_top_right[:, 2] = bbox_top_right[:, 2] * right_w/mosaic_w
    bbox_top_right[:, 3] = bbox_top_right[:, 3] * top_h/mosaic_h

    # 左下角图片标签转换
    bbox_bottom_left[:, 0] = bbox_bottom_left[:, 0] * left_w / mosaic_w
    bbox_bottom_left[:, 1] = (bbox_bottom_left[:, 1] * bottom_h + top_h) / mosaic_h
    bbox_bottom_left[:, 2] = bbox_bottom_left[:, 2] * left_w / mosaic_w
    bbox_bottom_left[:, 3] = bbox_bottom_left[:, 3] * bottom_h / mosaic_h
    
    # 右下角图片标签转换
    bbox_bottom_right[:, 0] = (bbox_bottom_right[:, 0] * right_w + left_w) / mosaic_w
    bbox_bottom_right[:, 1] = (bbox_bottom_right[:, 1] * bottom_h + top_h) / mosaic_h
    bbox_bottom_right[:, 2] = bbox_bottom_right[:, 2] * right_w / mosaic_w
    bbox_bottom_right[:, 3] = bbox_bottom_right[:, 3] * bottom_h / mosaic_h

    # 合并所有bboxes标签
    label_bboxes = torch.cat([
        bbox_top_left,
        bbox_top_right,
        bbox_bottom_left,
        bbox_bottom_right
    ], dim=0)

    if label_ids.numel() == 0:
        fail = True
        return images[0], labels[0], fail
    else:
        mosaic_labels = torch.cat([label_ids.unsqueeze(1), label_bboxes], dim=1)
        fail = False
        return mosaic_image, mosaic_labels, fail



def hflip(image, target):
    """
    DETR的水平翻转
    image: PIL处理得到的格式, 可以被torch接受, size:(orign_w, orign_h)
    target: yolo格式数据 dim=(N, 5)的tensor, 每一行为(class, cx, cy, w, h)且为归一化数据
    flipped_image: PIL格式数据, 已经水平翻转过了
    new_target: yolo格式数据 dim=(N, 5)的tensor, 每一行为(class, cx, cy, w, h)且为归一化数据, , 已经水平翻转过了
    """
    flipped_image = F.hflip(image)
    
    # DETR训练必须保证有标签
    assert target is not None and len(target) > 0, "DETR needs labels in a single image"
    # YOLO格式直接处理
    new_target = target.clone()
    new_target[:, 1] = 1.0 - new_target[:, 1]  # cx = 1 - cx
    return flipped_image, new_target

# 适配YOLO格式的resize - YOLO格式不需要改变
def resize(image, target, size, max_size=None):
    """
    image: PIL处理得到的格式, 可以被torch接受, size:(orign_w, orign_h)
    target: yolo格式数据 dim=(N, 5)的tensor, 每一行为(class, cx, cy, w, h)且为归一化数据
    size: can be min_size (scalar) or (w, h) tuple
    """
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    new_size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, new_size)
    
    # 这个地方返回值的rescaled_image是训练、验证、预测都存在, target则是在预测的时候不存在
    return rescaled_image, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target # image, target

class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size) # image, target

class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        # 确保裁切尺度处于self.min_size~min(img.width or img.height, self.max_size), 防止超出图像尺寸的裁切
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        # 通过get_crop_params(img, target, [h, w])获取到region=i,j,h,w尺寸(绝对坐标), 分别为裁切后的y,x,h,w坐标
        region = get_crop_params(img, target, [h, w])
        return crop(img, target, region) # image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        # 只对图像进行归一化，YOLO标注已经是归一化格式
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target # image, target

class ToTensor(object):
    def __call__(self, img, target=None):
        return F.to_tensor(img), target

class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

# 颜色扰动
class RandomColorJitter(object):
    """
    对图像应用随机颜色抖动增强，不影响标签。

    Args:
        brightness (float or tuple of float (min, max)): 控制亮度调整的幅度。
            如果是单个数字，则亮度范围为 (1 - brightness, 1 + brightness)。
            如果是元组 (min, max)，则亮度在该范围内随机选择。
        contrast (float or tuple of float (min, max)): 控制对比度调整的幅度。
            如果是单个数字，则对比度范围为 (1 - contrast, 1 + contrast)。
            如果是元组 (min, max)，则对比度在该范围内随机选择。
        saturation (float or tuple of float (min, max)): 控制饱和度调整的幅度。
             如果是单个数字，则饱和度范围为 (1 - saturation, 1 + saturation)。
             如果是元组 (min, max)，则饱和度在该范围内随机选择。
        hue (float or tuple of float (min, max)): 控制色调调整的幅度。
            如果是单个数字，则色调范围为 (-hue, hue)。
            如果是元组 (min, max)，则色调在该范围内随机选择。
        p (float): 应用此变换的概率，默认为 0.5。
    """
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1, p=0.5):
        # 存储概率
        self.p = p
        # 使用 torchvision.transforms.ColorJitter 创建抖动变换实例
        self.color_jitter_transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): 输入图像。
            target (Tensor): 输入标签 (N, 5)，格式为 [class_id, cx, cy, w, h]。

        Returns:
            tuple: (变换后的图像, 原始标签)。标签保持不变。
        """
        # 以概率 p 应用颜色抖动
        if random.random() < self.p:
            # 只对图像应用变换
            jittered_img = self.color_jitter_transform(img)
            return jittered_img, target
        # 不应用变换时，返回原图和原标签
        return img, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    # 这个是调用打印操作
    # 如：transform=Compose(...), print(Compose)就可以打印出内部的调用
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
    
