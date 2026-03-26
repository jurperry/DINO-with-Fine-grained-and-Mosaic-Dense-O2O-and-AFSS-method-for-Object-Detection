import os
from PIL import Image
import torch
from copy import deepcopy
from torch.utils.data import Dataset
import datasets.transforms_for_detr as T
import random
import torch.nn.functional as F

# 普通detr增强变换
def make_detr_transforms(image_set, 
                         scales=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                         max_size=1333, 
                         val_size=800):
    normalize = T.Compose([
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    # scales是短边尺寸, max_size是长边最大值, val_size是验证集或者预测时候用的短边尺寸
    if image_set == 'train':
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomColorJitter(),
            T.RandomSelect(
                            T.RandomResize(scales, max_size=max_size),
                            T.Compose([
                                    # 保证scales的最小值要比随机裁切值大
                                    T.RandomResize([400, 500, 600]), 
                                    # 裁切左侧为最小裁切值, 右侧为最大裁切值(min(img w or h, max_crop))
                                    T.RandomSizeCrop(384, 600),      
                                    T.RandomResize(scales, max_size=max_size)])
                          ),
            normalize,])
        return transform
    
    if image_set == 'val':
        transform = T.Compose([
            T.RandomResize([val_size], max_size=max_size),
            normalize,
        ])
        return transform
    
    raise ValueError(f'unknown {image_set}')

# mosaic增强变换
def make_mosaic_transforms(image_set, 
                         scales=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                         max_size=1333, 
                         val_size=800):
    normalize = T.Compose([
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    # scales是短边尺寸, max_size是长边最大值, val_size是验证集或者预测时候用的短边尺寸
    if image_set == 'train':
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomColorJitter(),
            T.RandomSelect(
                            T.RandomResize(scales, max_size=max_size),
                            T.Compose([
                                    # 保证scales的最小值要比随机裁切值大
                                    T.RandomResize([400, 500, 600]), 
                                    # 裁切左侧为最小裁切值, 右侧为最大裁切值(min(img w or h, max_crop))
                                    T.RandomSizeCrop(384, 600),      
                                    T.RandomResize(scales, max_size=max_size)])
                          ),
            normalize,])
        return transform
    
    if image_set == 'val':
        transform = T.Compose([
            T.RandomResize([val_size], max_size=max_size),
            normalize,
        ])
        return transform
    
    raise ValueError(f'unknown {image_set}')

# 无增强变换(仅仅做多尺度训练)
def make_simple_transforms(image_set, 
                           scales=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                           max_size=1333, 
                           val_size=800):
    normalize = T.Compose([
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    # scales是短边尺寸, max_size是长边最大值, val_size是验证集或者预测时候用的短边尺寸
    if image_set == 'train':
        transform = T.Compose([T.RandomResize(scales, max_size=max_size),
                               normalize,
                               ])
        return transform
    
    if image_set == 'val':
        transform = T.Compose([
            T.RandomResize([val_size], max_size=max_size),
            normalize,
        ])
        return transform
    
    raise ValueError(f'unknown {image_set}')

def process_txtdata(txt_path):
    with open(txt_path, 'r') as f:
        content = f.readlines()
        boxes = []
        for i in content:
            line = i.strip()
            parts = line.split()# 去除空格
            new_part = [int(parts[x]) if x==0  else float(parts[x])  for x in range(len(parts))]
            boxes.append(deepcopy(new_part))
        boxes = torch.tensor(boxes) # [num_bbox, 5], 第一维度为一张图片中的框数量, 第二维度为每个框的[id, xc, yc, w, h]的归一化数据
        box_ids = boxes[:, 0].long() # tensor(1,0,1,2...), dim = (num_bbox,)
        box_cxcywhs = boxes[:, 1:] # dim=(num_bbox, 4)
        return boxes, box_ids, box_cxcywhs

class TrainDataset_for_DETR(Dataset):
    def __init__(self, 
                 imgdir_path, 
                 txtdir_path, 
                 image_set="train", 
                 mosaic_ratio=0.625, 
                 mosaic_prob=0.5, 
                 total_epochs=50,
                 warmup_epochs=5, 
                 final_finetune_epochs=5,
                 scales=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800], 
                 max_size=1333, 
                 val_size=800):
        super().__init__()
        self.img_lst = []
        self.txt_lst = []
        self.image_set = image_set
        self.use_mosaic_prob = mosaic_prob
        
        # 训练阶段参数
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.final_finetune_epochs = final_finetune_epochs
        self.mosaic_ratio = mosaic_ratio
        
        # 计算各个阶段的epoch数
        self._calculate_stage_epochs()
        
        # 当前epoch（将在训练过程中设置）
        self.current_epoch = 0
        
        # 构建图像和标签路径列表
        self.img_lst = [os.path.join(imgdir_path, f) for f in os.listdir(imgdir_path)]
        self.txt_lst = [os.path.join(txtdir_path, f) for f in os.listdir(txtdir_path)]
        
        # 确保图像和标签匹配, 显式的sort(), 使得加载数据顺序一致
        self.img_lst.sort()
        self.txt_lst.sort()
        assert len(self.img_lst) == len(self.txt_lst), "Images don't match labels"

        # ========= 保存原始列表, 并初始化活动索引 =========
        self.original_img_lst = self.img_lst.copy()
        self.original_txt_lst = self.txt_lst.copy()
        self.active_indices = list(range(len(self.original_img_lst)))
        # =========================================================

        print(f"Dataset initialization completed, a total of {len(self.img_lst)} samples")
        print(f"Training stages: warmup={self.warmup_epochs} epochs, "
              f"mosaic_aug={self.mosaic_epochs} epochs, "
              f"normal_aug={self.normal_aug_epochs} epochs, "
              f"finetune={self.final_finetune_epochs} epochs")
        
        # 普通数据增强变换(水平翻转, 随机裁切, 随机大小)
        self.normal_transforms = make_detr_transforms(image_set=self.image_set, 
                                                    scales=scales, max_size=max_size, val_size=val_size)
        # mosaic增强变换
        self.mosaic_transforms = make_mosaic_transforms(image_set=self.image_set, 
                                                    scales=scales, max_size=max_size, val_size=val_size)
        # 无增强变换(仅做多尺度训练)
        self.simple_transforms = make_simple_transforms(image_set=self.image_set, 
                                                        scales=scales, max_size=max_size, val_size=val_size)

    def _calculate_stage_epochs(self):
        """计算各个训练阶段的epoch数"""
        # 第一阶段：warmup_epochs
        # 第四阶段：final_finetune_epochs
        # 中间阶段：剩余epochs按比例分配
        remaining_epochs = self.total_epochs - self.warmup_epochs - self.final_finetune_epochs
        
        # 根据mosaic_ratio计算马赛克阶段和普通增强阶段的epoch数
        self.mosaic_epochs = int(remaining_epochs * self.mosaic_ratio)
        self.normal_aug_epochs = remaining_epochs - self.mosaic_epochs
        
        # 计算各阶段的起始epoch
        self.mosaic_start_epoch = self.warmup_epochs
        self.mosaic_end_epoch = self.mosaic_start_epoch + self.mosaic_epochs
        self.normal_aug_start_epoch = self.mosaic_end_epoch
        self.normal_aug_end_epoch = self.normal_aug_start_epoch + self.normal_aug_epochs
        self.finetune_start_epoch = self.normal_aug_end_epoch
        
        # 确保最低训练轮次为warm up和finetune epochs的总和
        if self.total_epochs < self.warmup_epochs + self.final_finetune_epochs:
            raise ValueError(f"Total epochs must be at least as many as warm up and finetune epochs!"
                             f" Current total epochs({self.total_epochs}) is less than warm up and finetune epochs combined({self.warmup_epochs + self.final_finetune_epochs})!!!")
        
        print(f"Stage breakdown: ")
        if self.warmup_epochs > 0:
            print(f"  Warmup (no aug): epochs 0-{self.warmup_epochs-1}")
        else:
            print(f"  Warmup (no aug): None")
        if self.mosaic_epochs > 0:
            print(f"  Mosaic aug: epochs {self.mosaic_start_epoch}-{self.mosaic_end_epoch-1}")
        else:
            print(f"  Mosaic aug: None")    
        if self.normal_aug_epochs > 0:
            print(f"  Normal aug: epochs {self.normal_aug_start_epoch}-{self.normal_aug_end_epoch-1}")
        else:
            print(f"  Normal aug: None")
        if self.final_finetune_epochs > 0:
            print(f"  Finetune (no aug): epochs {self.finetune_start_epoch}-{self.total_epochs-1}")
        else:
            print(f"  Finetune (no aug): None")

    def set_epoch(self, epoch):
        """设置当前epoch, 用于决定使用哪种数据增强策略"""
        self.current_epoch = epoch
        
        # 确定当前阶段
        if epoch < self.warmup_epochs:
            print(f"Current stage: warmup")
            self.current_stage = "warmup"
            self.use_mosaic = False
            self.use_detr_aug = False
        elif epoch < self.mosaic_end_epoch:
            print(f"Current stage: mosaic")
            self.current_stage = "mosaic"
            self.use_mosaic = True
            self.use_detr_aug = True
        elif epoch < self.normal_aug_end_epoch:
            print(f"Current stage: normal_aug")
            self.current_stage = "normal_aug"
            self.use_mosaic = False
            self.use_detr_aug = True
        else:
            print(f"Current stage: finetune")
            self.current_stage = "finetune"
            self.use_mosaic = False
            self.use_detr_aug = False
            
        # 设置mosaic概率
        if self.current_stage == "mosaic":
            self.mosaic_prob = self.use_mosaic_prob
        else:
            self.mosaic_prob = 0.0

    # ========= 动态设置当前Epoch所需的子集 =========
    def set_subset(self, indices):
        """传入当前Epoch需要训练的原始图片索引"""
        self.active_indices = indices

    def __len__(self):
        # 返回当前激活的子集长度
        return len(self.active_indices) 
    
    def load_image_and_label(self, index):
        # ========= 通过 active_indices 映射到真实的图片索引 =========
        real_idx = self.active_indices[index]
        img_path = self.original_img_lst[real_idx]
        txt_path = self.original_txt_lst[real_idx]
        image = Image.open(img_path).convert('RGB')
        labels, _, _ = process_txtdata(txt_path)
        return image, labels
    
    def __getitem__(self, index):
        # 根据当前阶段决定是否使用mosaic
        if self.current_stage == "mosaic":
            if random.random() < self.mosaic_prob: 
                # 随机选3个其他索引(4张不同的图, 不允许重复)
                indices = [index] + random.sample([i for i in range(len(self)) if i != index], 3)
                images_mosaic = []
                labels_mosaic = []

                for idx in indices:
                    img, labs = self.load_image_and_label(idx)
                    # 使用当前阶段的变换
                    img_resized, labs_resized = self.mosaic_transforms(img, labs)
                    images_mosaic.append(img_resized)
                    labels_mosaic.append(labs_resized)
                # 准备马赛克增强
                mosaic_img, mos_label, fail = T.mosaic(
                    images_mosaic,  # list of 4 x [C, Hi, Wi]
                    labels_mosaic  # list of 4 x [N_i, 5]
                )
                mosaic_h, mosiac_w = mosaic_img.shape[-2:]
                mosaic_img = mosaic_img.unsqueeze(0)  # 添加 batch 维度: (C, H, W) -> (1, C, H, W)
                if fail:
                    # 如果mosaic失败, 则直接返回原始图像
                    mosaic_img_half = mosaic_img.squeeze(0)  # 插值后移除 batch 维度
                else:
                    # 如果mosaic成功, 则进行插值减半, 使得图片的尺寸能够处于max_size内
                    mosaic_img_half = F.interpolate(mosaic_img, 
                                                    size=(int(mosaic_h//2), int(mosiac_w//2))).squeeze(0)
                class_ids = mos_label[:, 0].long()
                normed_bboxes = mos_label[:, 1:].float()
                orign_h, orign_w = mosaic_img_half.shape[-2:]
                return mosaic_img_half, class_ids, normed_bboxes, orign_w, orign_h
            else:
                image, labels = self.load_image_and_label(index)
                image, labels = self.normal_transforms(image, labels)
                class_ids = torch.as_tensor(labels[:, 0], dtype=torch.long)
                normed_bboxes = torch.as_tensor(labels[:, 1:], dtype=torch.float32)
                orign_h, orign_w = image.shape[-2:]
                return image, class_ids, normed_bboxes, orign_w, orign_h
        elif self.current_stage == "normal_aug":
            image, labels = self.load_image_and_label(index)
            image, labels = self.normal_transforms(image, labels)
            class_ids = torch.as_tensor(labels[:, 0], dtype=torch.long)
            normed_bboxes = torch.as_tensor(labels[:, 1:], dtype=torch.float32)
            orign_h, orign_w = image.shape[-2:]
            return image, class_ids, normed_bboxes, orign_w, orign_h
        else:
            image, labels = self.load_image_and_label(index)
            image, labels = self.simple_transforms(image, labels)
            class_ids = torch.as_tensor(labels[:, 0], dtype=torch.long)
            normed_bboxes = torch.as_tensor(labels[:, 1:], dtype=torch.float32)
            orign_h, orign_w = image.shape[-2:]
            return image, class_ids, normed_bboxes, orign_w, orign_h

class ValDataset_for_DETR(Dataset):
    def __init__(self, 
                 imgdir_path, 
                 txtdir_path, 
                 image_set="val", 
                 max_size=1333, 
                 val_size=800):
        super().__init__()
        self.img_lst = []
        self.txt_lst = []
        # 构建图像和标签路径列表
        self.img_lst = [os.path.join(imgdir_path, f) for f in os.listdir(imgdir_path)]
        self.txt_lst = [os.path.join(txtdir_path, f) for f in os.listdir(txtdir_path)]
        
        # 确保图像和标签匹配, 显式的sort(), 使得加载数据顺序一致(注意验证集部分中有读取图片名, windows和linux加载顺序不一致!)
        self.img_lst.sort() # 排序十分重要
        self.txt_lst.sort() # 排序十分重要
        assert len(self.img_lst) == len(self.txt_lst), "Images don't match labels"
        print(f"Dataset initialization completed, a total of {len(self.img_lst)} samples")

        # 普通数据增强变换(水平翻转, 随机裁切, 随机大小)
        self.detr_transforms = make_detr_transforms(image_set=image_set, max_size=max_size, val_size=val_size)

    def __len__(self):
        return len(self.img_lst)   
    
    def load_image_and_label(self, index):
        img_path = self.img_lst[index]
        txt_path = self.txt_lst[index]
        image = Image.open(img_path).convert('RGB')
        labels, _, _ = process_txtdata(txt_path)
        return image, labels
    
    def __getitem__(self, index):
        image, labels = self.load_image_and_label(index)
        # image是经过图像数据增强后的尺度dim=(C=3, H, W), labels是经过图像变换之后的yolo归一化数据dim=(valid_N, 5)
        image, labels = self.detr_transforms(image, labels)
        class_ids = torch.as_tensor(labels[:, 0], dtype=torch.long) # (valid_N, )
        normed_bboxes = torch.as_tensor(labels[:, 1:], dtype=torch.float32) # (valid_N, 4) # 格式cx,cy, w, h的yolo格式
        orign_h, orign_w = image.shape[-2:] # 用于可视化检查, 不是训练需要的参数, 也不是预测需要的参数
        return image, class_ids, normed_bboxes, orign_w, orign_h

class PredictDataset_for_DETR(Dataset):
    def __init__(self, imgdir_path, image_set="val", max_size=1333, val_size=800):
        super().__init__()
        self.img_lst = []
        
        # 构建图像路径列表
        self.img_lst = [os.path.join(imgdir_path, f) for f in os.listdir(imgdir_path)]
        # 显式的顺序加载, 防止顺序不一致, 注意后期预测的img_name也要对应顺序筛选！！
        self.img_lst.sort()
        print(f"Dataset initialization completed, a total of {len(self.img_lst)} samples")
        self.transform = make_detr_transforms(image_set, max_size=max_size, val_size=val_size)

    def __len__(self):
        return len(self.img_lst)   
    
    def __getitem__(self, index):
        img_path = self.img_lst[index]
        image = Image.open(img_path).convert('RGB')
        # image是经过图像数据增强后的尺度dim=(C=3, H, W), 预测集无标签
        image, _ = self.transform(image, target=None)
        return image, None

