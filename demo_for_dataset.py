# 数据集可视化
from datasets import TrainDataset_for_DETR
from torch.utils.data import DataLoader
from utils import collate_fn
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    imgdir_path = "your/path/to/images"
    txtdir_path = "your/path/to/yolo_like/txt"

    my_dataset = TrainDataset_for_DETR(
        imgdir_path=imgdir_path,  # 替换为实际路径
        txtdir_path=txtdir_path,  # 替换为实际路径
        total_epochs=50,
        mosaic_ratio=0.625,
        warmup_epochs=5,
        final_finetune_epochs=5,
        mosaic_prob=1.0,
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        max_size = 1333,
        val_size=800,
    )
    
    batch_size = 4
    mydataloader = DataLoader(my_dataset, 
                                   batch_size=batch_size, 
                                   shuffle=True, 
                                   collate_fn=collate_fn, 
                                   num_workers=0, 
                                   pin_memory=True)
    # (bs, m, n)---x    (bs,[tensor1, tensor2...])

    my_dataset.set_epoch(15)
    for batch in mydataloader:
        nested_tensor, real_id, real_norm_bboxes, _, _ = batch

        # 用于输入模型的参数部分
        imgs, masks, real_ids, real_bboxes = nested_tensor.tensors, nested_tensor.mask, real_id, real_norm_bboxes
        # 构建targets 用于去噪训练
        targets = {}
        targets['labels'] = real_ids
        targets['bbox'] = real_bboxes
        
        print(f"图像尺寸: {nested_tensor.tensors.shape}")          # torch.Size([bs, 3, H, W])
        print(f"掩码尺寸: {nested_tensor.mask.shape}")          # torch.Size([bs, 3, H, W])
        print(f"标注数据: {real_id}") # list存储每一张图的类别信息
        print(f"标注框: {real_norm_bboxes}") # list存储cxcywh归一化标注框张量, 处于transform变换后的图像坐标系, 而不是nested之后的图像坐标系
        print(real_norm_bboxes[0][0][0])
        break
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # 如果想要展示一个batch中所有图像，可以遍历imgs并逐一显示
    for i in range(batch_size):
        img = imgs[i] * std + mean  # 反归一化
        img = img.permute(1, 2, 0).clamp(0, 1)  # 转换为 HWC 格式并限制到 [0, 1]
        plt.figure()
        plt.imshow(img)
        plt.title(f'Image {i+1} in Batch')
        plt.axis('off')
    plt.show()
    
