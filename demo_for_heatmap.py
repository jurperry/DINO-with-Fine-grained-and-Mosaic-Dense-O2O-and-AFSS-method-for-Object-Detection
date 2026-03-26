from datasets import PredictDataset_for_DETR
from utils import collate_fn
from torch.utils.data import DataLoader
from models import GatedAttention_FineGrained_DINO_Swin

import torch
import matplotlib.pyplot as plt

# 训练好后做推理的热力图绘制
if __name__ == '__main__':
    # 数据集配置路径
    imgdir_path = "your/path/to/images"
    batch_size = 4

    mydataset = PredictDataset_for_DETR(imgdir_path, image_set="val", max_size=640, val_size=640)
    mydataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    model = GatedAttention_FineGrained_DINO_Swin(num_queries=300, num_classes=80, gate_attn=True).cuda()
    # load checkpoint
    checkpoint_path = "your/model/path/to/best_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model_weights = checkpoint['ema_state_dict']['module']
    model.load_state_dict(model_weights)
    model.eval()
    # inference with heatmap
    for batch in mydataloader:
        nested_tensor, _ = batch
        imgs, masks = nested_tensor.tensors, nested_tensor.mask
        class_outs, bbox_outs, heatmap = model(imgs.cuda(), mask=masks.cuda(), return_all_layers=False, draw_heatmap=True)
        print(f"Shape of first layer class predictions: {class_outs.shape}")
        print(f"Shape of first layer bbox predictions: {bbox_outs.shape}")
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        for i in range(imgs.shape[0]):
            img = imgs[i] * std + mean  # 反归一化
            img = img.permute(1, 2, 0).clamp(0, 1)  # 转换为 HWC 格式并限制到 [0, 1]
            # 可变型注意力的热力图的可视化
            plt.figure()
            plt.imshow(img)
            plt.imshow(heatmap[i], cmap='viridis', alpha=0.5)  # 设置透明度
            plt.colorbar(label='Attention Weight')
            plt.title("Heatmap")
            plt.show()
        break
