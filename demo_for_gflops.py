# 计算模型参数量
import torch
from models import GatedAttention_FineGrained_DINO_Swin
from fvcore.nn import FlopCountAnalysis, parameter_count_table

batch_size = 1
img_h, img_w = 640, 640  # 或 640x640，根据实际需求调整
num_classes = 80

# 创建模型实例
model = GatedAttention_FineGrained_DINO_Swin(num_classes=num_classes, num_queries=300)

# 创建 dummy input
x = torch.randn(batch_size, 3, img_h, img_w)
mask = torch.zeros(batch_size, img_h, img_w, dtype=torch.bool)
model.eval()

# 计算 FLOPs 和参数量
with torch.no_grad():
    flops = FlopCountAnalysis(model, (x, mask))
    print("Total FLOPs: {:.2f} GFLOPs".format(flops.total() / 1e9))
    print("Parameter Count:")
    print(parameter_count_table(model))