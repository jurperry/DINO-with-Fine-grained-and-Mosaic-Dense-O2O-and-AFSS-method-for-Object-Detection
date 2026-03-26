import timm
import re
import torch.nn as nn
from safetensors.torch import load_file

# 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k'
# 模型必须是4的倍数, 可以用于任意4倍数尺度的训练了
class SwinTransformerBackbone(nn.Module):
    def __init__(self, 
                 model_name='swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', 
                 checkpoint_path=None,
                 out_indices=[0, 1, 2, 3],
                 freeze=False):
        super().__init__()
        
        # Step 1: 创建模型, 启用动态输入
        self.backbone = timm.create_model(
            model_name,
            pretrained=(checkpoint_path is None),
            img_size=None,              # 不固定尺寸
            dynamic_img_size=True,      # 启用动态尺寸
            features_only=True,         # 直接输出多尺度特征
            out_indices=out_indices
        )
        
        # Step 2: 如果有 checkpoint，加载权重
        if checkpoint_path is not None:
            ckpt_dict = load_file(checkpoint_path)
            
            # 下划线转点, timm创建的是layers_X, 而checkpoint是layers.X
            converted_dict = {}
            for key, value in ckpt_dict.items():
                # print(f"Original key: {key}")
                # 转换 layers_0.blocks.0 为 layers.0.blocks.0
                new_key = key
                if 'layers.' in new_key:
                    # 使用正则表达式替换所有 layers.X 为 layers_X
                    new_key = re.sub(r'layers\.(\d+)', r'layers_\1', new_key)
                    # print(f"Converted key: {new_key}")
                converted_dict[new_key] = value
            
            # 过滤掉不需要的键（如分类头）
            filtered_dict = {}
            for key, value in converted_dict.items():
                if not key.startswith('head.'):  # 移除分类头
                    filtered_dict[key] = value
            
            missing, unexpected = self.backbone.load_state_dict(filtered_dict, strict=False)

            if missing:
                print("Missing keys:", missing)

        # 修改patch_embed支持任意尺寸
        self.backbone.patch_embed.img_size = None
        self.backbone.dynamic_img_size = True

        for m in self.backbone.modules():
            if hasattr(m, 'attn_mask') and m.attn_mask is not None:
                m.register_buffer('attn_mask', None, persistent=False)
        
        # 冻结权重
        self.freeze_weights(freeze)
    
    def freeze_weights(self, freeze=True):
        """冻结/解冻模型权重"""
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
    
    # 👇 必须添加 forward 方法！
    def forward(self, x):
        return self.backbone(x)

# checkpoint_path = "models/swin_tiny.safetensors"
# model = SwinTransformerBackbone(checkpoint_path=checkpoint_path)
        