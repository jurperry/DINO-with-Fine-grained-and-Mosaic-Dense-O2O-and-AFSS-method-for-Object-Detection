import copy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .swin_backbone import SwinTransformerBackbone
from .gated_attention import MultiheadGatedAttention
from msdga_module import MSDeformAttention, MSDeformGatedAttention
from .denoising import prepare_for_cdn, get_cdn_indices
from .fine_grained import weighting_function, distance2bbox
from tools import draw_heatmap
from utils import inverse_sigmoid, gen_sineembed_for_position, bias_init_with_prob

# fine_dino_gated_swin
# tricks:
# (1) FDR fine grained distribution refinement 
# (2) GO-LSD global optimal-location self distillation 
# (3) use swin_transformer_tiny backbone for  better performance
# (4) use mosaic augmentation for the fast convergence performance
# (5) MSDeformGatedAttention + pre_norm, use cuda to compute attention
#     可以使得学习率lr变大进行训练, 达到1e-3也能训练, 范围很广

class MultiScalePatchEmbed(nn.Module):
    """提取多尺度特征并生成对应的掩码和空间信息"""
    def __init__(self, embed_dim=256, freeze=False):
        super().__init__()
        # swin_tiny骨干网络, 微调权重, 性能最佳
        self.backbone = SwinTransformerBackbone(checkpoint_path="models/backbone/swin_tiny.safetensors", 
                                                freeze=freeze)
        # 特征投影层, 使用swin 192, 384, 768的conv_dim
        self.proj_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(192, embed_dim, kernel_size=1),
                          nn.GroupNorm(32, embed_dim)),           # layer2 output(s2)
            nn.Sequential(nn.Conv2d(384, embed_dim, kernel_size=1),
                          nn.GroupNorm(32, embed_dim)),           # layer3 output(s3)
            nn.Sequential(nn.Conv2d(768, embed_dim, kernel_size=1),
                          nn.GroupNorm(32, embed_dim)),           # layer4 output(s4)
            # 1个3x3conv +GroupNorm
            nn.Sequential(nn.Conv2d(768, embed_dim, kernel_size=3, stride=2, padding=1),  # 3x3conv s=2 -> 256channel
                          nn.GroupNorm(32, embed_dim)),           # layer5 output(s5)
        ])
        # 初始化
        self.get_parameter()

    # 根据掩码生成有效比例约束采样点
    @staticmethod
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1) # ~是取反, ~mask[:, :, 0]=(bs, Hi), 后面的1指代的是维度dim=1, 结果是dim=(bs,)
        valid_W = torch.sum(~mask[:, 0, :], 1) # ~是取反, ~mask[:, 0, :]=(bs, Wi), 后面的1指代的是维度dim=1, 结果是dim=(bs,)
        valid_ratio_h = valid_H.float() / H # 计算有效值, 是一个dim=(bs, )
        valid_ratio_w = valid_W.float() / W # 计算有效值, 是一个dim=(bs, )
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1) # [[w_ratio, h_ratio],...], (bs, 2)的维度
        return valid_ratio # (bs, 2), 这样wh-xy就是对齐的, 直接乘积就可以约束采样点了

    # 初始化函数
    def get_parameter(self):
        for proj in self.proj_layers:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        swin_features = self.backbone(x)
        # x1 = swin_features[0].permute(0, 3, 1, 2)   # (bs, H/4, W/4, 96) -> (bs, 96, H/4, W/4)
        x2 = swin_features[1].permute(0, 3, 1, 2)  # (bs, H/8, W/8, 192) -> (bs, 192, H/8, W/8)
        x3 = swin_features[2].permute(0, 3, 1, 2)  # (bs, H/16, W/16, 384) -> (bs, 384, H/16, W/16)
        x4 = swin_features[3].permute(0, 3, 1, 2)  # (bs, H/32, W/32, 768) -> (bs, 768, H/32, W/32)
        
        features = []
        spatial_shapes = []
        masks = []
        
        # 使用layer2, layer3, layer4的输出作为多尺度特征
        multi_scale_features = [x2, x3, x4]
        for i, feat in enumerate(multi_scale_features):
            proj_feat = self.proj_layers[i](feat)  # (bs, embed_dim, h, w)
            h, w = proj_feat.shape[-2:]
            # 展平特征
            flattened = proj_feat.flatten(2).transpose(1, 2)  # (bs, h*w, embed_dim)
            # 处理掩码, 如果掩码是[bs, H, W]三维, 则插入维度变成[bs, 1, h, w]变成4维, 方便后续插值
            if mask is not None:
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                mask_down = F.interpolate(mask.float(), size=(h, w), mode='nearest') # 最近临插值到目标尺寸(h, w)--(bs, 1, h, w)
                mask_down = mask_down.squeeze(1).bool() # (bs, h, w)
                masks.append(mask_down)
            features.append(flattened)
            spatial_shapes.append([h, w])
        
        s5_feat = self.proj_layers[-1](multi_scale_features[-1]) # (bs, 256, H/64, W/64)
        s5_h, s5_w = s5_feat.shape[-2:]
        s5_flatten = s5_feat.flatten(2).transpose(1, 2) # (bs, H/64*W/64, embed_dim)
        # 处理掩码, 如果掩码是[bs, H, W]三维, 则插入维度变成[bs, 1, h, w]变成4维, 方便后续插值
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            s5_mask = F.interpolate(mask.float(), size=(s5_h, s5_w), mode='nearest') # 最近临插值到目标尺寸(h, w)--(bs, 1, h, w)
            s5_mask = s5_mask.squeeze(1).bool() # (bs, H/64, W/64)
            masks.append(s5_mask)
        features.append(s5_flatten)
        spatial_shapes.append([s5_h, s5_w])

        # 合并所有尺度的特征, dim=(bs, total_tokens, embed_dim)
        all_features = torch.cat(features, dim=1) 
        # 保存各个尺度的形状 dim=[n_level, 2], 每一行是(h, w)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=x.device) 
        
        # 计算level_start_index
        level_start_index = torch.cat([spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]])

        # 各尺度特征图中非padding部分的边长占其边长的比例  [bs, n_level, 2]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        
        # 合并所有尺度的掩码
        if mask is not None:
            all_masks = torch.cat([m.flatten(1) for m in masks], dim=1) # (bs, total_tokens)
        else:
            all_masks = None

        return all_features, spatial_shapes, level_start_index, valid_ratios, all_masks

# Encoder部分的二维位置编码PositionEncode
class PositionEncode(nn.Module):
    '''
    x: [bs, total_tokens, embed_dim], 仅仅用作设备放置以及batch_size获取
    h: 对应尺度下的高, 外来输入
    w: 对应尺度下的宽, 外来输入
    '''
    def __init__(self, embed_dim=256):
        super().__init__()
        # nn.Embedding  相当于 nn.Parameter  其实就是初始化函数
        self.row_embed = nn.Embedding(300, embed_dim//2) # (300, 128)
        self.col_embed = nn.Embedding(300, embed_dim//2) # (300, 128)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, h, w):
        i = torch.arange(w, device=x.device) # [0, 1, 2, 3, ..., w-1]
        j = torch.arange(h, device=x.device) # [0, 1, 2, 3, ..., h-1]
        x_emb = self.col_embed(i)   # 初始化x方向位置编码, 把self.row_embed中索引为[0, 1, 2, 3, ..., w-1]的初始化元素取出--->(w, 128)
        y_emb = self.row_embed(j)   # 初始化y方向位置编码, 把self.col_embed中索引为[0, 1, 2, 3, ..., h-1]的初始化元素取出--->(h, 128)
        # concat x y 方向位置编码
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        # x_emb.unsqueeze(0).repeat(h, 1, 1) = (h, w, 128)
        # y_emb.unsqueeze(1).repeat(1, w, 1) = (h, w, 128), dim = -1按最后一个维度cat拼接
        # pos=(bs, h, w, 256)
        return pos.flatten(1, 2)  # (bs, num_tokens=h*w, embed_dim=256)

# 为多尺度特征生成位置编码:包含二维位置编码+尺度位置编码
# 共享二维xy位置positional encoding, DeformableDETR做法
class MultiPositionEncode(nn.Module):
    """
    x: [bs, total_tokens, embed_dim]
    spatial_shapes: [n_levels, 2]
    """
    def __init__(self, n_levels=4, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        # 不同尺度有不同的xy二维位置编码和层级位置编码
        self.position_embedding = PositionEncode(embed_dim)
        self.level_embedding = nn.Parameter(torch.Tensor(n_levels, embed_dim)) # dim=[4, 256], 用于不同层级的位置编码
        nn.init.normal_(self.level_embedding)
        
    def forward(self, x, spatial_shapes):
        """为多尺度特征生成位置编码"""
        pos_embeddings = []
        for level, (h, w) in enumerate(spatial_shapes):
            # 共享同一个xy位置编码初始化
            pos_embed = self.position_embedding(x, h, w)
            # 不同尺度层级位置编码
            # self.level_embedding[level]-->(embed_dim,).view(1, 1,-1)-->(1, 1, 256), 维度广播
            lvl_pos_embed = pos_embed + self.level_embedding[level].view(1, 1, -1)  # (bs, hi*wi, embed_dim)
            pos_embeddings.append(lvl_pos_embed)
        return torch.cat(pos_embeddings, dim=1) # (batch_size, total_tokens, embed_dim)

# Encoder最重要的部分, 不同层级下的采样点生成与采样约束
class DeformableEncoderBlock(nn.Module):
    '''
    src: [bs, total_tokens, embed_dim], 多尺度特征图
    pos: [bs, total_tokens, embed_dim], 多尺度位置编码(二维位置+尺度位置)
    reference_points: [bs, total_tokens, n_levels, 2]
    spatial_shapes: [n_levels, 2]
    level_start_index: [0, h0*w0, h1*w1...]
    mask: [bs, total_tokens]
    '''
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, gate_attn=True):
        super().__init__()
        self.gate_attn = gate_attn
        if self.gate_attn:
            self.self_attn = MSDeformGatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                batch_first=True,
            )
        else:
            self.self_attn = MSDeformAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                batch_first=True,
            )
        self.ffn = FFN_Layer(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, mask=None):
        # 多尺度可变形-自注意力, 只有q和v, query=src+pos, value=src
        residual = src # 残差块
        # pre_norm
        src = self.norm1(src)
        q_self = src + pos
        v_self = src
        if self.gate_attn:
            attn_out = self.self_attn(
                query=q_self,
                value=v_self,
                reference_points=reference_points,
                Xq=src,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=mask
            )[0]
        else:
            attn_out = self.self_attn(
                query=q_self,
                value=v_self,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=mask
            )[0]
        src = residual + self.dropout1(attn_out)
        residual = src
       
        # ffn
        # pre_norm
        src = self.norm2(src)
        src = residual + self.dropout2(self.ffn(src))

        return src # (batch_size, total_tokens, embed_dim)

class DeformableEncoder(nn.Module):
    '''
    src: [bs, total_tokens, embed_dim]
    pos: [bs, total_tokens, embed_dim]
    spatial_shapes: [n_level, 2]
    level_start_index: [0, h0*w0, h1*w1]
    valid_ratios: [bs, n_level, 2]
    mask: [bs, total_tokens]
    '''
    def __init__(self, num_layers=6, embed_dim=256, num_heads=8, num_levels=4, num_points=4, gate_attn=True):
        super().__init__()
        self.gate_attn = gate_attn
        self.layers = nn.ModuleList([
            DeformableEncoderBlock(embed_dim, num_heads, num_levels, num_points, gate_attn=self.gate_attn) 
            for _ in range(num_layers)
        ])

    @staticmethod # 这一个部分还没有完全学会, 需要进一步学习
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        生成参考点reference points
        spatial_shapes: n_level个特征图的shape [n_level, 2]
        valid_ratios: n_level个特征图中非padding部分的边长占其边长的比例, 即有效比例[bs, n_level, 2]
        device: cuda:0
        """
        reference_points_list = []
        # 遍历4个特征图的shape  比如 H_=100  W_=150
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 显式添加indexing='ij', 消除兼容性问题
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device), indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # list3: [bs, H/8*W/8, 2] + [bs, H/16*W/16, 2] + [bs, H/32*W/32, 2]
        reference_points = torch.cat(reference_points_list, 1) # 在像素维度上进行拼接
        reference_points_input = reference_points[:, :, None] * valid_ratios[:, None] # 乘以有效比例, 防止采样到填充区
        return reference_points_input # dim=[bs, total_tokens, n_level, 2]
    
    # 在总的Encoder里面需要传递valid_ratio, 修复了原来只有一层EncoderBlock的bug
    def forward(self, src, pos, spatial_shapes, level_start_index, valid_ratios, mask=None):
        reference_points_input = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        memory = src
        for layer in self.layers:
            memory = layer(memory, pos, reference_points_input, spatial_shapes, level_start_index, mask)
        return memory # (bs, total_tokens, embed_dim)

class DeformableDecoderBlock(nn.Module):
    '''
    queries: [bs, Len_q, embed_dim]
    memory: [bs, total_tokens, embed_dim]
    query_pos: [bs, Len_q, embed_dim]
    reference_points: [bs, Len_q, n_level, 4], 已经经过sigmoid归一化与src_valid_ratio约束
    spatial_shapes: [n_level, 2]
    level_start_index: [0, h0*w0, h1*w1,...]
    mask: [bs, total_tokens, embed_dim]
    '''
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, gate_attn=True):
        super().__init__()
        self.gate_attn = gate_attn
        if self.gate_attn:
            self.self_attn = MultiheadGatedAttention(embed_dim, num_heads, dropout=0.1)
            self.cross_attn =  MSDeformGatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                batch_first=True,
            )
        else:
            self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
            self.cross_attn =  MSDeformAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                batch_first=True,
            )
        
        self.ffn = FFN_Layer(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.gateway = Gate(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        
    def forward(self, queries, memory, query_pos, reference_points, spatial_shapes, level_start_index, 
                mask=None, attn_mask=None):
        # 完整的多头自注意力机制, q, k, v均存在, q=k为随机初始化的300个query+query_pos, v就是queries
        residual = queries # 残差块
        # pre_norm1
        queries = self.norm1(queries)
        q_self = k_self = queries + query_pos
        v_self = queries

        # self attn
        if self.gate_attn:
            if attn_mask is not None:
                # attn_mask: [total_queries, total_queries]
                self_attn_out = self.self_attn(
                    q_self, k_self, v_self, queries,
                    attn_mask=attn_mask
                )[0]
            else:
                self_attn_out = self.self_attn(q_self, k_self, v_self, queries)[0]
        else:
            if attn_mask is not None:
                # attn_mask: [total_queries, total_queries]
                self_attn_out = self.self_attn(
                    q_self, k_self, v_self,
                    attn_mask=attn_mask
                )[0]
            else:
                self_attn_out = self.self_attn(q_self, k_self, v_self)[0]
        queries = residual + self.dropout1(self_attn_out)
        residual = queries # 残差块

        # pre_norm2
        queries = self.norm2(queries)
        q_cross = queries + query_pos
        v_cross = memory
        # cross attn
        if self.gate_attn:
            cross_attn_out, attn_weights, sampling_locations = self.cross_attn(
                query=q_cross,
                value=v_cross, 
                reference_points=reference_points,
                Xq=queries,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=mask
            )
        else:
            cross_attn_out, attn_weights, sampling_locations = self.cross_attn(
                query=q_cross,
                value=v_cross,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=mask
            )
        # 门控融合
        queries = self.gateway(residual, self.dropout2(cross_attn_out))
        residual = queries # 残差块
        
        # ffn
        # pre_norm3
        queries = self.norm3(queries)
        queries = residual + self.dropout3(self.ffn(queries))

        # 热力图可视化, queries=(bs, Len_q, embed_dim)
        return queries, attn_weights, sampling_locations

class DeformableDecoder(nn.Module):
    '''
    queries: [bs, Len_q, embed_dim]
    memory: [bs, total_tokens, embed_dim]
    reference_points: [bs, Len_q, xywh=4]
    spatial_shapes: [n_level, 2]
    level_start_index: [0, h0*w0, h1*w1...]
    src_valid_ratio: [bs, n_level, 2]
    mask: [bs, total_tokens]
    '''
    def __init__(self, num_layers=6, embed_dim=256, num_heads=8, num_levels=4, num_points=4, 
                 query_pos_dim=4, reg_max=32, gate_attn=True):
        super().__init__()
        self.num_layers = num_layers
        self.reg_max = reg_max
        self.gate_attn = gate_attn
        # 主程序赋值
        # class_embed就是7个传统头, bbox_embed是6个FDR头+1个传统头
        # class_embed: 6x FDR(layer0是传统头) + 1 enc_class
        # bbox_embed: 6 x FDR + 1 x enc_bbox 
        self.class_embed = None 
        self.bbox_embed = None 
        # 外界传入的一个传统边界头, traditional_class = class_embed[0]
        self.traditional_bbox_embed = None

        # 超参数从外部传入
        self.up = None
        self.reg_scale = None

        # query_pos_dim=4->xywh, 对高频位置进行映射, 使用pos_mlp进行转化处理
        self.query_pos_head = PositionalQuery_MLP(query_pos_dim // 2 * embed_dim, embed_dim, embed_dim, num_layers=2)
        
        # decoder叠层 6 layers
        self.layers = nn.ModuleList([
            DeformableDecoderBlock(embed_dim, num_heads, num_levels, num_points, gate_attn)
            for _ in range(num_layers)
        ])
        # lqe定位质量评估器叠层 6 layers
        self.lqe_layers = nn.ModuleList(
                    [copy.deepcopy(LQE(4, 64, 2, self.reg_max)) for _ in range(num_layers)]
                )
        # 偏移预测融合, 从分布形式变为ltrb偏移
        self.integral = None
    
    # 热力图可视化
    def forward(self, queries, memory, reference_points, spatial_shapes, level_start_index, src_valid_ratio, 
                mask=None, attn_mask=None, image_shape=None):
        # 输入的queries其实就是内容查询content_queries, 其为去噪与非去噪分支的融合
        if self.training:
            project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=False)
        else:
            project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        
        # 第一个decoder的queries_detach与pred_coners_undetach赋值均为0
        queries_detach = pred_corners_undetach = 0

        dec_out_logits=[]
        dec_out_pred_corners = []
        dec_out_bboxes = []
        dec_out_refs = []
        traditional_bboxes = None
        traditional_scores = None

        # 一共迭代6层次
        for layer_id, layer in enumerate(self.layers):
            # 进行decoder_layer的ref_points_input计算, 其中传入的reference_points已经是sigmoid后的(bs, q, cxcywh=4)
            # (bs, q, 1, 4) * (bs, 1, n_level, 4) -> (bs, q, n_level, 4)
            reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratio, src_valid_ratio], -1)[:, None]
            
            # 迭代高频位置编码, reference_points_input[:, :, 0, :]=(bs, q, 4) -> (bs, q, 512)
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # (bs, nq, 512)
            query_pos = self.query_pos_head(query_sine_embed).clamp(min=-10, max=10)  # (bs, nq, 256)
            
            # 计算一个decoder的结果, 一共6个decoder
            # 热力图可视化需要attn_weights, sampling_locations
            queries, attn_weights, sampling_locations = layer(queries, 
                                                              memory, 
                                                              query_pos, 
                                                              reference_points_input, 
                                                              spatial_shapes, 
                                                              level_start_index, 
                                                              mask, 
                                                              attn_mask)
            if layer_id == 0:
                # 传统边界头与传统分类头
                # Initial bounding box predictions with inverse sigmoid refinement
                traditional_bboxes = F.sigmoid(self.traditional_bbox_embed(queries) + inverse_sigmoid(reference_points))
                traditional_scores = self.class_embed[0](queries) # 传统分类头和lqe在laeyr0公用一个class_embed
                # 第一层decoder得到的FDR初始参考点, 这个参考点是在后续decoder层都不变的值, 且第一个decoder也用这个做了FDR
                ref_points_initial = traditional_bboxes.detach() # 归一化且梯度截断的初始参考点, (bs, q, cxcywh=4)

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            # pred_corners就是本层decoder最终的累计偏移量l,t,r,b坐标, 是相对于ref_points_initial的偏移量(offset)
            # 第0层queries_detach与pred_corners_undetach都是0, 目的是为了实现0偏移
            pred_corners = self.bbox_embed[layer_id](queries + queries_detach) + pred_corners_undetach # (bs, q, 132)
            # 细粒度分布优化FDR计算迭代参考点(中间过程的参考点), 实际上就是在做: ref + FDR_offset
            # 但是却不同于RT-DETR的做法, 因为其参考点是固定值ref_points_initial
            # inter_ref_bbox用于下一个decoder的注意力计算, 同时这个迭代参考点也是输出的最终预测框
            inter_ref_bbox = distance2bbox(
                ref_points_initial, self.integral(pred_corners, project), self.reg_scale
            ) # (bs, q, cxcywh=4)

            # 分类置信度scores计算, 分类-定位置信度lqe-scores计算
            scores = self.class_embed[layer_id](queries)
            scores = self.lqe_layers[layer_id](scores, pred_corners) # 分类置信度与定位分布结合的定位分类置信度
            dec_out_logits.append(scores)
            # 区别于rt-detr, 细粒度分布优化FDR, 并不会返回每一个decoder的初始参考点, 而是在decoder内就解决偏移相加了
            dec_out_bboxes.append(inter_ref_bbox)
            # 偏移值返回, 用于监督损失FGLoss, 偏移值(bs, dnq+nq, 132)
            dec_out_pred_corners.append(pred_corners)
            # 注意这里需要放入的是每一层decoder的ref_points_initial, 复用同一个值
            dec_out_refs.append(ref_points_initial)

            pred_corners_undetach = pred_corners
            # 参考点更新, 每一层的decoder注意力计算的需要, 每一层MSD需要的参考点, 不同decoder层级不能混淆梯度
            reference_points = inter_ref_bbox.detach()
            # 对queries进行detach, 作为下一层融合queries(本层) + queries_detach(上层)
            queries_detach = queries.detach()

            # 最后一层decoder, 绘制热力图
            if layer_id == len(self.layers) - 1 and image_shape is not None:
                heatmap = draw_heatmap(attn_weights, sampling_locations, image_shape)
            else:
                heatmap = None

        # 需要绘制热力图
        return dec_out_logits, dec_out_bboxes, dec_out_pred_corners, dec_out_refs, \
                traditional_scores, traditional_bboxes, heatmap

# 残差门控
class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return gate1 * x1 + gate2 * x2 # (bs, nq, embed_dim)

# 用于计算累计偏移量(bs, q, 132)的加权和, 得到l,t,r,b偏移量(bs, q, 4), 最终用于和ref_points_initial加和得到最终预测值
# 输出的是l, t, r, b偏移量, 可正可负
class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    # x输入为pred_corners当前层总偏移量, 第一层decoder的偏移量之前的偏移量pred_corners_undetach为0, 加和得到第一层偏移量
    # x.shape = (bs, denoising_query + num_query, 4x33=132) l,t,r,b四个偏移
    # project: weighting_function生成的权重值, (33, ), 33个权处于-4~4之间
    def forward(self, x, project):
        shape = x.shape # (bs, all_queries, 4x33)
        # softmax归一化到0~1
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1) # x.reshape(bs*all_queries*4, 33)-> softmax(dim=1)即为33所在维度
        # weight_function=project加权 权重值处于-4~4之间, 为W(n)/c = W(n) / 0.25 = 4.0 * W(n)m, 之后distance_to_bbox会把4除掉
        x = F.linear(x, project.to(x.device)).reshape(-1, 4) # x与project做线性乘积加权求和, 从(bs*q*4, )变成(bs*q, 4)
        return x.reshape(list(shape[:-1]) + [-1]) # (bs*q, 4).reshape([bs, q, -1]) -> (bs, q, 4) 值域-4~+4

# Location Quality Estimator
# 定位-分类感知, 结合分类置信度与定位偏移预测分布的一个新型分类置信度
class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score

# decoder与encoder的前向传播ffn层, 可以把hidden_dim=2048从而使得训练更强
class FFN_Layer(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        return self.net(x)
  
### MLP 与 PositionalQuery_MLP是同样的, 等价的写法, 且PositionalQuery_MLP更加鲁棒
# bbox_embed的MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim,
                                    hidden_dim if i < num_layers - 1 else output_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers) # 最后一层仍然是Linear层, 只有中间层是ReLU

    def forward(self, x):
        return self.layers(x)

# 迭代位置查询positional query的MLP层
class PositionalQuery_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# 主模型: GatedAttention_FineGrained_DINO_Swin
class GatedAttention_FineGrained_DINO_Swin(nn.Module):
    def __init__(self, num_queries=300, num_classes=80, embed_dim=256, num_encoder_layer=6, num_decoder_layer=6, gate_attn=True, 
                 num_cdn_groups=100, dn_labelbook_size=100, dn_label_noise_ratio=0.5, dn_box_noise_scale=1.0, learn_init_tgt=True,
                 reg_max=32, up=0.5, reg_scale=4.0, freeze_backbone=False):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.gate_attn = gate_attn # whether or not using gated attn
        self.two_stage_topk_num = num_queries
        # denoising params
        self.num_cdn_groups = num_cdn_groups
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_box_noise_scale = dn_box_noise_scale
        # weighting_function params
        self.reg_max = reg_max
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=False) # frozen grads
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False) # frozen grads

        # 主干网络嵌入层
        self.patch_embed = MultiScalePatchEmbed(embed_dim, freeze=freeze_backbone)
        
        self.encoder_pos = MultiPositionEncode(embed_dim)
        self.encoder = DeformableEncoder(num_encoder_layer, gate_attn=self.gate_attn)
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, embed_dim)

        # init the content query, (nq=topk=300, embed_dim=256)
        self.tgt_embed = nn.Embedding(num_queries, embed_dim)
        nn.init.normal_(self.tgt_embed.weight.data)

        # 对encoder输出memory进行处理：全连接层 + 层归一化
        self.encoder_linear_layer = nn.Linear(embed_dim, embed_dim)
        self.encoder_layer_norm = nn.LayerNorm(embed_dim)
        self.learn_init_tgt = learn_init_tgt
        self.decoder = DeformableDecoder(num_decoder_layer, reg_max=self.reg_max, gate_attn=self.gate_attn)
        
        # 7xclass_head层 6x class_embed + 1 enc_class
        self.class_embed = nn.ModuleList([
            nn.Linear(256, num_classes) for _ in range(num_decoder_layer + 1)
        ])
        # 7个bbox_embed层 6x FDR_bbox_embed + 1x enc_bbox_embed
        self.bbox_embed = nn.ModuleList(
            [ MLP(256, 256, 4 * (self.reg_max + 1), 3) for _ in range(num_decoder_layer)] + [MLP(256, 256, 4, 3)]
            )

        self.traditional_bbox_embed = MLP(256, 256, 4, 3)
        self.integral = Integral(self.reg_max)

        # 主程序给decoder的各个属性进行赋值
        self.decoder.bbox_embed = self.bbox_embed
        self.decoder.class_embed = self.class_embed
        self.decoder.traditional_bbox_embed = self.traditional_bbox_embed
        self.decoder.reg_scale = self.reg_scale
        self.decoder.up = self.up
        self.decoder.integral = self.integral
  
        # 初始化分类头和边界框头
        self._init_weights()
    # encoder的提议内容与提议锚点的生成
    # proposal_memory, proposal_anchors(参考点reference_points)生成
    # proposal_memory用于经过class_embed生成(bs, 300, num_classes), 经过bbox_embed生成(bs, 300, xywh=4)偏移量
    @staticmethod
    def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes, 
                                    encoder_linear_layer, encoder_layer_norm):
        """得到第一阶段预测的所有proposal_anchors和处理后的Encoder输出propsal_memory
        memory: Encoder输出特征  [bs, H/8 * W/8 + ... + H/64 * W/64, 256]
        memory_padding_mask: Encoder输出特征对应的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        spatial_shapes: [4, 2] backbone输出的4个特征图的shape
        """
        bs, seq_len, embed_dim = memory.shape  # bs  H/8 * W/8 + ... + H/64 * W/64  256
        anchor_boxes = []
        _cur = 0   # 帮助找到mask中每个特征图的初始index
        for lvl, (H_, W_) in enumerate(spatial_shapes):  # 如H_=76  W_=112
            # 1、生成所有proposal box的中心点坐标xy
            # 展平后的mask [bs, 76, 112, 1]
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(bs, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            # grid_y = [76, 112]   76行112列  第一行全是0  第二行全是1 ... 第76行全是75
            # grid_x = [76, 112]   76行112列  76行全是 0 1 2 ... 111, torch.linspace坑: 两端都取得到!!!
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device), indexing='ij')
            # grid = [76, 112, 2(xy)]   这个特征图上的所有坐标点x,y
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(bs, 1, 1, 2)  # [bs, 1, 1, 2(xy)]
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale

            # 2、生成所有proposal box的宽高wh  第i层特征默认wh = 0.05 * (2**i), dim=(bs, Hi, Wi, wh=2)
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            # 3、concat xy+wh -> proposal xywh [bs, 76x112, 4(xywh)]
            anchor_box = torch.cat((grid, wh), -1).view(bs, -1, 4)
            anchor_boxes.append(anchor_box)
            # 确定下一尺度像素起点index, 本质是求level_start_index
            _cur += (H_ * W_)
        # concat 4 feature map proposals [bs, H/8 x W/8 + ... + H/64 x W/64] = [bs, 11312, 4]
        proposal_anchors = torch.cat(anchor_boxes, 1)
        proposal_anchors_valid_mask = ((proposal_anchors > 0.01) & (proposal_anchors < 0.99)).all(-1, keepdim=True)
        proposal_anchors = torch.log(proposal_anchors / (1 - proposal_anchors))
        proposal_anchors = proposal_anchors.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        proposal_anchors = proposal_anchors.masked_fill(~proposal_anchors_valid_mask, float('inf')) # 未sigmoid归一化

        proposal_memory = memory
        proposal_memory = proposal_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        proposal_memory = proposal_memory.masked_fill(~proposal_anchors_valid_mask, float(0))
        proposal_memory = encoder_layer_norm(encoder_linear_layer(proposal_memory)) # 层归一化了
        return proposal_memory, proposal_anchors

    def _init_weights(self):
        # 分类头初始化
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        for cls_layer in self.class_embed:
            nn.init.constant_(cls_layer.bias, bias_value)
        
        # 回归头初始化, 使用two_stage + with_box_refine, 由于预选框是topk, 已经是较好的起点, 设定无偏要求即可(已有先验)
        for i, bbox_layer in enumerate(self.bbox_embed):
            nn.init.zeros_(bbox_layer.layers[-1].weight)
            nn.init.zeros_(bbox_layer.layers[-1].bias)

    def forward(self, x, mask=None, targets=None, return_all_layers=True, draw_heatmap=False):
        # 多尺度特征提取, 获取所有需要的参数
        src, spatial_shapes, level_start_index, valid_ratios, src_mask = self.patch_embed(x, mask)
        
        # 编码器部分
        # 为Encoder部分生成位置编码(二维位置+尺度位置)
        encoder_pos = self.encoder_pos(src, spatial_shapes)
        memory = self.encoder(src, encoder_pos, spatial_shapes, level_start_index, valid_ratios, src_mask)
        
        # 解码器部分
        bs, _, _ = memory.shape
        
        # 对memory进行处理得到提议内容proposal_memory: [bs, H/8 * W/8 + ... + H/64 * W/64, 256]
        # 并生成初步提议框proposal_anchors: [bs, H/8 * W/8 + ... + H/64 * W/64, cxcywh=4], 其实就是特征图上的一个个的点坐标
        encoder_linear_layer = self.encoder_linear_layer
        encoder_layer_norm = self.encoder_layer_norm
        # 获取提议proposal_memory和proposal_anchors
        proposal_memory, proposal_anchors = self.gen_encoder_output_proposals(memory, src_mask, spatial_shapes,
                                                                              encoder_linear_layer, encoder_layer_norm)

        # 多分类：[bs, H/8 * W/8 + ... + H/64 * W/64, 256] -> [bs, H/8 * W/8 + ... + H/64 * W/64, num_classes]
        # 使用最后一个class_layer, bbox_layer(第7层)
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](proposal_memory) # (bs, sum(hi*wi), num_classes)
        enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](proposal_memory) + proposal_anchors # (bs, sum(hi*wi), 4)

        # 得到参考点reference_points/先验框
        topk = self.two_stage_topk_num  # topk=nq=300
        # topk_proposals: [bs, topk=nq]  topk index, 找到num_classes维度中最大logits值, 随后对这个值排序, 在∑Hi*Wi个查询中排前k个值的索引query_index
        topk_ind = torch.topk(enc_outputs_class.max(-1).values, topk, dim=1)[1]
        # refpoints_embed_unact_undetach: topk个分类得分最高的index对应的预测bbox (bs, nq, 4)
        refpoint_embed_unact_undetach = torch.gather(enc_outputs_coord_unact, 1, topk_ind.unsqueeze(-1).repeat(1, 1, 4))
        refpoint_embed_unact_ = refpoint_embed_unact_undetach.detach() # 先验参考点分离

        # 对encoder输出结果进行监督(包含init_box_proposals, enc_ref两部分联合监督)
        init_box_proposals = torch.gather(proposal_anchors, 1, topk_ind.unsqueeze(-1).repeat(1, 1, proposal_anchors.shape[-1])).sigmoid()
        enc_class = torch.gather(enc_outputs_class, 1, topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])) # (bs, topk=300, num_classes)
        enc_ref = refpoint_embed_unact_undetach.sigmoid() # (bs, topk=300, cxcywh=4)

        # cdn对比去噪训练(contrastive denoising)准备, num_cdn_groups=100, 生成噪声的代码
        # self.training是pytorch默认的行为, 其默认就是训练模式
        if self.training and targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta = \
                prepare_for_cdn(dn_args=(targets, self.num_cdn_groups, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=True, num_queries=self.num_queries, num_classes=self.num_classes,
                                hidden_dim=self.embed_dim, label_enc=self.label_enc)
        else:
            # 推理模式, 全部设置为None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        # dn_query 与dn_ref
        tgt = input_query_label # (bs, dn_q, 256)
        refpoint_embed_unact = input_query_bbox # (bs, dn_q, 4), 没有经过sigmoid, 且未detach的denoising_ref_unact_undetach
        
        # topk的操作之后一定要detach, 否则会干扰训练稳定性
        # 随机初始化生成tgt_ (num_queries), 否则从encoder中选取topk
        if self.learn_init_tgt:
            tgt_ = self.tgt_embed.weight[None, :, :].repeat(bs, 1, 1) # (bs, nq=300, embed_dim=256)
        else:
            tgt_ = proposal_memory.gather(dim=1, \
                    index=topk_ind.unsqueeze(-1).repeat(1, 1, proposal_memory.shape[-1]))
            tgt_ = tgt_.detach()
        # refpoint_embed_unact初始未曾归一化参考点获取
        if refpoint_embed_unact is not None:
            refpoint_embed_unact = torch.cat([refpoint_embed_unact, refpoint_embed_unact_], dim=1) # (bs, dnq+nq, 4), 未经过sigmoid
            tgt = torch.cat([tgt, tgt_], dim=1) # (bs, dnq+nq, 256)
        else:
            # 这种是推理模式, 没有去噪的内容, 传入的参数refpoint_embed_unact和tgt没有使用, 均为None
            # refpoint_embed维度 (bs, nq=300, xywh=4), tgt维度(bs, nq=topk300, embed_dim=256)
            refpoint_embed_unact, tgt = refpoint_embed_unact_, tgt_
        
        # 生成decoder的queries和init_reference, query_pos显式的迭代
        queries = tgt # (bs, dn_q + nq, 256)
        # 初始化的归一化参考点坐标 (bs, dn_q + nq, xywh=4)
        init_reference = refpoint_embed_unact.sigmoid() # 做初始归一化
        
        # 输出6个层级的预测结果, 6个层级的预测偏移分布, 6个层级的初始参考点, 1个传统头
        # 需要绘制热力图
        if draw_heatmap:
            image_shape = x.shape[2:]
        # 不需要绘制热力图
        else:
            image_shape = None
        dec_out_logits, dec_out_bboxes, dec_out_pred_corners, dec_out_refs, \
            traditional_logits, traditional_bboxes, heatmap = self.decoder(queries, 
                                                                       memory, 
                                                                       init_reference, 
                                                                       spatial_shapes, 
                                                                       level_start_index, 
                                                                       valid_ratios, 
                                                                       mask=src_mask, 
                                                                       attn_mask=attn_mask, 
                                                                       image_shape=image_shape)
        
        # 把最后一个enc的分类和定位层拿出作为2次联合损失监督, 使用传统vfl/mal、l1、giou损失监督
        # enc_class=(bs, topk=nq, num_classes), enc_ref=(bs, topk=nq, 4), init_box_proposals=(bs, topk=nq, 4)
        enc_out_logits = [enc_class, enc_class]
        enc_out_bboxes = [enc_ref, init_box_proposals]

        # 6个decoder输出结果, 训练时应包含对比去噪部分, 验证时没有对比去噪部分, 做一个stack的拼接
        dec_out_logits = torch.stack(dec_out_logits) # (6, bs, dnq+nq, num_classes) or (6, bs, nq, num_classes)
        dec_out_bboxes = torch.stack(dec_out_bboxes) # (6, bs, dnq+nq, 4) or (6, bs, nq, 4)
        dec_out_pred_corners = torch.stack(dec_out_pred_corners) # (6, bs, dnq+nq, 132) or (6, bs, nq, 132)
        dec_out_refs = torch.stack(dec_out_refs) # # (6, bs, dnq+nq, cxcywh=4) or (6, bs, nq, cxcywh=4)

        # 这段代码必须独立写在这里, 不要动位置, 对去噪与非去噪部分进行拆分
        if self.training and dn_meta is not None:
            # (bs, dnq, num_classes), (bs, nq, num_classes)
            dn_traditional_logits, traditional_logits = torch.split(traditional_logits, dn_meta["num_dn_split"], dim=1)
            # (bs, dnq, cxcywh=4), (bs, nq, cxcywh=4)
            dn_traditional_bboxes, traditional_bboxes = torch.split(traditional_bboxes, dn_meta["num_dn_split"], dim=1)
            # (6, bs, dnq, num_classes), (6, bs, nq, num_classes)
            dn_dec_logits, dec_out_logits = torch.split(dec_out_logits, dn_meta["num_dn_split"], dim=2)
            # (6, bs, dnq, cxcywh=4), (6, bs, nq, cxcywh=4)
            dn_dec_bboxes, dec_out_bboxes = torch.split(dec_out_bboxes, dn_meta["num_dn_split"], dim=2)
            dn_dec_corners, dec_out_pred_corners = torch.split(dec_out_pred_corners, dn_meta["num_dn_split"], dim=2)
            # 参考点都是同一个值
            dn_dec_refs, dec_out_refs = torch.split(dec_out_refs, dn_meta["num_dn_split"], dim=2)

        # 去噪分支部分层级输出
        dn_outs = {}
        # 去噪训练损失监督部分, 获取去噪索引
        if self.training and dn_meta is not None and targets is not None:
            max_gt, num_dn_groups = dn_meta['max_gt'], dn_meta['num_dn_groups']
            # dn_neg_idx没有用到
            dn_pos_idx, dn_neg_idx = get_cdn_indices(bs, max_gt, num_dn_groups, targets)
            
            # 去噪分支输出部分, 训练时启用
            dn_outs['indices'] = dn_pos_idx # 损失计算用匹配索引indices (query_indices, target_indices)
            dn_outs['num_dn_groups'] = num_dn_groups
            dn_outs['dn_traditional_logits'] = dn_traditional_logits # (bs, dnq, num_classes)
            dn_outs['dn_traditional_bboxes'] = dn_traditional_bboxes # (bs, dnq, cxcywh=4)
            dn_outs['dn_dec_logits'] = dn_dec_logits   # (6, bs, dnq, num_classes)
            dn_outs['dn_dec_bboxes'] = dn_dec_bboxes   # (6, bs, dnq, cxcywh=4)
            dn_outs['dn_dec_corners'] = dn_dec_corners # (6, bs, dnq, 4*33)
            dn_outs['dn_dec_refs'] = dn_dec_refs       # (6, bs, dnq, cxcywh=4)
        else:
            # 推理时不需要这些去噪参数, 全设置为None, 且代码不是刚需, 只是为了鲁棒性
            dn_outs['indices'] = None
            dn_outs['num_dn_groups'] = None
            dn_outs['dn_traditional_logits'] = None
            dn_outs['dn_traditional_bboxes'] = None
            dn_outs['dn_dec_logits'] = None
            dn_outs['dn_dec_bboxes'] = None
            dn_outs['dn_dec_corners'] = None
            dn_outs['dn_dec_refs'] = None

        # 这里的输出始终都是list[layers=8, (bs, nq, 256 or 4)]
        class_outs = [layer for layer in dec_out_logits] + [layer for layer in enc_out_logits] # layer x8
        bbox_outs = [layer for layer in dec_out_bboxes] + [layer for layer in enc_out_bboxes] # layer x8
        dec_out_corners = [layer for layer in dec_out_pred_corners] # layer x6
        dec_out_ref_initials = [layer for layer in dec_out_refs] # layer x6

        # 创建fine_grained_out, 用于存储细粒度分布权重方程所需要的参数
        fine_grained_out = {}
        # 参数用于fgl与ddf的损失计算
        fine_grained_out['reg_scale'] = self.reg_scale # 无梯度
        fine_grained_out['up'] = self.up # 无梯度

        if return_all_layers:
            # 用于训练, 默认是model.train(), 但是最好显式的调用model.train()
            # 返回6x dec_class + 2x enc_class, 6x dec_box + 2x enc_box, pred_corners, ref_initial, 均为(6, bs, num_query, num_classes or 4 or 132)
            # traditional class and bboxes, 均为(bs, num_query, num_classes or 4)
            # dn_outs 包括各种去噪分支的参数
            # fine_grained_out, 包括了所需要及计算fgl与ddf损失的无梯度参数
            return class_outs, bbox_outs, dn_outs, \
                   dec_out_corners, dec_out_ref_initials, traditional_logits, traditional_bboxes,\
                   fine_grained_out
        else:
            # 用于推理, 显式的用model.eval()调用, 调用索引idx=5, 也就是第6个decoder的输出结果 
            # (bs, num_queries, num_classes) and (bs, num_queries, cxcywh=4)
            # 需要热力图
            return class_outs[5], bbox_outs[5], heatmap