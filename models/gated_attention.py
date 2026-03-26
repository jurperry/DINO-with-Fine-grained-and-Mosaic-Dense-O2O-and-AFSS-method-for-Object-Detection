import torch
import torch.nn as nn

# 多头门控注意力机制(使用于自注意力或交叉注意力)
# 修改自通义千问Qwen3 Next
class MultiheadGatedAttention(nn.Module):
    def __init__(self, embed_dim=256, n_head=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        assert self.head_dim * n_head == self.embed_dim

        # 权重映射
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)

        # 门控映射
        self.gate_proj = nn.Linear(embed_dim, embed_dim)

        # dropout防止过拟合, 对注意力权重随机置换为0
        self.attn_dropout = nn.Dropout(dropout)  # 作用于attention weights

    def forward(self, query, key, value, Xq, attn_mask=None, key_padding_mask=None):
        '''
        query: 查询, query = query + query_pos, dim=(bs, Nq, embed_dim)
        key: 键, key = key + key_pos, 自注意力时, key = query + query_pos, dim=(bs, Nk, embed_dim)
        value: 值, 自注意力时, value = query(无query_pos), 交叉注意力, value = key(无key_pos), dim=(bs, Nv = Nk or Nq, embed_dim)
        Xq: 前归一化后得到的未添加位置编码的查询, dim=(bs, Nq, embed_dim)
        attn_mask: 因果掩码 (Nq, Nk)
        key_padding_mask: 填充掩码(bs, Nk)
        '''
        B, Nq, _ = query.shape  # (bs, Nq)
        B, Nk, _ = key.shape # (bs, Nk)

        q = self.Wq(query).reshape(B, Nq, self.n_head, self.head_dim).permute(0, 2, 1, 3) # (bs, n_head, Nq, head_dim)
        k = self.Wk(key).reshape(B, Nk, self.n_head, self.head_dim).permute(0, 2, 1, 3) # (bs, n_head, Nk, head_dim)
        v = self.Wv(value).reshape(B, Nk, self.n_head, self.head_dim).permute(0, 2, 1, 3) # (bs, n_head, Nk, head_dim)
        # q*kT = (bs, n_head, Nq, head_dim) * (bs, n_head, heda_dim, Nk) -> (bs, n_head, Nq, Nk)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))

        # 应用attn_mask掩码
        if attn_mask is not None:
            # attn_mask=(Nq, Nk) ---> (1, 1, Nq, Nk)
            attn = attn.masked_fill(attn_mask.unsqueeze(0).unsqueeze(1), float('-inf'))

        # 应用key_padding_mask掩码
        if key_padding_mask is not None:
            # key_padding_mask=(bs, Nk) --->(bs, 1, 1, Nk), padding掩码掩蔽的是注意力矩阵的列
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = attn.softmax(dim=-1) # softmax把-inf变成0
        
        # 训练时dropout防止过拟合
        if self.training:
            attn = self.attn_dropout(attn) # 对注意力权重进行dropout正则化

        # 多头输出, [B, n_head, Nq, head_dim]
        x_heads = attn @ v  # QWen3 Next论文中的Y
        # 用Xq生成门控logits: [B, Nq, embed_dim], 这里的Xq是前归一化后的输入query, 不加位置编码
        gate_logits = self.gate_proj(Xq)  # [B, Nq, embed_dim], 论文中的XWθ
        # reshape成多头形式: [B, Nq, n_head, head_dim] → permute → [B, n_head, Nq, head_dim]
        gate = torch.sigmoid(
            gate_logits.reshape(B, Nq, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        )  # [B, n_head, Nq, head_dim], 论文中的σ(XWθ)
        # 逐元素门控(每个head独立)
        x_heads_gated = x_heads * gate  # [B, n_head, Nq, head_dim], 论文中的Y*σ(XWθ)哈达玛积
        # 拼接所有head
        x_concat = x_heads_gated.transpose(1, 2).reshape(B, Nq, self.embed_dim)  # [B, Nq, embed_dim]
        # Wo映射输出最终结果
        output = self.Wo(x_concat) 
        return output, attn

        