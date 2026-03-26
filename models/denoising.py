import torch
from utils import inverse_sigmoid

# cdn对比去噪实现部分
def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
        dn_args: (targets, num_cdn_groups, label_noise_ratio, box_noise_scale), 其中设置如下
            num_cdn_groups = 100
            dn_label_noise_ratio = 0.5, 标签加噪比例
            dn_box_noise_scale = 1.0, 框加噪比例
        training: if it is training or inference
        num_queries: number of queires
        num_classes: number of classes
        hidden_dim: transformer hidden dim
        label_enc: encode labels in dn, self.label_enc = nn.Embedding(dn_labelbook_size+1, hidden_dim=256)
        一般dn_labelbook_size=num_classes, 加1是为了保险起见, 只要dn_labelbook_size词汇表长度大于num_classes即可
        return:
            input_query_label: [batch_size, num_dn_query, hidden_dim]  # 去噪查询的标签embedding, dn_queries(tgt)
            input_query_bbox:  [batch_size, num_dn_query, 4]           # 去噪查询的框embedding, dn_refpoints(refpoints_embed_unact)
            attn_mask:         [tgt_size, tgt_size]                # 注意力掩码(decoder自注意力), target_size = num_dn_query + num_query
            dn_meta: Dict      # 包含num_dn_query和num_dn_group的元数据
    """
    if training:
        targets, num_cdn_groups, label_noise_ratio, box_noise_scale = dn_args
        # 正负样本数量相同, positive and negative dn queries
        # 每个GT生成2*num_cdn_groups个噪声样本(正负各一半, 均为num_cdn_groups个)
        num_cdn_groups = num_cdn_groups * 2
        # bs大小的list, item是全是1的tensor, size是各个image gt的数量
        known = [torch.ones_like(t) for t in targets['labels']]
        batch_size = len(known)
        # 各个image gt框的数量, 用list的形式存储下来
        known_num = [sum(k) for k in known]
        max_gt = int(max(known_num))

        # 如果没有gt, 则设为1, 如果num_cdn_groups太大, 根据最大GT数量调整, 这里是做cdn对比去噪组数
        if int(max(known_num)) == 0:
            num_cdn_groups = 1
        else:
            if num_cdn_groups >= 100:
                num_cdn_groups = num_cdn_groups // (max_gt * 2)
            elif num_cdn_groups < 1:
                num_cdn_groups = 1
        if num_cdn_groups == 0:
            num_cdn_groups = 1

        # 取出所有gt的label值
        labels = torch.cat([t for t in targets['labels']]) # [total_gt]
        # 取出所有gt的box
        boxes = torch.cat([t for t in targets['boxes']]) # [total_gt, 4]
        # 标识每个GT属于哪个图片 如bs=4, batch_idx = tensor([0, 0, 0, 0,// 1,// 2, 2,// 3...], device='cuda:0')
        batch_idx = torch.cat([torch.full_like(t.long(), i) for i, t in enumerate(targets['labels'])]) # (total_gt,)

        # 这三项是gt相关的值
        known_labels = labels.repeat(2 * num_cdn_groups, 1).view(-1) # [total_gt * 2*num_cdn_groups,]
        known_bid = batch_idx.repeat(2 * num_cdn_groups, 1).view(-1) # [total_gt * 2*num_cdn_groups,]
        known_bboxs = boxes.repeat(2 * num_cdn_groups, 1) # [total_gt * 2*num_cdn_groups, 4]
        # 克隆, 防止内存共享
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()
        
        # 正负样本是根据bbox的偏移决定的(人为设定正负样本)
        # 对标签加噪
        if label_noise_ratio > 0:
            # 随机值, 0-1内
            p = torch.rand_like(known_labels_expaned.float()) # [total_gt * 2*num_cdn_groups]
            # 随机选择一部分样本（比例为label_noise_ratio/2）
            # torch.nonzero(...) → 默认返回一个 二维"索引"张量, shape为(M, 1), 其中M是满足条件的元素个数（即被选中的样本数）
            # 只有 p < (label_noise_ratio * 0.5)才是True, 索引被保留下来, 需要被加噪
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # shape=(M,)
            # 将这些样本的标签替换为随机类别(int random), 给被选择的gt一个随机的label_id, 范围[0, num_classes)
            new_label = torch.randint_like(chosen_indice, 0, num_classes) # shape=(M,)
            # 使用 scatter_ 原地修改known_labels_expaned：在维度dim=0上, 将chosen_indice对应位置的值替换为new_label中的值
            known_labels_expaned.scatter_(0, chosen_indice, new_label) # [total_gt * 2*num_cdn_groups]

        # bs中某单一图片还有最多的gt的数量, num_denoising_query就是一个batch中总的正负样本数量之和
        num_denoising_query = int(max_gt * 2*num_cdn_groups) # 注意区分是max_gt *2*num_cdn_groups, 而不是total_gt *2*num_cdn_groups
        # [num_cdn_groups, total_gt], 也即[组数, total_gt], range(len(boxes)): [0, total_gt)
        # positive_idx: (dn_num, total_gt), 每一行都是这个batch下的gt索引 num_cdn_groups x [0, 1, 2, 3..., total_gt-1]
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(num_cdn_groups, 1)
        # 加上了group间的偏移量, G=total_gt=len(boxes)
        # torch.tensor(range(num_cdn_groups)) → [0, 1, 2, ..., num_cdn_groups-1]
        # * len(boxes) * 2 → 每组占 2 × G 个位置（正 + 负）
        # 所以偏移量为：[0, 2G, 4G, 6G, ..., 2*(num_cdn_groups-1)*G]
        # [0, 2G, 4G, 6G, ..., 2*(num_cdn_groups-1)*G].unsqueeze(1) --> (num_cdn_groups, 1)二维列向量, 每一行加组间距0, 2G, 4G...
        positive_idx += (torch.tensor(range(num_cdn_groups)) * len(boxes) * 2).long().cuda().unsqueeze(1) 
        # 保证构建的正负样本索引连续, total_gt * 2*num_cdn_groups, 其中total_gt*2 是连续索引的正负样本(正样本在前, 负样本在后), 共num_cdn_groups个组
        # [dn_numbnum_cdn_groupser * total_gt, ], 展平得到正样本索引, 这个正样本索引用于构造负样本索引
        positive_idx = positive_idx.flatten()
        # [num_cdn_groups * total_gt, ] 正好剩下的位置是留给negative的, 通过加上G=total_gt, 对negtive_idx索引计算
        negative_idx = positive_idx + len(boxes) # 负样本索引用于负样本bbox构建

        # 对边界框加噪
        if box_noise_scale > 0:
            # 创建全零张量, 只是一个模版
            known_bbox_ = torch.zeros_like(known_bboxs) # [total_gt * 2*num_cdn_groups, 4]
            # 中心坐标减掉宽高的一半，左上的边界点
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2 # cxcy -> x1y1
            # 中心坐标加上宽高的一半，右下的边界点 
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2 # wh -> x2y2

            # diff是角点允许偏移最大范围幅度
            diff = torch.zeros_like(known_bboxs) # [total_gt * 2*num_cdn_groups, 4]
            # 宽高的一半放到中心坐标的位置(占位, 为了计算使用)
            diff[:, :2] = known_bboxs[:, 2:] / 2  # 左上点的扰动范围 = 半宽高
            # 宽高是宽高的一半
            diff[:, 2:] = known_bboxs[:, 2:] / 2  # 右下点的扰动范围 = 半宽高
            # torch.randint_like(known_bboxs, low=0, high=2) 选出的值都是0, 1这两种整数值
            # torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0 选出的值是-1 或者 1
            # rand_sign为-1或1, 随机规定角点移动是正是负
            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0

            ### ========== DeNoising对比去噪框浮动 ==========
            # 正样本λ1 < 负样本λ2,   0<=λ1<1, 1<=λ2<2
            # rand_part 0-1内的值浮点数值, 正样本浮动乘区[0, 1)
            rand_part = torch.rand_like(known_bboxs) # [total_gt * 2*num_cdn_groups, 4]
            # 负样本位置的值+1, 负样本的偏离比正样本更多, 负样本浮动乘区[1, 2)
            rand_part[negative_idx] += 1.0 # [total_gt * 2*num_cdn_groups, 4]
            # [total_gt * 2*num_cdn_groups, 4] 坐标位置随机的乘上1或者-1, 正偏移或者负偏移
            rand_part *= rand_sign
            ### ========== DeNoising对比去噪框浮动 ==========

            # 加上随机的偏移，左上, 右下的点随机的进行了偏移, rand_part乘区 * 最大偏移量diff(半宽高) * 加噪尺度box_noise_scale(=1.0)
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            # 裁剪, 防止溢出, 但是有可能裁剪之后没有框了
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            # 左上和右下点的和除2就是中心点坐标
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            # 右下减去左上的差值，就是高宽
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2] # known_bbox_expand已经加噪, (total_gt*2*num_cdn_groups, 4)

        # 这里的known_labels_expaned 已经被添加过随机的噪声了
        m = known_labels_expaned.long().to('cuda') # (total_gt * 2*num_cdn_groups, )
        # label_enc = nn.Embedding(dn_labelbook_size + 1 = num_classes+1, hidden_dim=256)
        input_label_embed = label_enc(m) # (total_gt * 2*num_cdn_groups, 256)
        
        # 对坐标取反函数, 对应于特征图上的坐标
        input_bbox_embed = inverse_sigmoid(known_bbox_expand) # (total_gt * 2*num_cdn_groups, 4)

        # 全零dn_label_query(tgt), 填充处理到同一批次尺度, 便于训练
        padding_label = torch.zeros(num_denoising_query, hidden_dim).cuda() # [num_denoising_query, 256]
        # 全零dn_anchors(ref), 填充处理到同一批次尺度, 便于训练
        padding_bbox = torch.zeros(num_denoising_query, 4).cuda() # [num_denoising_query, 4]
        # 重复bs [num_denoising_query, 256] -> [bs, num_denoising_query, 256]
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        # 重复bs [num_denoising_query,4] -> [bs, num_denoising_query, 4]
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        # 如果有gt的话
        if len(known_num):
            # 各个image的合并在一起了, map_known_indice=like tensor([0, 1, 2, 3, 4, 5, 6,//  0, 1, 2, 3,//  0, 1, 2,//...])
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num]) # (total_gt, )
            # map_known_indice=like tensor([0, 1, 2, 3, 4, 5, 6,//  0, 1, 2, 3,//  0, 1, 2,//..., 0+s*i, 1+s*i...])
            map_known_indice = torch.cat([map_known_indice + max_gt * i for i in range(2 * num_cdn_groups)]).long() # shape:(total_gt * 2*num_cdn_groups,)

        # known_bid标识属于哪个图片的, shape = (total_gt * 2*num_cdn_groups, )
        if len(known_bid):
            # 按每个dn_query对应的批次bs与idx, 把input_label_embed按顺序填充进input_query_label
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed # [bs, pad_size, 256]

            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed # [bs, pad_size, 4]

        # cdn总共的denoising_query数量, 包括了正负样本以及填充(pad部分: bs*max_gt - total_gt), num_queries是正常的query的数量
        tgt_size = num_denoising_query + num_queries

        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0 # all False

        # 注意力矩阵中每一行代表该行对应query与其他query之间的注意力分数
        # 每一行横着看, [num_denoising_query:, : num_denoising_query]就是前者为行索引中num_query部分, 后者为列索引dn_query部分, num_query不能看到dn_query
        # 因此这些部分设为True屏蔽, 而dn_query可以看到num_query, 因此列索引中dn_query之后的部分依旧为False
        # 这种设计是单项屏蔽, 只允许dn_query看num_query进行训练, 但是num_query不能看dn_query训练(推理没有dn_query)
        attn_mask[num_denoising_query:, :num_denoising_query] = True

        # 各个组的掩码
        # num_denoising_query = max_gt * 2*num_cdn_groups
        # max_gt * 2  num_cdn_groups = num_dn_query, 而dn_query又被分为num_cdn_groups个组, 一个组包含max_gt*2(正负样本) 个 query

        # 接下来是dn_query内部的屏蔽, 组与组之间不能有注意力(上三角 + 下三角掩码)
        # 一个组是 max_gt*2 个dn_query, 注意力矩阵为(max_gt*2, max_gt*2), 组内可以注意力计算, 组外全部屏蔽
        for i in range(num_cdn_groups):
            # 第一组
            if i == 0:
                # 看不到他后面的所有
                attn_mask[max_gt * 2 * i:num_denoising_query * 2 * (i + 1), max_gt * 2 * (i + 1):num_denoising_query] = True
            # 最后一组
            if i == num_cdn_groups - 1:
                # 看不到他前面的所有
                attn_mask[max_gt * 2 * i:max_gt * 2 * (i + 1), :max_gt * i * 2] = True
            else:
                # 中间组
                # 看不到他后面的
                attn_mask[max_gt * 2 * i:max_gt * 2 * (i + 1), max_gt * 2 * (i + 1):num_denoising_query] = True
                # 也看不到他前面的
                attn_mask[max_gt * 2 * i:max_gt * 2 * (i + 1), :max_gt * 2 * i] = True

        dn_meta = {
            'num_dn_split': [num_denoising_query, num_queries],
            'num_dn_groups': num_cdn_groups,
            'max_gt': max_gt,
        }
    # 如果不是训练阶段的话, 去噪关闭
    else:
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    # input_query_bbox是dn_query锚点, 没有sigmoid归一化, shape = (bs, num_dn_query, 4)
    # input_query_label是dn_query的tgt(Cq部分), shape = (bs, num_dn_query, 256)
    return input_query_label, input_query_bbox, attn_mask, dn_meta

def get_cdn_indices(batch_size, max_gt, num_dn_groups, targets):
    dn_pos_idx = []
    dn_neg_idx = []
    # targets包含了labels 和 boxes信息, 以字典列表的形式存储
    for i in range(0, batch_size):
        # 如果目标图片存在gt
        if len(targets['labels'][i]) > 0:
            # torch.range(i, j)是两个闭区间 [i~j], 因此要人为个gt_num-1作为末尾索引, 建议使用torch.arange, 符合python语法惯例
            t = torch.arange(0, len(targets['labels'][i])).long().cuda() # range从 0 到 "样本i的总gt数 - 1", 为整数, 其实就是样本索引
            t = t.unsqueeze(0).repeat(num_dn_groups, 1)  # [num_dn_groups, gt_num], 也就是样本序号重复num_dn_group次
            # 目标图像的gt索引
            tgt_idx = t.flatten()  # (num_dn_groups*gt_num, )-->([gt_num], [gt_num], ..., [gt_num])-cat起来(也就是flatten效果)
            
            # decoder输出的索引dn_query索引
            # (num_dn_groups, gt_num) + (num_dn_groups, 1) ->(num_dn_groups, gt_num)
            output_idx = t + (torch.tensor(range(num_dn_groups)) * max_gt*2).long().cuda().unsqueeze(1)
            output_idx = output_idx.flatten() # (num_dn_group*gt_num, )
        
        # 防止空标注, 导致报错
        else:
            output_idx = tgt_idx = torch.tensor([]).long().cuda()
        # 这里第一个参数就是提议框的id(顺序排列), 第二个参数就是gt的id的意思(与提议id一一对应)
        # 把它看做是匈牙利匹配结束后的结果indice即可
        dn_pos_idx.append((output_idx, tgt_idx)) # 用于后续损失监督, 每一个decoder下的索引都是相同的
        dn_neg_idx.append((output_idx + max_gt, tgt_idx)) # 这个变量并没有使用
    
    return dn_pos_idx, dn_neg_idx