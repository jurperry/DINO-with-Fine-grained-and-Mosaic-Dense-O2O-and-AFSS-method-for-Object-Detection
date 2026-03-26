import torch
import random
from tqdm import tqdm 
from utils import box_cxcywh_to_xyxy, collate_fn 
from datasets import ValDataset_for_DETR
from torch.utils.data import DataLoader

def calculate_iou_tensor(box1, box2):
    """计算单框对多框的IoU (xyxy格式)"""
    lt = torch.max(box1[:2], box2[:, :2])
    rb = torch.min(box1[2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter
    return inter / union

# 抗遗忘采样策略(Anti-Forgetting Sampling Strategy, AFSS)
class AFSSManager:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        # 初始化状态字典：{'P': float, 'R': float, 'ep': int}
        # 初始时, P和R都设为0 (全部视为 Hard, 强制学习)
        self.state_dict = {i: {'P': 0.0, 'R': 0.0, 'ep': -1} for i in range(num_samples)}
    
    # 根据当前状态, 生成本轮参与训练的图片索引, 每一个epoch都会更新生成一次omega, 用于训练的样本索引集, 
    # 这个方法并不更新state_dict中的P, R记录, 更新ep的记录, 用于下一次epoch的训练,
    # 在跨 5 epochs更新P,R之前, 其中的hard, moderate与easy样本的image_id均相同, 
    # 额外随机采样M1与E2使得中等样本与简单样本在每个epoch中都有机会被采样,
    # 同时, 为了保持样本的多样性, 每个epoch中, 中等样本与简单样本的采样数量不能超过40%的样本总数,
    def get_epoch_subset(self, current_epoch):
        # omega是用于训练的部分样本的索引集
        omega = set()
        # 存储img_i的3个难度等级列表
        easy_pool, moderate_pool, hard_pool = [], [], []
        
        # 1. 划分难度等级
        for img_id, state in self.state_dict.items():
            sufficiency = min(state['P'], state['R'])
            # 充分度划分: Easy, Moderate, Hard
            if sufficiency > 0.85:
                easy_pool.append(img_id)
            elif 0.55 <= sufficiency <= 0.85:
                moderate_pool.append(img_id)
            else:
                hard_pool.append(img_id)
        
        # 2. Hard 样本全采样
        omega.update(hard_pool)
        
        # 3. Moderate 样本短时覆盖 (距离上次超过3轮的强制加入)
        # current_epoch是当前训练轮次, 从0开始, state_dict 中的ep从-1开始所以需要减1
        forced_mod = [img for img in moderate_pool if current_epoch - 1 - self.state_dict[img]['ep'] >= 3]
        omega.update(forced_mod)
        
        # moderat额外采样数M1, 总共需要40%的中等难度样本训练, M1 = 40% 总量 - 已强制数量(遗忘掉了的moderate样本)
        # 这是一种优先覆盖遗忘样本, 再随机补充剩余样本
        remain_mod = list(set(moderate_pool) - set(forced_mod))
        M1 = max(0, int(0.4 * len(moderate_pool)) - len(forced_mod))
        if M1 > 0 and remain_mod:
            omega.update(random.sample(remain_mod, min(M1, len(remain_mod))))
        
        # 4. Easy 样本持续复习 (距离上次超过10轮的强制加入)
        forced_easy = [img for img in easy_pool if current_epoch - 1 - self.state_dict[img]['ep'] >= 10]
        # 强制加入的数量最多只能占 easy 样本的 1%(0.5 × 2%)
        max_forced_easy = int(0.5 * 0.02 * len(easy_pool))
        if len(forced_easy) > max_forced_easy:
            forced_easy = random.sample(forced_easy, max_forced_easy)
        omega.update(forced_easy)
        
        # easy额外采样数E2, 总共只希望 2% 的easy样本参与训练, E2 = 2% 总量 - 已强制数量(遗忘掉了的easy样本, 最多只有1%)
        remain_easy = list(set(easy_pool) - set(forced_easy))
        E2 = max(0, int(0.02 * len(easy_pool)) - len(forced_easy))
        if E2 > 0 and remain_easy:
            omega.update(random.sample(remain_easy, min(E2, len(remain_easy))))
        
        # 5. 更新参与本轮训练的样本的记录
        # omega是用于强制完整训练的训练集样本, 需要被更新到current_epoch
        for img_id in omega:
            self.state_dict[img_id]['ep'] = current_epoch
            
        return list(omega)
    
    def print_sufficiency_distribution(self):
        """统计并打印当前数据集的学习充分度分布 (Easy/Moderate/Hard)"""
        easy_count, mod_count, hard_count = 0, 0, 0
        for state in self.state_dict.values():
            suff = min(state['P'], state['R'])
            if suff > 0.85:
                easy_count += 1
            elif 0.55 <= suff <= 0.85:
                mod_count += 1
            else:
                hard_count += 1
        
        total = len(self.state_dict)
        if total == 0:
            print("  AFSS State Dict is empty!")
            return
        
        print(f"  -> Easy     (>0.85): {easy_count:5d} / {total} ({easy_count/total*100:.1f}%)")
        print(f"  -> Moderate (0.55-0.85): {mod_count:5d} / {total} ({mod_count/total*100:.1f}%)")
        print(f"  -> Hard     (<0.55): {hard_count:5d} / {total} ({hard_count/total*100:.1f}%)")
        print("-" * 60)

    # 利用验证集的无增强DataLoader对全体训练集做一次快速推断, 更新state_dict中的P, R记录, 跨5个epochs更新一次
    # 全体训练集边推理边计算, 效率不高, 但可以实时监控模型在训练集上的学习进度, 所以设置5epochs做一次更新
    @torch.no_grad()
    def evaluate_and_update(self, 
                            model, 
                            train_imgdir, 
                            train_txtdir, 
                            device, 
                            max_size, 
                            val_size, 
                            conf_thresh=0.2, 
                            iou_thresh=0.5, 
                            batch_size=4, 
                            num_workers=0, 
                            pin_memory=True, 
                            prefetch_factor=None):
        """利用验证集的无增强DataLoader对训练集做一次快速推断, 更新状态"""
        # 使用 ValDataset (无数据增强) 来测试训练集图像的真实学习程度
        # 验证数据集
        eval_dataset = ValDataset_for_DETR(train_imgdir, 
                                           train_txtdir, 
                                           image_set='val', 
                                           max_size=max_size, 
                                           val_size=val_size)
        # 防御性编程, 避免 num_workers 为0时, prefetch_factor 为None
        safe_prefetch = prefetch_factor if num_workers > 0 else None
        # 验证数据集加载器
        eval_loader = DataLoader(eval_dataset,
                                      batch_size=batch_size, 
                                      shuffle=False, 
                                      collate_fn=collate_fn, 
                                      num_workers=num_workers,
                                      prefetch_factor=safe_prefetch,
                                      pin_memory=pin_memory)
        
        model.eval()
        img_idx = 0
        
        # ==== 包装 tqdm 进度条 ====
        # leave=False 表示进度条不会在最后显示, 避免占用屏幕空间, 建议保留leave=True
        pbar = tqdm(eval_loader, desc=f"AFSS Updating States (Thresh={conf_thresh:.3f})", leave=True)
        
        for batch in pbar:
            nested_tensor, real_id, real_norm_bboxes, _, _ = batch
            imgs, masks = nested_tensor.tensors.to(device), nested_tensor.mask.to(device)
            
            # ==== 调用 model 推理训练集 ====
            last_cls, last_bbox, heatmap = model(imgs, masks, return_all_layers=False, draw_heatmap=False)
            
            pred_cls_probability = last_cls.sigmoid()
            pred_bbox = box_cxcywh_to_xyxy(last_bbox)
            
            for i in range(imgs.size(0)):
                gt_classes = real_id[i].to(device)
                # 真实框需要转回 xyxy 格式计算 IoU
                gt_boxes = box_cxcywh_to_xyxy(real_norm_bboxes[i]).to(device)
                
                max_prob, max_cls_idx = torch.max(pred_cls_probability[i], dim=-1)
                
                # 单图计算 P 和 R
                p, r = self._compute_single_pr_v0(gt_boxes, gt_classes, pred_bbox[i], max_cls_idx, max_prob, conf_thresh, iou_thresh)
                
                self.state_dict[img_idx]['P'] = p
                self.state_dict[img_idx]['R'] = r
                img_idx += 1
        # 防御性编程
        model.train() # 切回训练模式, 这里实际可以省略, 没有什么作用

        # ==== 替换为调用独立的打印方法 ====
        print(f"\n[AFSS Update Complete] Dataset Sufficiency Distribution:")
        self.print_sufficiency_distribution()

    # 工程化P, R指标计算, 计算效率高(2x times faster), 但是评价效果不算客观, 依赖于验证集的conf_thresh指标, 精度却更高了
    def _compute_single_pr_v0(self, gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, conf_thresh, iou_thresh):
        # 直接使用最佳F1_max下的score阈值作为每一张图片的真实推理阈值, 以此评定这个图片PR值用于AFSS分类
        # 不能使用score=0.001 + top100做预测, 因为那是验证时候用的, 计算复杂度太高
        valid_mask = pred_scores >= conf_thresh
        pred_boxes = pred_boxes[valid_mask]
        pred_classes = pred_classes[valid_mask]
        pred_scores = pred_scores[valid_mask]
        
        sort_idx = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sort_idx]
        pred_classes = pred_classes[sort_idx]
        
        tp_count, fp_count = 0, 0
        gt_matched = [False] * len(gt_boxes)
        
        for p_box, p_cls in zip(pred_boxes, pred_classes):
            best_iou, best_idx = 0.0, -1
            if len(gt_boxes) > 0:
                # 一个pred_box与多个gt_boxes做ious计算
                ious = calculate_iou_tensor(p_box, gt_boxes)
                for j, (iou, g_cls) in enumerate(zip(ious, gt_classes)):
                    if not gt_matched[j] and p_cls == g_cls and iou > best_iou:
                        best_iou = iou.item()
                        best_idx = j
                        
            if best_iou >= iou_thresh and best_idx >= 0:
                gt_matched[best_idx] = True
                tp_count += 1
            else:
                fp_count += 1
        
        # 补上针对纯背景图片(没有任何 GT)的保护逻辑
        if len(gt_boxes) == 0:
            # 如果图里没目标, 且模型也没瞎报 (fp_count == 0), 说明学得极好 (P=1, R=1)
            # 如果模型瞎报了目标, 说明根本没学好 (P=0, R=0)
            return (1.0, 1.0) if fp_count == 0 else (0.0, 0.0)
            
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / len(gt_boxes)
        return precision, recall
    
    # 单一图片更加客观的P, R指标计算, 计算开销较大, 其中的conf_thresh是验证集的conf_thresh指标, 保留接口但是不使用
    def _compute_single_pr_v1(self, gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, conf_thresh, iou_thresh):
        # 1. 极低阈值过滤与 Top-K 提取
        # 保留 0.001 的极低底线，防止将纯噪音（如 1e-5 的得分）纳入计算, 同时保护纯背景图逻辑
        min_mask = pred_scores >= 0.001 
        pred_boxes = pred_boxes[min_mask]
        pred_classes = pred_classes[min_mask]
        pred_scores = pred_scores[min_mask]
        
        # 根据置信度降序排序, 并截取 Top-K=100
        sort_idx = torch.argsort(pred_scores, descending=True)[:100]
        pred_boxes = pred_boxes[sort_idx]
        pred_classes = pred_classes[sort_idx]
        pred_scores = pred_scores[sort_idx]
        
        num_preds = len(pred_boxes)
        num_gts = len(gt_boxes)
        
        # 2. 纯背景图的保护逻辑
        if num_gts == 0:
            # 纯背景图：如果经过 0.001 的极低过滤后没有任何预测，说明模型学得极好 (返回 1.0, 1.0)
            # 如果模型连纯背景图都报出了 > 0.001 的框，说明完全没学好 (返回 0.0, 0.0)
            return (1.0, 1.0) if num_preds == 0 else (0.0, 0.0)
        
        if num_preds == 0:
            # 有真实目标, 但模型连一个 > 0.001 的预测都没有
            return 0.0, 0.0
        
        # 3. 匹配过程：记录每个预测框是 TP 还是 FP
        # 使用 Tensor 记录，方便后续做 cumsum (累加) 操作
        tp_array = torch.zeros(num_preds, device=pred_boxes.device)
        fp_array = torch.zeros(num_preds, device=pred_boxes.device)
        gt_matched = [False] * num_gts
        
        for i, (p_box, p_cls) in enumerate(zip(pred_boxes, pred_classes)):
            best_iou, best_idx = 0.0, -1
            # 沿用你原有的 IoU 计算逻辑
            ious = calculate_iou_tensor(p_box, gt_boxes)
            for j, (iou, g_cls) in enumerate(zip(ious, gt_classes)):
                if not gt_matched[j] and p_cls == g_cls and iou > best_iou:
                    best_iou = iou.item()
                    best_idx = j
                    
            if best_iou >= iou_thresh and best_idx >= 0:
                gt_matched[best_idx] = True
                tp_array[i] = 1.0 # 此处用到i
            else:
                fp_array[i] = 1.0 # 此处用到i
        
        # 4. 核心逻辑：累计计算各置信度截断下的 Precision, Recall
        # cumsum 可以一次性计算出 Top-1, Top-2 ... Top-N 时刻的累积 TP 和 FP
        acc_tp = torch.cumsum(tp_array, dim=0)
        acc_fp = torch.cumsum(fp_array, dim=0)
        
        # 加上 1e-6 防止分母为 0, 当预测框为空时, 0除以0会报错(原则上不会出现这种情况, 但为了保险起见, 还是加上1e-6)
        precision = acc_tp / (acc_tp + acc_fp + 1e-6)
        recall = acc_tp / num_gts
        
        # 5. 计算整个序列的 F1 Score, 并找到使 F1 最大的截断点, 1e-6保证数值稳定性
        f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
        best_idx = torch.argmax(f1_scores)
        
        # 返回单图 F1_max 下的 Precision 和 Recall
        return precision[best_idx].item(), recall[best_idx].item()