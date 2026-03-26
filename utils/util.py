import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.boxes import box_area

import math
import inspect
import functools
import importlib
from collections import defaultdict
from typing import Any, List, Optional

# 源码补充: 要检查合法性, 不能出现x1 > x2, 或者 y1 > y2, 这样会导致giou_loss为负数
# giou_loss为负数的情况复现: pred=[0.3, 0.3, 0.1, 0.1], gt=[0.1, 0.1, 0.2, 0.2], 这两个框的giou_loss=-3.000
def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w.clamp(min=0)), (y_c - 0.5 * h.clamp(min=0)),
         (x_c + 0.5 * w.clamp(min=0)), (y_c + 0.5 * h.clamp(min=0))]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

# for single box iou
def calculate_iou(box1, box2):
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])
    
    inter_area = max(0, xmax_inter - xmin_inter) * max(0, ymax_inter - ymin_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou

def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

# the_list:[[3,h1,w1], [3,h2,w2]...]
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0] # 取第一个子列表作为初始值
    for sublist in the_list[1:]: # 遍历除第一个子列表外的列表, 进行逐维度对比, 选出较大的那一个值
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

# 用于适应swin_transformer的输入要求, 要求输入的高度H和宽度W必须是4的倍数
# resnet等cnn算法不需要, 但是为了代码简洁性, 依旧采取这个措施实现4倍数
def _round_up_to_multiple_of_4(x):
    """将数值向上取整到最近的4的倍数"""
    return ((x + 3) // 4) * 4

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    # 检查是否是三维的图像 (channel, H, W), tensor_list存储的是一张一张图片张量
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        # make sure the H&W is 4x multiple
        max_size[1] = _round_up_to_multiple_of_4(max_size[1])  # H
        max_size[2] = _round_up_to_multiple_of_4(max_size[2])  # W
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img也是可行的
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

# 类的实例化: NestedTensor(tensor, mask), 这样就实例就创建了, 以后直接多次调用即可, 也可创建多个实例
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    # 这是NestedTensor的一个方法, 能够分解出一个批次内的图像tensor和对应的填充掩码mask
    def decompose(self):
        return self.tensors, self.mask

    # 打印结果只是一个字符串, 就是打印出tensors属性的字符串
    def __repr__(self):
        return str(self.tensors)

# 多阶段调度器, 在训练中只会初始化一次, 后续调用step方法时, 会根据当前epoch更新学习率
class MultiStageLRScheduler:
    """
    多阶段学习率调度器, 根据训练的4个阶段调整学习率
    
    阶段划分：
    1. Warmup阶段: 学习率从0线性增长到配置学习率
    2. Mosaic增强阶段: 使用配置学习率
    3. 普通增强阶段: 使用配置学习率
    4. Finetune阶段: 学习率降低到原来的0.1倍
    """
    def __init__(self, optimizer, warmup_epochs, final_finetune_epochs, total_epochs, mosaic_ratio=0.625):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.final_finetune_epochs = final_finetune_epochs
        self.total_epochs = total_epochs
        self.mosaic_ratio = mosaic_ratio
        
        # 计算各阶段的epoch数
        remaining_epochs = total_epochs - warmup_epochs - final_finetune_epochs
        self.mosaic_epochs = int(remaining_epochs * mosaic_ratio)
        self.normal_aug_epochs = remaining_epochs - self.mosaic_epochs
        
        # 计算各阶段的起始和结束epoch
        self.mosaic_start_epoch = warmup_epochs
        self.mosaic_end_epoch = self.mosaic_start_epoch + self.mosaic_epochs
        self.normal_aug_start_epoch = self.mosaic_end_epoch
        self.normal_aug_end_epoch = self.normal_aug_start_epoch + self.normal_aug_epochs
        self.finetune_start_epoch = self.normal_aug_end_epoch
        
        # 初始学习率只初始化一次, 由配置文件或命令行决定, 不会改变
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # 初始化last_epoch
        self.last_epoch = -1
        
        print(f"MultiStageLRScheduler initialized:")
        # warmup_stage
        if self.warmup_epochs > 0:
            print(f"  Warmup (linear): epochs 0-{self.warmup_epochs-1}")
        else:
            print(f"  Warmup (linear): None")
        # mosaic_stage
        if self.mosaic_epochs > 0:
            print(f"  Mosaic aug (base lr): epochs {self.mosaic_start_epoch}-{self.mosaic_end_epoch-1}")
        else:
            print(f"  Mosaic aug (base lr): None")    
        # normal_aug_stage
        if self.normal_aug_epochs > 0:
            print(f"  Normal aug (base lr): epochs {self.normal_aug_start_epoch}-{self.normal_aug_end_epoch-1}")
        else:
            print(f"  Normal aug (base lr): None")
        # finetune_stage
        if self.final_finetune_epochs > 0:
            print(f"  Finetune (0.1x lr): epochs {self.finetune_start_epoch}-{self.total_epochs-1}")
        else:
            print(f"  Finetune (0.1x lr): None")
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            if self.warmup_epochs > 0:
                lr_scale = (epoch + 1) / self.warmup_epochs
            else:
                lr_scale = 1.0
        elif epoch < self.finetune_start_epoch:
            # Mosaic和普通增强阶段：使用基础学习率
            lr_scale = 1.0
        else:
            # Finetune阶段：学习率降低到0.1倍
            lr_scale = 0.1
        
        # 更新所有参数组的学习率
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale
    
    # 貌似没什么用(用于打印每轮训练时候的optimizer的学习率)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr_scale = (self.last_epoch + 1) / self.warmup_epochs
        elif self.last_epoch < self.finetune_start_epoch:
            lr_scale = 1.0
        else:
            lr_scale = 0.1
        return [base_lr * lr_scale for base_lr in self.base_lrs]

    def state_dict(self):
        """保存调度器状态字典, 
           保存训练参数:
                    base_lrs (基础学习率配置)
                    last_epoch (当前训练的epoch数)
        """
        return {
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs, # 可能有用, 可能没用
        }
    
    def load_state_dict(self, state_dict):
        """
        断点继续训练, 需要从checkpoints加载last_epoch, base_lrs, 不支持继续训练时修改base_lrs
        """
        # 如果resume, 则加载last_epoch, base_lrs
        self.last_epoch = state_dict['last_epoch'] # 重要参数必须加载
        self.base_lrs = state_dict['base_lrs'] # 视情况选择使用不修改还是修改

def build_optimizer_and_scheduler(model, 
                                  lr=2e-4, 
                                  lr_backbone=2e-5, 
                                  lr_linear_proj_mult=0.1, 
                                  weight_decay=0.0001, 
                                  optimizer_type="adamw", 
                                  warmup_epochs=5,
                                  final_finetune_epochs=5, 
                                  total_epochs=50, 
                                  mosaic_ratio=0.625):
    """
    构建优化器和学习率调度器, 支持4个阶段的学习率调度
    
    Args:
        model: 模型实例
        lr: 主学习率 (默认: 2e-4)
        lr_backbone: backbone学习率 (默认: 2e-5)
        lr_linear_proj_mult: 线性投影层学习率倍数 (默认: 0.1)
        weight_decay: 权重衰减 (默认: 1e-4)
        optimizer_type: 优化器类型, "adamw" 或 "sgd" (默认: "adamw")
        lr_drop: 学习率下降的epoch间隔 (默认: 40) - 已废弃，保留参数兼容性
        warmup_epochs: warmup阶段epoch数 (默认: 5)
        final_finetune_epochs: 最后finetune阶段epoch数 (默认: 5)
        total_epochs: 总训练epoch数 (默认: 50)
        mosaic_ratio: mosaic阶段占中间阶段的比例 (默认: 0.625)
    """
    
    # 定义参数分组关键词
    lr_backbone_names = ["patch_embed"]
    lr_linear_proj_names = ['reference_points', 'sampling_offsets']
    
    def match_name_keywords(n, name_keywords):
        """检查参数名称是否包含关键词"""
        for b in name_keywords:
            if b in n:
                return True
        return False

    # 参数分组 - 分为三组
    param_dicts = [
        # 第1组：主干网络参数 (ResNet)
        {
            "params": [p for n, p in model.named_parameters()
                      if match_name_keywords(n, lr_backbone_names) and p.requires_grad],
            "lr": lr_backbone,
        },
        # 第2组：线性投影层和其他映射层
        {
            "params": [p for n, p in model.named_parameters() 
                      if match_name_keywords(n, lr_linear_proj_names) and p.requires_grad],
            "lr": lr * lr_linear_proj_mult,
        },
        # 第3组：剩余所有参数 (Encoder/Decoder的注意力机制、ffn等)
        {
            "params": [p for n, p in model.named_parameters()
                      if not match_name_keywords(n, lr_backbone_names) 
                      and not match_name_keywords(n, lr_linear_proj_names) 
                      and p.requires_grad],
            "lr": lr,
        }
    ]
    
    # 调试信息：打印各组的参数数量和示例
    print("Group params statistic:")
    total_params = 0
    for i, group in enumerate(param_dicts):
        param_count = sum(p.numel() for p in group["params"])
        total_params += param_count
        
        group_name = ["Backbone(ResNet)", "Linear_proj", "Main"][i]
        print(f" group{i+1}({group_name}): {param_count:,} params, lr: {group['lr']:.1e}")

    print(f"  total_params: {total_params:,}")
    
    # 选择优化器, 优化器优化param_dicts中的参数, 学习率为lr
    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(param_dicts, lr=lr, momentum=0.9, weight_decay=weight_decay)
        print(f"Using SGD optimizer, momentum=0.9")
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
        print(f"Using AdamW optimizer")
    
    # 学习率调度器 - 使用多阶段调度器
    lr_scheduler = MultiStageLRScheduler(
        optimizer, 
        warmup_epochs=warmup_epochs,
        final_finetune_epochs=final_finetune_epochs,
        total_epochs=total_epochs,
        mosaic_ratio=mosaic_ratio
    )
    
    return optimizer, lr_scheduler

# sigmoid反归一化函数
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm

# 高频位置查询需要的高频位置生成(decoder每次迭代需要的高频位置查询pos_query的高频位置)
def gen_sineembed_for_position(pos_tensor:torch.Tensor):
    '''
    pos_tensor: (bs, len_q, cxcywh)
    sineembed_tensor: (bs, len_q, 512)
    '''
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale # (bs, nq)
    y_embed = pos_tensor[:, :, 1] * scale # (bs, nq)
    pos_x = x_embed[:, :, None] / dim_t # (bs, nq, 128)
    pos_y = y_embed[:, :, None] / dim_t # (bs, nq 128)
    # (bs, nq, 64) x2 -> (bs, nq, 64, 2) -> (bs, nq, 128)
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    # (bs, nq, 64) x2 -> (bs, nq, 64, 2) -> (bs, nq, 128)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    w_embed = pos_tensor[:, :, 2] * scale
    pos_w = w_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    h_embed = pos_tensor[:, :, 3] * scale
    pos_h = h_embed[:, :, None] / dim_t
    pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

    # (bs, nq, 128) x4 -> (bs, nq, 512)
    sinembed_tensor = torch.cat((pos_x, pos_y, pos_w, pos_h), dim=2)

    return sinembed_tensor # (bs, nq, 512)


GLOBAL_CONFIG = defaultdict(dict)
def register(dct: Any = GLOBAL_CONFIG, name=None, force=False):
    """
    dct:
        if dct is Dict, register foo into dct as key-value pair
        if dct is Clas, register as modules attibute
    force
        whether force register.
    """

    def decorator(foo):
        register_name = foo.__name__ if name is None else name
        if not force:
            if inspect.isclass(dct):
                assert not hasattr(dct, foo.__name__), f"module {dct.__name__} has {foo.__name__}"
            else:
                assert foo.__name__ not in dct, f"{foo.__name__} has been already registered"

        if inspect.isfunction(foo):

            @functools.wraps(foo)
            def wrap_func(*args, **kwargs):
                return foo(*args, **kwargs)

            if isinstance(dct, dict):
                dct[foo.__name__] = wrap_func
            elif inspect.isclass(dct):
                setattr(dct, foo.__name__, wrap_func)
            else:
                raise AttributeError("")
            return wrap_func

        elif inspect.isclass(foo):
            dct[register_name] = extract_schema(foo)

        else:
            raise ValueError(f"Do not support {type(foo)} register")

        return foo

    return decorator


def extract_schema(module: type):
    """
    Args:
        module (type),
    Return:
        Dict,
    """
    argspec = inspect.getfullargspec(module.__init__)
    arg_names = [arg for arg in argspec.args if arg != "self"]
    num_defualts = len(argspec.defaults) if argspec.defaults is not None else 0
    num_requires = len(arg_names) - num_defualts

    schame = dict()
    schame["_name"] = module.__name__
    schame["_pymodule"] = importlib.import_module(module.__module__)
    schame["_inject"] = getattr(module, "__inject__", [])
    schame["_share"] = getattr(module, "__share__", [])
    schame["_kwargs"] = {}
    for i, name in enumerate(arg_names):
        if name in schame["_share"]:
            assert i >= num_requires, "share config must have default value."
            value = argspec.defaults[i - num_requires]

        elif i >= num_requires:
            value = argspec.defaults[i - num_requires]

        else:
            value = None

        schame[name] = value
        schame["_kwargs"][name] = value

    return schame

def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (
        torch.nn.parallel.DataParallel,
        torch.nn.parallel.DistributedDataParallel,
    )

def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def get_activation(act: str, inpace: bool = True):
    """get activation"""
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act

    act = act.lower()

    if act == "silu" or act == "swish":
        m = nn.SiLU()

    elif act == "relu":
        m = nn.ReLU()

    elif act == "leaky_relu":
        m = nn.LeakyReLU()

    elif act == "silu":
        m = nn.SiLU()

    elif act == "gelu":
        m = nn.GELU()

    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError("")

    if hasattr(m, "inplace"):
        m.inplace = inpace

    return m