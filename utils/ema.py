import math
from copy import deepcopy
import torch
import torch.nn as nn
# 包内部使用相对导入比较合适
from .util import register, de_parallel
__all__ = ["ModelEMA"]

# d = decay * (1 - exp(-updates / warmups))
# ema_w = ema_w_old * d + cur_w * (1-d)
# 初始化时, ema_w_old = cur_w = init_w
# 0 step后, cur_w更新
# d = 0.9999*(1-0.990) = 0.01, 1-d = 0.99, ema_w = 0.01*init_w + 0.99*cur_w
# 当达到极大步(从第1个epochs训练迭代n步, 到训练结束第m个epochs, 一共走过m*n步)
# 此时 m*n几乎无穷大, d = 0.9999 * (1-0) = 0.9999, ema_w = ema_w_old * 0.9999 + 0.0001 * cur_w
# 这说明后期的权重更新几乎是很小的, 起到了稳定训练的作用

@register()
class ModelEMA(object):
    """
    指数移动平均EMA, 用于模型参数的平滑更新, 以提高模型的稳定性和泛化能力.
        decay: 衰减率, 用于控制EMA的更新速度, 取值范围为(0, 1), 建议值为0.9999
        warmups: 预热步数, 用于控制EMA的初始更新速度, 默认值为1000, 通常不建议设置为0
        start: 开始更新的步数, 默认为0, 即从第0步开始更新EMA
        以上的decay, warmups, start是初始化设置1次, 不再改变
    """

    def __init__(
        self, model: nn.Module, decay: float = 0.9999, warmups: int = 1000, start: int = 0
    ):
        super().__init__()

        self.module = deepcopy(de_parallel(model)).eval()
        # if next(model.parameters()).device.type != 'cpu':
        #     self.module.half()  # FP16 EMA
        self.decay = decay
        self.warmups = warmups
        self.before_start = 0
        self.start = start
        self.updates = 0  # number of EMA updates
        if warmups == 0:
            self.decay_fn = lambda x: decay # 预热步数为0, 则衰减率为固定值
        else:
            self.decay_fn = lambda x: decay * (
                1 - math.exp(-x / warmups)
            )  # decay exponential ramp (to help early epochs)

        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        if self.before_start < self.start:
            self.before_start += 1
            return
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            msd = de_parallel(model).state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(self,):
        return dict(
            module=self.module.state_dict(),
            updates=self.updates,
            decay=self.decay,
            warmups=self.warmups,
            start=self.start,
        )

    def load_state_dict(self, state, strict=True):
        '''
        ema模型不应该用于微调, 其训练新数据集的时候应重新初始化
        resume继续训练, 加载updates, decay, warmups, start, 不支持断点后训练修改decay, warmups, start
        '''
        self.module.load_state_dict(state["module"], strict=strict)
        self.updates = state.get("updates", self.updates)
        self.decay = state.get("decay", self.decay)
        self.warmups = state.get("warmups", self.warmups)
        self.start = state.get("start", self.start)
        # 重建 decay_fn
        if self.warmups == 0:
            self.decay_fn = lambda x: self.decay
        else:
            self.decay_fn = lambda x: self.decay * (1 - math.exp(-x / self.warmups))

    def forwad(self,):
        raise RuntimeError("ema...")

    def extra_repr(self) -> str:
        return f"decay={self.decay}, warmups={self.warmups}"