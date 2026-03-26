import numpy as np
from scipy.ndimage import gaussian_filter

def draw_heatmap(attention_weights, sampling_locations, image_shape, sigma=15):
    """
    根据给定的注意力权重和采样位置绘制热力图。
    
    参数:
    - attention_weights: 形状为 (bs, Len_q, num_heads=8, num_levels=4, num_points=4) 的张量。
    - sampling_locations: 形状为 (bs, Len_q, num_heads=8, num_levels=4, num_points=4, cxcy=2) 的张量。
    - image_shape: 原始图像的形状 (height, width)。
    - sigma: 高斯模糊的sigma值, 越大点越弥散。
    
    返回:
    - heatmap_out: 归一化后的热力图数组, 形状为 (bs, height, width)。
    """
    img_height, img_width = image_shape
    attention_weights = attention_weights.cpu().detach().numpy()
    sampling_locations = sampling_locations.cpu().detach().numpy()
    # 对一个批次的可变形注意力的多个头求均值
    # 获取一个批次的可变形注意力值
    bs, Len_q, num_heads, num_levels, num_points = attention_weights.shape
    
    heatmap_list = []
    # 遍历一个批次中的每个查询token
    for i in range(bs):
        # 初始化热力图数组, 形状为 (height, width)
        heatmap = np.zeros((img_height, img_width))
        # 遍历一个批次中的每个查询token
        for j in range(Len_q):
            # 遍历该查询token的所有头
            for h in range(num_heads):  
                # 遍历该查询token的每个层级
                for l in range(num_levels):
                    # 遍历该查询token的每个采样点
                    for p in range(num_points):
                        # 取用可变形注意力值, 将采样位置映射到图像坐标
                        # sampling_locations[i, j, h, l, p]取出第i,j,h,l,p处的采样位置cxcy归一化值, 再将其映射到图像坐标
                        x, y = sampling_locations[i, j, h, l, p] * np.array([img_width, img_height])
                        # 取出第i,j,h,l,p处的可变形注意力值
                        w = attention_weights[i, j, h, l, p].item()
                        # 防止x,y超出图像范围
                        x = min(max(int(x), 0), img_width - 1)
                        y = min(max(int(y), 0), img_height - 1)
                        # 把该采样的位置的注意力值w添加到热力图中
                        heatmap[y, x] += w
        heatmap = np.clip(heatmap, 0, None)
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # 归一化热力图
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        heatmap_list.append(heatmap)
        heatmap_out = np.stack(heatmap_list, axis=0)
    return heatmap_out
