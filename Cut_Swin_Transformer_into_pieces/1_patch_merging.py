import torch
import torch.nn as nn
import math
import numpy as np

 
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
 
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
 
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
 
    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
 
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
 
        x = x.view(B, H, W, C)
        print('--------------------------')
        print(x)
        print('原始图像4D维度：',x.shape)
 
        # 在行和列方向上间隔1选取元素
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        print('--------------------------')
        print(x0)
        print('切分图像4D维度：',x0.shape)
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        print('--------------------------')
        print(x1)
        print('切分图像4D维度：',x1.shape)
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        print('--------------------------')
        print(x2)
        print('切分图像4D维度：',x2.shape)
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        print('--------------------------')
        print(x3)
        print('切分图像4D维度：',x3.shape)
 
        # 拼接到一起作为一整个张量
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        print('--------------------------')
        print(x)
        print('拼接整个张量后：',x.shape)
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        print('--------------------------')
        print(x)
        print('合并行和列后：',x.shape)
 
        x = self.norm(x)           # 归一化操作
        print('--------------------------')
        print(x)
        print('归一化操作后:', x.shape)
        x = self.reduction(x)      # 降维，通道降低2倍
        print('--------------------------')
        print(x)
        print('通道降低2倍后:', x.shape)
 
        return x
if __name__ == "__main__":
 
    x = np.array([[0, 2, 0, 2],[ 1, 3, 1, 3 ],[ 0, 2, 0, 2 ],[ 1, 3, 1, 3 ]])
    x = torch.from_numpy(x)
    x = x.view(1, 4*4, 1)
    x=x.to(torch.float32)
    model = PatchMerging(1)
    print('--------------------------')
    print(x)
    print('原始图像3D维度：', x.shape)
    y = model(x)
 
 