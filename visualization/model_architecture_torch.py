import os
import sys
import torch
from torchinfo import summary

# 添加父目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_se import UNet, SELayer

def visualize_models(save_path):
    # 创建模型实例
    unet = UNet(n_channels=3, n_classes=5, use_se=False)
    unet_se = UNet(n_channels=3, n_classes=5, use_se=True)
    
    # 创建示例输入大小
    input_size = (1, 3, 224, 224)
    
    # 生成模型总结
    with open(f"{save_path}/unet_architecture.txt", 'w') as f:
        f.write("Standard U-Net Architecture\n")
        f.write("=" * 50 + "\n")
        f.write(str(summary(unet, input_size=input_size, verbose=0)))
    
    with open(f"{save_path}/unet_se_architecture.txt", 'w') as f:
        f.write("U-Net with SE Module Architecture\n")
        f.write("=" * 50 + "\n")
        f.write(str(summary(unet_se, input_size=input_size, verbose=0)))

if __name__ == '__main__':
    vis_dir = '/Users/very/源代码/Thesis_data/visualization'
    os.makedirs(vis_dir, exist_ok=True)
    visualize_models(vis_dir)