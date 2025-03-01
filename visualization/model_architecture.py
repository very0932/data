import os
import sys
import torch
import matplotlib.pyplot as plt

# 添加父目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_se import UNet, SELayer

def visualize_unet(save_path):
    """可视化U-Net架构"""
    plt.figure(figsize=(15, 10))
    plt.title('U-Net Architecture')
    
    # 设置节点位置
    nodes = {
        'input': (0.1, 0.5, 'Input Image\n(3×H×W)'),
        'conv1': (0.2, 0.5, 'Conv Block 1\n(64×H×W)'),
        'pool1': (0.3, 0.5, 'MaxPool\n(64×H/2×W/2)'),
        'conv2': (0.4, 0.4, 'Conv Block 2\n(128×H/2×W/2)'),
        'pool2': (0.5, 0.4, 'MaxPool\n(128×H/4×W/4)'),
        'conv3': (0.6, 0.3, 'Conv Block 3\n(256×H/4×W/4)'),
        'pool3': (0.7, 0.3, 'MaxPool\n(256×H/8×W/8)'),
        'conv4': (0.8, 0.2, 'Conv Block 4\n(512×H/8×W/8)'),
        'pool4': (0.9, 0.2, 'MaxPool\n(512×H/16×W/16)'),
        'conv5': (0.9, 0.1, 'Conv Block 5\n(1024×H/16×W/16)'),
        'up4': (0.8, 0.3, 'UpConv 4\n(512×H/8×W/8)'),
        'dconv4': (0.7, 0.4, 'Conv Block 6\n(512×H/8×W/8)'),
        'up3': (0.6, 0.5, 'UpConv 3\n(256×H/4×W/4)'),
        'dconv3': (0.5, 0.6, 'Conv Block 7\n(256×H/4×W/4)'),
        'up2': (0.4, 0.7, 'UpConv 2\n(128×H/2×W/2)'),
        'dconv2': (0.3, 0.8, 'Conv Block 8\n(128×H/2×W/2)'),
        'up1': (0.2, 0.9, 'UpConv 1\n(64×H×W)'),
        'dconv1': (0.1, 0.9, 'Conv Block 9\n(64×H×W)'),
        'output': (0.05, 0.9, 'Output\n(C×H×W)')
    }
    
    # 绘制节点
    for name, (x, y, label) in nodes.items():
        plt.plot(x, y, 'o', markersize=10)
        plt.text(x, y-0.05, label, ha='center')
    
    # 绘制连接
    edges = [
        ('input', 'conv1'), ('conv1', 'pool1'),
        ('pool1', 'conv2'), ('conv2', 'pool2'),
        ('pool2', 'conv3'), ('conv3', 'pool3'),
        ('pool3', 'conv4'), ('conv4', 'pool4'),
        ('pool4', 'conv5'), ('conv5', 'up4'),
        ('up4', 'dconv4'), ('conv4', 'dconv4'),
        ('dconv4', 'up3'), ('conv3', 'dconv3'),
        ('up3', 'dconv3'), ('dconv3', 'up2'),
        ('conv2', 'dconv2'), ('up2', 'dconv2'),
        ('dconv2', 'up1'), ('conv1', 'dconv1'),
        ('up1', 'dconv1'), ('dconv1', 'output')
    ]
    
    for start, end in edges:
        x1, y1 = nodes[start][0], nodes[start][1]
        x2, y2 = nodes[end][0], nodes[end][1]
        plt.plot([x1, x2], [y1, y2], '-k')
    
    plt.axis('off')
    plt.savefig(os.path.join(save_path, 'unet_architecture.png'), bbox_inches='tight', dpi=300)
    plt.close()

def visualize_se(save_path):
    """可视化SE模块架构"""
    plt.figure(figsize=(10, 4))
    plt.title('Squeeze-and-Excitation Module')
    
    # 定义节点
    nodes = [
        (0.1, 0.5, 'Input\nC×H×W'),
        (0.3, 0.5, 'Global Pooling\nC×1×1'),
        (0.5, 0.5, 'FC + ReLU\nC/r'),
        (0.7, 0.5, 'FC + Sigmoid\nC'),
        (0.9, 0.5, 'Scale\nC×H×W')
    ]
    
    # 绘制节点和连接
    for i, (x, y, label) in enumerate(nodes):
        plt.plot(x, y, 'o', markersize=10)
        plt.text(x, y-0.1, label, ha='center')
        if i < len(nodes)-1:
            plt.arrow(x+0.02, y, 0.16, 0, head_width=0.02, head_length=0.02, fc='k', ec='k')
    
    plt.axis('off')
    plt.savefig(os.path.join(save_path, 'se_module.png'), bbox_inches='tight', dpi=300)
    plt.close()

def visualize_unet_se(save_path):
    """可视化U-Net+SE架构"""
    plt.figure(figsize=(15, 10))
    plt.title('U-Net with SE Module')
    
    # 首先绘制基本的U-Net结构
    visualize_unet(save_path)
    
    # 添加SE模块
    se_positions = [
        (0.2, 0.55), (0.4, 0.45), (0.6, 0.35),
        (0.8, 0.25), (0.7, 0.45), (0.5, 0.65),
        (0.3, 0.85), (0.1, 0.95)
    ]
    
    for x, y in se_positions:
        plt.plot(x, y, 's', markersize=8, color='red')
        plt.text(x, y-0.05, 'SE', color='red', ha='center')
    
    plt.savefig(os.path.join(save_path, 'unet_se_architecture.png'), bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    vis_dir = '/Users/very/源代码/Thesis_data/visualization'
    os.makedirs(vis_dir, exist_ok=True)
    
    visualize_unet(vis_dir)
    visualize_se(vis_dir)
    visualize_unet_se(vis_dir)