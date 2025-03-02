import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

# 创建输出目录
output_dir = '/Users/very/源代码/Thesis_data/visualization/output'
os.makedirs(output_dir, exist_ok=True)

# 创建一个绘制U-Net架构的函数
def draw_unet_architecture():
    # 创建更大的图形以容纳左侧图例
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 调整绘图区域，为左侧图例留出空间
    ax.set_xlim(10, 110)
    ax.set_ylim(-10, 100)
    ax.axis('off')
    
    # 设置图表样式
    plt.title('U-Net Architecture', fontsize=20, fontweight='bold', pad=20)
    
    # 定义更柔和的颜色
    colors = {
        'input': '#B3E0FF',  # 淡蓝色
        'conv': '#FFD6A5',   # 淡橙色
        'maxpool': '#CAFFB9', # 淡绿色
        'upconv': '#D8BFD8',  # 淡紫色
        'bottom': '#D3D3D3',  # 淡灰色
        'output': '#FFFACD',  # 淡黄色
        'skip': '#E6E6E6'     # 浅灰色
    }
    
    # 绘制输入层
    draw_block(ax, 45, 90, 10, 6, 'Input Image\n(3xHxW)', colors['input'], 'ellipse')
    
    # 绘制收缩路径 (Contracting Path)
    # 第一层
    draw_block(ax, 45, 80, 10, 6, 'Conv Block 1\n(64xHxW)', colors['conv'])
    draw_arrow(ax, 45, 87, 45, 83)
    
    draw_block(ax, 45, 70, 10, 6, 'MaxPool\n(64xH/2xW/2)', colors['maxpool'])
    draw_arrow(ax, 45, 77, 45, 73)
    
    # 第二层
    draw_block(ax, 45, 60, 10, 6, 'Conv Block 2\n(128xH/2xW/2)', colors['conv'])
    draw_arrow(ax, 45, 67, 45, 63)
    
    draw_block(ax, 45, 50, 10, 6, 'MaxPool\n(128xH/4xW/4)', colors['maxpool'])
    draw_arrow(ax, 45, 57, 45, 53)
    
    # 第三层
    draw_block(ax, 45, 40, 10, 6, 'Conv Block 3\n(256xH/4xW/4)', colors['conv'])
    draw_arrow(ax, 45, 47, 45, 43)
    
    draw_block(ax, 45, 30, 10, 6, 'MaxPool\n(256xH/8xW/8)', colors['maxpool'])
    draw_arrow(ax, 45, 37, 45, 33)
    
    # 第四层
    draw_block(ax, 45, 20, 10, 6, 'Conv Block 4\n(512xH/8xW/8)', colors['conv'])
    draw_arrow(ax, 45, 27, 45, 23)
    
    draw_block(ax, 45, 10, 10, 6, 'MaxPool\n(512xH/16xW/16)', colors['maxpool'])
    draw_arrow(ax, 45, 17, 45, 13)
    
    # 底部 - 进一步降低y坐标，增大间隙
    draw_block(ax, 45, -5, 12, 6, 'Conv Block 5\n(1024xH/16xW/16)', colors['bottom'])
    # 修复箭头连接，保持间隙一致
    draw_arrow(ax, 45, 7, 45, -2)
    
    # 绘制扩展路径 (Expansive Path)
    # 第四层
    draw_block(ax, 65, 20, 10, 6, 'UpConv 4\n(512xH/8xW/8)', colors['upconv'])
    # 修复箭头连接
    draw_arrow(ax, 51, -5, 65, 17)
    
    draw_block(ax, 75, 20, 10, 6, 'Conv Block 6\n(512xH/8xW/8)', colors['conv'])
    draw_arrow(ax, 70, 20, 75, 20)
    # 跳跃连接
    draw_skip_connection(ax, 50, 20, 70, 20)
    
    # 第三层
    draw_block(ax, 65, 40, 10, 6, 'UpConv 3\n(256xH/4xW/4)', colors['upconv'])
    draw_arrow(ax, 80, 20, 65, 37)
    
    draw_block(ax, 75, 40, 10, 6, 'Conv Block 7\n(256xH/4xW/4)', colors['conv'])
    draw_arrow(ax, 70, 40, 75, 40)
    # 跳跃连接
    draw_skip_connection(ax, 50, 40, 70, 40)
    
    # 第二层
    draw_block(ax, 65, 60, 10, 6, 'UpConv 2\n(128xH/2xW/2)', colors['upconv'])
    draw_arrow(ax, 80, 40, 65, 57)
    
    draw_block(ax, 75, 60, 10, 6, 'Conv Block 8\n(128xH/2xW/2)', colors['conv'])
    draw_arrow(ax, 70, 60, 75, 60)
    # 跳跃连接
    draw_skip_connection(ax, 50, 60, 70, 60)
    
    # 第一层
    draw_block(ax, 65, 80, 10, 6, 'UpConv 1\n(64xHxW)', colors['upconv'])
    draw_arrow(ax, 80, 60, 65, 77)
    
    draw_block(ax, 75, 80, 10, 6, 'Conv Block 9\n(64xHxW)', colors['conv'])
    draw_arrow(ax, 70, 80, 75, 80)
    # 跳跃连接
    draw_skip_connection(ax, 50, 80, 70, 80)
    
    # 输出层
    draw_block(ax, 75, 90, 10, 6, 'Output\n(CxHxW)', colors['output'], 'ellipse')
    draw_arrow(ax, 75, 83, 75, 87)
    
    # 添加图例
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=colors['input'], label='Input'),
        Rectangle((0, 0), 1, 1, facecolor=colors['conv'], label='Conv Block'),
        Rectangle((0, 0), 1, 1, facecolor=colors['maxpool'], label='Max Pooling'),
        Rectangle((0, 0), 1, 1, facecolor=colors['upconv'], label='Up-Convolution'),
        Rectangle((0, 0), 1, 1, facecolor=colors['bottom'], label='Bottleneck'),
        Rectangle((0, 0), 1, 1, facecolor=colors['output'], label='Output'),
        FancyArrowPatch((0, 0), (1, 0), linestyle='dashed', color='#808080', label='Skip Connection (特征复用)')
    ]
    ax.legend(handles=legend_elements, 
              loc='center left',
              bbox_to_anchor=(-0.1, 0.5),
              fontsize=12)
    
    # 保存图像
    output_path = os.path.join(output_dir, 'U-Net_Architecture_matplotlib.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"U-Net架构图已保存至: {output_path}")
    
    return fig

# 辅助函数：绘制方块
def draw_block(ax, x, y, width, height, text, color, shape='rectangle'):
    if shape == 'ellipse':
        ellipse = plt.matplotlib.patches.Ellipse((x, y), width, height, 
                                                fill=True, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(ellipse)
    else:
        rect = plt.matplotlib.patches.Rectangle((x - width/2, y - height/2), width, height, 
                                               fill=True, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
    
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

# 辅助函数：绘制箭头
def draw_arrow(ax, x1, y1, x2, y2, style='solid', color='black'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           arrowstyle='->', 
                           linewidth=1.5,
                           linestyle=style,
                           color=color,
                           connectionstyle='arc3,rad=0.0',
                           shrinkA=5,  # 起点缩短
                           shrinkB=5)  # 终点缩短
    ax.add_patch(arrow)

# 修改跳跃连接的绘制方式
def draw_skip_connection(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->',
                           linewidth=1.5,
                           linestyle='dashed',
                           color='#808080',
                           connectionstyle='arc3,rad=0.0',
                           shrinkA=8,  # 增加缩短距离
                           shrinkB=8)
    ax.add_patch(arrow)

# 执行绘图
if __name__ == "__main__":
    try:
        fig = draw_unet_architecture()
        plt.close(fig)
    except Exception as e:
        print(f"绘制U-Net架构图时出错: {e}")