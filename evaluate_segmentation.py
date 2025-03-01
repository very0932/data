import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from scipy.stats import entropy
from torch.utils.data import Dataset, DataLoader
from models.unet_se import UNet
from models.unet_se_mc import MCDropoutUNet

class SegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images = np.load(images_path)
        self.masks = np.load(masks_path)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 调整图像通道顺序从 [H, W, C] 到 [C, H, W]
        image = np.transpose(self.images[idx], (2, 0, 1))
        # 调整掩码通道顺序从 [H, W, C] 到 [C, H, W]
        mask = np.transpose(self.masks[idx], (2, 0, 1))
        
        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask)
        return image, mask

def load_model_and_data():
    """加载模型和验证数据"""
    save_dir = '/Users/very/源代码/Thesis_data/models/checkpoints'
    data_dir = '/Users/very/源代码/Thesis_data/processed_data/segmentation/processed'
    
    # 加载验证集数据
    val_dataset = SegmentationDataset(
        os.path.join(data_dir, 'val_images.npy'),
        os.path.join(data_dir, 'val_masks.npy')
    )
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    # 加载训练好的模型
    models = {}
    for model_name in ['UNet', 'UNetSE', 'MCDropoutUNet']:
        checkpoint_path = os.path.join(save_dir, f'{model_name}_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            # 修改这里：设置 weights_only=False
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            if model_name == 'MCDropoutUNet':
                model = MCDropoutUNet(n_channels=3, n_classes=5)
            else:
                model = UNet(n_channels=3, n_classes=5, use_se=(model_name=='UNetSE'))
            model.load_state_dict(checkpoint['model_state_dict'])
            models[model_name] = model
    
    return models, val_loader

def generate_segmentation_comparison(model, image, mask, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image = image.to(device)
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # 定义类别和颜色（RGB格式）
    class_names = ['Microaneurysms', 'Hemorrhages', 'Hard Exudates', 
                  'Soft Exudates', 'Optic Disc']
    colors = [
        [1, 0, 0],    # 红色
        [0, 1, 0],    # 绿色
        [1, 1, 0],    # 黄色
        [0, 1, 1],    # 青色
        [1, 0, 1]     # 品红
    ]
    
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(image)) > 0.5
    
    # 创建带有子图的大图
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.5], height_ratios=[1, 1])
    
    # 原始图像
    img_display = image.squeeze().cpu().permute(1, 2, 0)
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    
    # 显示原始图像
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img_display)
    ax_orig.set_title('Original Image')
    ax_orig.axis('off')
    
    # 显示真实标注和预测结果的叠加
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_gt.imshow(img_display)
    for i in range(5):
        mask_display = mask.squeeze()[i].cpu().numpy()
        if mask_display.max() > 0:  # 只显示有标注的区域
            mask_rgb = np.zeros((*mask_display.shape, 3))
            mask_rgb[mask_display > 0] = colors[i]
            ax_gt.imshow(mask_rgb, alpha=0.5)
    ax_gt.set_title('Ground Truth')
    ax_gt.axis('off')
    
    # 显示预测结果
    ax_pred = fig.add_subplot(gs[0, 2])
    ax_pred.imshow(img_display)
    for i in range(5):
        pred_display = pred.squeeze().cpu()[i].numpy()
        if pred_display.max() > 0:  # 只显示有预测的区域
            pred_rgb = np.zeros((*pred_display.shape, 3))
            pred_rgb[pred_display > 0] = colors[i]
            ax_pred.imshow(pred_rgb, alpha=0.5)
    ax_pred.set_title('Prediction')
    ax_pred.axis('off')
    
    # 显示每个类别的对比
    for i in range(5):
        if i < 3:
            ax = fig.add_subplot(gs[1, i])
        else:
            ax = fig.add_subplot(gs[1, i-3])
        ax.imshow(img_display, alpha=0.7)
        
        # 真实标注
        mask_display = mask.squeeze()[i].cpu().numpy()
        mask_contour = np.zeros((*mask_display.shape, 3))
        mask_contour[mask_display > 0] = colors[i]
        ax.imshow(mask_contour, alpha=0.3)
        
        # 预测结果（用虚线轮廓标记）
        pred_display = pred.squeeze().cpu()[i].numpy()
        from skimage import measure
        contours = measure.find_contours(pred_display, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], '--', 
                   color=colors[i], linewidth=2, alpha=0.8)
        
        ax.set_title(f'{class_names[i]}')
        ax.axis('off')
    
    # 添加图例
    ax_legend = fig.add_subplot(gs[:, -1])
    ax_legend.axis('off')
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.5, label=f'{name} (GT)')
        for color, name in zip(colors, class_names)
    ]
    legend_elements.extend([
        plt.Line2D([0], [0], color=color, linestyle='--', 
                  label=f'{name} (Pred)')
        for color, name in zip(colors, class_names)
    ])
    ax_legend.legend(handles=legend_elements, loc='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_uncertainty_map(model, image, save_path, n_samples=30):
    """生成不确定性热图"""
    # 设置matplotlib字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    model.train()  # 启用dropout
    predictions = []
    
    for _ in range(n_samples):
        with torch.no_grad():
            pred = torch.sigmoid(model(image))
            predictions.append(pred)
    
    # 计算预测的平均值和标准差
    predictions = torch.stack(predictions)
    mean_pred = torch.mean(predictions, dim=0)
    uncertainty = entropy(mean_pred.cpu().numpy(), axis=1)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(image.squeeze().permute(1,2,0))
    plt.title('Original Image')
    
    plt.subplot(122)
    plt.imshow(uncertainty[0], cmap='hot')
    plt.colorbar()
    plt.title('Uncertainty Heatmap')
    
    plt.savefig(save_path)
    plt.close()

def main():
    # 创建结果目录
    results_dir = '/Users/very/源代码/Thesis_data/visualization/segmentation_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载模型和数据
    models, val_loader = load_model_and_data()
    
    # 获取一个样本进行可视化
    image, mask = next(iter(val_loader))
    
    # 为每个模型生成结果
    for model_name, model in models.items():
        print(f"\n评估 {model_name} 模型:")
        
        # 生成分割对比图
        save_path = os.path.join(results_dir, f'{model_name}_segmentation.png')
        generate_segmentation_comparison(model, image, mask, save_path)
        
        # 如果是MC-Dropout模型，生成不确定性热图
        if model_name == 'MCDropoutUNet':
            save_path = os.path.join(results_dir, f'{model_name}_uncertainty.png')
            generate_uncertainty_map(model, image, save_path)

if __name__ == '__main__':
    main()