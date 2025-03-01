import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.unet_se import UNet
from models.unet_se_mc import MCDropoutUNet  # 添加这行
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

def dice_coefficient(y_true, y_pred):
    """计算Dice系数"""
    smooth = 1e-7
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred):
    """计算IoU分数"""
    smooth = 1e-7
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)
    return iou

from models.unet_se_aspp import UNetSEASPP
from losses.combined_loss import CombinedLoss, DiceLoss, BoundaryLoss
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
#from pytorch_grad_cam import GradCAM
#from pytorch_grad_cam.utils.image import show_cam_on_image

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    """训练模型并记录性能指标"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 增加早停的严格性
    best_val_loss = float('inf')
    patience = 5  # 减少等待轮数
    patience_counter = 0
    best_model_state = None
    
    # 添加学习率预热
    warmup_epochs = 3
    warmup_factor = 0.1
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': []
    }
    
    for epoch in range(num_epochs):
        # 学习率预热
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = optimizer.param_groups[0]['lr'] * (
                    1 + epoch * (1 - warmup_factor) / warmup_epochs
                )
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 添加L2正则化
            l2_lambda = 0.001
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            
            loss = criterion(outputs, targets) + l2_lambda * l2_reg
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 计算指标
            with torch.no_grad():
                pred = torch.sigmoid(outputs) > 0.5
                train_loss += loss.item()
                train_dice += dice_coefficient(targets.cpu().numpy(), pred.cpu().numpy())
                train_iou += iou_score(targets.cpu().numpy(), pred.cpu().numpy())
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # 添加L2正则化到验证损失
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                
                loss = criterion(outputs, targets) + l2_lambda * l2_reg
                
                val_loss += loss.item()
                pred = torch.sigmoid(outputs) > 0.5
                
                # 对每个类别分别计算指标
                for c in range(targets.shape[1]):
                    val_dice += dice_coefficient(
                        targets[:, c].cpu().numpy(),
                        pred[:, c].cpu().numpy()
                    )
                    val_iou += iou_score(
                        targets[:, c].cpu().numpy(),
                        pred[:, c].cpu().numpy()
                    )
        
        # 计算平均值
        train_loss = train_loss / len(train_loader)
        train_dice = train_dice / len(train_loader)
        train_iou = train_iou / len(train_loader)
        
        val_loss = val_loss / len(val_loader)
        val_dice = val_dice / (len(val_loader) * targets.shape[1])
        val_iou = val_iou / (len(val_loader) * targets.shape[1])
        
        # 记录性能指标
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        
        # 更新学习率
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_state)
                break
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, '
              f'Dice: {train_dice:.4f}, '
              f'IoU: {train_iou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, '
              f'Dice: {val_dice:.4f}, '
              f'IoU: {val_iou:.4f}')
    
    # 保存模型和训练历史
    save_dir = '/Users/very/源代码/Thesis_data/models/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # 根据模型类型选择保存路径
    if isinstance(model, MCDropoutUNet):
        save_path = os.path.join(save_dir, 'MCDropoutUNet_checkpoint.pth')
    elif isinstance(model, UNet):
        if model.use_se:
            save_path = os.path.join(save_dir, 'UNetSE_checkpoint.pth')
        else:
            save_path = os.path.join(save_dir, 'UNet_checkpoint.pth')
    
    # 保存模型状态和训练历史
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, save_path)
    
    return history, model

def load_training_results():
    """加载之前的训练结果"""
    save_dir = '/Users/very/源代码/Thesis_data/models/checkpoints'
    results = {}
    
    # 加载U-Net结果
    unet_path = os.path.join(save_dir, 'UNet_checkpoint.pth')
    if os.path.exists(unet_path):
        unet_data = torch.load(unet_path)
        results['unet'] = unet_data['history']
    
    # 加载U-Net+SE结果
    unet_se_path = os.path.join(save_dir, 'UNetSE_checkpoint.pth')
    if os.path.exists(unet_se_path):
        unet_se_data = torch.load(unet_se_path)
        results['unet_se'] = unet_se_data['history']
    
    # 加载MC-Dropout结果
    mc_path = os.path.join(save_dir, 'MCDropoutUNet_checkpoint.pth')
    if os.path.exists(mc_path):
        mc_data = torch.load(mc_path)
        results['mc_dropout'] = mc_data['history']
    
    return results

def evaluate_segmentation(model, data_loader, save_dir=None):
    """评估分割模型性能并可选地保存可视化结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'ks_pvals': []
    }
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(data_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # 对于MC-Dropout模型，进行多次预测
            if isinstance(model, MCDropoutUNet):
                model.train()  # 启用dropout
                n_samples = 10
                all_preds = []
                
                for _ in range(n_samples):
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                    all_preds.append(probs)
                
                # 计算平均预测和不确定性
                mean_preds = torch.stack(all_preds).mean(0)
                std_preds = torch.stack(all_preds).std(0)
                preds = (mean_preds > 0.5).float()
                
                # 进行KS检验
                from scipy import stats
                ks_pvals = []
                for j in range(len(images)):
                    sample_preds = [p[j].cpu().numpy().flatten() for p in all_preds]
                    # 比较第一个样本与其他样本的分布
                    pvals = []
                    for k in range(1, len(sample_preds)):
                        _, pval = stats.ks_2samp(sample_preds[0], sample_preds[k])
                        pvals.append(pval)
                    ks_pvals.append(np.mean(pvals))
                
                results['ks_pvals'].extend(ks_pvals)
                
                # 保存不确定性可视化
                if save_dir and i < 5:  # 只保存前5个批次的结果
                    for j in range(len(images)):
                        img = images[j].cpu().numpy().transpose(1, 2, 0)
                        img = (img - img.min()) / (img.max() - img.min())  # 归一化到0-1
                        
                        # 保存原始图像
                        plt.figure(figsize=(15, 10))
                        plt.subplot(2, 3, 1)
                        plt.imshow(img)
                        plt.title('Original Image')
                        plt.axis('off')
                        
                        # 保存真实掩码
                        plt.subplot(2, 3, 2)
                        mask_vis = np.zeros((masks.shape[2], masks.shape[3], 3))
                        for c in range(masks.shape[1]):
                            color = np.array([1, 0, 0]) if c == 0 else np.array([0, 1, 0]) if c == 1 else np.array([0, 0, 1])
                            mask_vis += masks[j, c].cpu().numpy()[..., None] * color
                        plt.imshow(mask_vis)
                        plt.title('Ground Truth')
                        plt.axis('off')
                        
                        # 保存预测掩码
                        plt.subplot(2, 3, 3)
                        pred_vis = np.zeros((preds.shape[2], preds.shape[3], 3))
                        for c in range(preds.shape[1]):
                            color = np.array([1, 0, 0]) if c == 0 else np.array([0, 1, 0]) if c == 1 else np.array([0, 0, 1])
                            pred_vis += preds[j, c].cpu().numpy()[..., None] * color
                        plt.imshow(pred_vis)
                        plt.title('Prediction')
                        plt.axis('off')
                        
                        # 保存不确定性图
                        plt.subplot(2, 3, 4)
                        uncertainty = std_preds[j].mean(0).cpu().numpy()
                        plt.imshow(uncertainty, cmap='viridis')
                        plt.colorbar()
                        plt.title('Uncertainty (Std)')
                        plt.axis('off')
                        
                        # 保存
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, f'sample_{i}_{j}.png'))
                        plt.close()
            else:
                # 标准模型预测
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            
            # 计算性能指标
            for j in range(len(images)):
                for c in range(masks.shape[1]):
                    true = masks[j, c].cpu().numpy()
                    pred = preds[j, c].cpu().numpy()
                    
                    # 只有当真实掩码不为空时才计算指标
                    if np.sum(true) > 0:
                        # Dice系数
                        dice = dice_coefficient(true, pred)
                        results['dice'].append(dice)
                        
                        # IoU分数
                        iou = iou_score(true, pred)
                        results['iou'].append(iou)
                        
                        # 精确率、召回率和F1分数
                        tp = np.sum(pred * true)
                        fp = np.sum(pred * (1 - true))
                        fn = np.sum((1 - pred) * true)
                        
                        precision = tp / (tp + fp + 1e-7)
                        recall = tp / (tp + fn + 1e-7)
                        f1 = 2 * precision * recall / (precision + recall + 1e-7)
                        
                        results['precision'].append(precision)
                        results['recall'].append(recall)
                        results['f1'].append(f1)
    
    return results

def plot_training_history(history, save_path):
    """绘制训练历史并保存图表"""
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(131)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Dice系数曲线
    plt.subplot(132)
    plt.plot(history['train_dice'], label='Train')
    plt.plot(history['val_dice'], label='Validation')
    plt.title('Dice Coefficient History')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    # IoU分数曲线
    plt.subplot(133)
    plt.plot(history['train_iou'], label='Train')
    plt.plot(history['val_iou'], label='Validation')
    plt.title('IoU Score History')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    # 设置数据路径
    data_dir = '/Users/very/源代码/Thesis_data/processed_data/segmentation/processed'
    vis_dir = '/Users/very/源代码/Thesis_data/visualization'
    os.makedirs(vis_dir, exist_ok=True)
    
    # 加载数据集
    train_dataset = SegmentationDataset(
        os.path.join(data_dir, 'train_images.npy'),
        os.path.join(data_dir, 'train_masks.npy')
    )
    val_dataset = SegmentationDataset(
        os.path.join(data_dir, 'val_images.npy'),
        os.path.join(data_dir, 'val_masks.npy')
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # 训练标准U-Net
    unet = UNet(n_channels=3, n_classes=5, use_se=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    history_unet, trained_unet = train_model(
        unet, train_loader, val_loader,
        criterion, optimizer, scheduler, num_epochs=50
    )
    
    # 训练U-Net+SE
    unet_se = UNet(n_channels=3, n_classes=5, use_se=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(unet_se.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    history_unet_se, trained_unet_se = train_model(
        unet_se, train_loader, val_loader,
        criterion, optimizer, scheduler, num_epochs=50
    )
    
    # 调整训练参数
    unet = UNet(n_channels=3, n_classes=5, use_se=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.0005, weight_decay=0.001)  # 降低学习率，增加权重衰减
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6, verbose=True
    )
    
    history_unet, trained_unet = train_model(
        unet, train_loader, val_loader,
        criterion, optimizer, scheduler, num_epochs=50
    )
    
    # 训练U-Net+SE（使用相同的参数）
    unet_se = UNet(n_channels=3, n_classes=5, use_se=True)
    optimizer = torch.optim.Adam(unet_se.parameters(), lr=0.0005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6, verbose=True
    )
    
    history_unet_se, trained_unet_se = train_model(
        unet_se, train_loader, val_loader,
        criterion, optimizer, scheduler, num_epochs=50
    )
    
    # 创建MC-Dropout U-Net+SE模型
    model = MCDropoutUNet(n_channels=3, n_classes=5, dropout_rate=0.1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6, verbose=True
    )
    
    # 训练模型
    history, trained_model = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler, num_epochs=50
    )
    
    # 评估模型
    eval_dir = '/Users/very/源代码/Thesis_data/visualization/evaluation'
    results = evaluate_segmentation(trained_model, val_loader, eval_dir)
    
    # 输出评估结果
    print("\n评估结果:")
    print(f"平均Dice系数: {np.mean(results['dice']):.4f} ± {np.std(results['dice']):.4f}")
    print(f"平均IoU: {np.mean(results['iou']):.4f} ± {np.std(results['iou']):.4f}")
    print(f"平均精确率: {np.mean(results['precision']):.4f} ± {np.std(results['precision']):.4f}")
    print(f"平均召回率: {np.mean(results['recall']):.4f} ± {np.std(results['recall']):.4f}")
    print(f"平均F1分数: {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")
    print(f"KS检验通过率: {np.mean(np.array(results['ks_pvals']) > 0.05):.2%}")
    
    # 绘制训练历史对比图
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(131)
    plt.plot(history_unet['train_loss'], label='U-Net Train')
    plt.plot(history_unet['val_loss'], label='U-Net Val')
    plt.plot(history_unet_se['train_loss'], label='U-Net+SE Train')
    plt.plot(history_unet_se['val_loss'], label='U-Net+SE Val')
    plt.title('Loss Comparison')
    plt.legend()
    
    # Dice系数曲线
    plt.subplot(132)
    plt.plot(history_unet['train_dice'], label='U-Net Train')
    plt.plot(history_unet['val_dice'], label='U-Net Val')
    plt.plot(history_unet_se['train_dice'], label='U-Net+SE Train')
    plt.plot(history_unet_se['val_dice'], label='U-Net+SE Val')
    plt.title('Dice Coefficient Comparison')
    plt.legend()
    
    # IoU分数曲线
    plt.subplot(133)
    plt.plot(history_unet['train_iou'], label='U-Net Train')
    plt.plot(history_unet['val_iou'], label='U-Net Val')
    plt.plot(history_unet_se['train_iou'], label='U-Net+SE Train')
    plt.plot(history_unet_se['val_iou'], label='U-Net+SE Val')
    plt.title('IoU Score Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'performance_comparison.png'))
    plt.close()
    
    # 保存各自的训练历史
    plot_training_history(history_unet, os.path.join(vis_dir, 'unet_training_history.png'))
    plot_training_history(history_unet_se, os.path.join(vis_dir, 'unet_se_training_history.png'))
if __name__ == '__main__':
    # 加载之前的训练结果
    results = load_training_results()
    
    if results:
        print("\n=== 之前的训练结果 ===")
        for model_name, history in results.items():
            print(f"\n{model_name} 最终性能:")
            print(f"训练损失: {history['train_loss'][-1]:.4f}")
            print(f"验证损失: {history['val_loss'][-1]:.4f}")
            print(f"训练Dice: {history['train_dice'][-1]:.4f}")
            print(f"验证Dice: {history['val_dice'][-1]:.4f}")
            print(f"训练IoU: {history['train_iou'][-1]:.4f}")
            print(f"验证IoU: {history['val_iou'][-1]:.4f}")
        
        # 绘制训练历史对比图
        vis_dir = '/Users/very/源代码/Thesis_data/visualization'
        os.makedirs(vis_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(131)
        for model_name, history in results.items():
            plt.plot(history['train_loss'], label=f'{model_name} Train')
            plt.plot(history['val_loss'], label=f'{model_name} Val')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Dice系数曲线
        plt.subplot(132)
        for model_name, history in results.items():
            plt.plot(history['train_dice'], label=f'{model_name} Train')
            plt.plot(history['val_dice'], label=f'{model_name} Val')
        plt.title('Dice Coefficient History')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.legend()
        
        # IoU分数曲线
        plt.subplot(133)
        for model_name, history in results.items():
            plt.plot(history['train_iou'], label=f'{model_name} Train')
            plt.plot(history['val_iou'], label=f'{model_name} Val')
        plt.title('IoU Score History')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'previous_training_comparison.png'))
        plt.close()
    else:
        print("未找到之前的训练结果，需要重新训练模型。")