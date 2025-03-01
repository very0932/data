import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from models.unet_se import UNet
from models.unet_se_mc import MCDropoutUNet
from models.unet_se_aspp import UNetSEASPP
from losses.combined_loss import CombinedLoss, DiceLoss, BoundaryLoss

# 修改字体设置，使用系统支持的中文字体
plt.rcParams['font.family'] = ['Arial', 'SimHei', 'STSong', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

def sensitivity(y_true, y_pred):
    """计算敏感性（召回率）"""
    smooth = 1e-7
    true_positives = np.sum(y_true * y_pred)
    possible_positives = np.sum(y_true)
    return (true_positives + smooth) / (possible_positives + smooth)

def specificity(y_true, y_pred):
    """计算特异性"""
    smooth = 1e-7
    true_negatives = np.sum((1 - y_true) * (1 - y_pred))
    possible_negatives = np.sum(1 - y_true)
    return (true_negatives + smooth) / (possible_negatives + smooth)

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
        'train_iou': [], 'val_iou': [],
        'train_sensitivity': [], 'val_sensitivity': [],
        'train_specificity': [], 'val_specificity': []
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
        train_sensitivity = 0
        train_specificity = 0
        
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
                train_sensitivity += sensitivity(targets.cpu().numpy(), pred.cpu().numpy())
                train_specificity += specificity(targets.cpu().numpy(), pred.cpu().numpy())
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        val_sensitivity = 0
        val_specificity = 0
        val_samples = 0
        
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
                
                # 对每个样本和每个类别分别计算指标
                batch_dice = 0
                batch_iou = 0
                batch_sensitivity = 0
                batch_specificity = 0
                valid_classes = 0
                
                for b in range(targets.shape[0]):
                    for c in range(targets.shape[1]):
                        true = targets[b, c].cpu().numpy()
                        prediction = pred[b, c].cpu().numpy()
                        
                        # 只在有正样本的情况下计算指标
                        if np.sum(true) > 0 or np.sum(prediction) > 0:
                            batch_dice += dice_coefficient(true, prediction)
                            batch_iou += iou_score(true, prediction)
                            batch_sensitivity += sensitivity(true, prediction)
                            batch_specificity += specificity(true, prediction)
                            valid_classes += 1
                
                # 只有在有有效类别的情况下才累加
                if valid_classes > 0:
                    val_dice += batch_dice / valid_classes
                    val_iou += batch_iou / valid_classes
                    val_sensitivity += batch_sensitivity / valid_classes
                    val_specificity += batch_specificity / valid_classes
                    val_samples += 1
        
        # 计算平均值，避免除零错误
        val_loss = val_loss / len(val_loader)
        if val_samples > 0:
            val_dice = val_dice / val_samples
            val_iou = val_iou / val_samples
            val_sensitivity = val_sensitivity / val_samples
            val_specificity = val_specificity / val_samples
        val_dice = val_dice / (len(val_loader) * targets.shape[1])
        val_iou = val_iou / (len(val_loader) * targets.shape[1])
        val_sensitivity = val_sensitivity / (len(val_loader) * targets.shape[1])
        val_specificity = val_specificity / (len(val_loader) * targets.shape[1])
        
        # 记录性能指标
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_sensitivity'].append(train_sensitivity)
        history['val_sensitivity'].append(val_sensitivity)
        history['train_specificity'].append(train_specificity)
        history['val_specificity'].append(val_specificity)
        
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
              f'IoU: {train_iou:.4f}, '
              f'Sensitivity: {train_sensitivity:.4f}, '
              f'Specificity: {train_specificity:.4f}')
        print(f'Val Loss: {val_loss:.4f}, '
              f'Dice: {val_dice:.4f}, '
              f'IoU: {val_iou:.4f}, '
              f'Sensitivity: {val_sensitivity:.4f}, '
              f'Specificity: {val_specificity:.4f}')
    
    # 保存模型和训练历史
    save_dir = '/Users/very/源代码/Thesis_data/models/optimized_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # 根据模型类型选择保存路径
    if isinstance(model, UNetSEASPP):
        save_path = os.path.join(save_dir, 'UNetSEASPP_checkpoint.pth')
    elif isinstance(model, MCDropoutUNet):
        save_path = os.path.join(save_dir, 'MCDropoutUNet_checkpoint.pth')
    elif isinstance(model, UNet):
        if hasattr(model, 'use_se') and model.use_se:
            save_path = os.path.join(save_dir, 'UNetSE_checkpoint.pth')
        else:
            save_path = os.path.join(save_dir, 'UNet_checkpoint.pth')
    
    # 保存模型状态和训练历史
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, save_path)
    
    return history, model
# 在文件末尾添加以下代码

def load_training_results(weights_only=False):
    """加载之前的训练结果"""
    save_dir = '/Users/very/源代码/Thesis_data/models/optimized_checkpoints'
    results = {}
    
    # 加载U-Net结果
    unet_path = os.path.join(save_dir, 'UNet_checkpoint.pth')
    if os.path.exists(unet_path):
        try:
            unet_data = torch.load(unet_path, weights_only=weights_only)
            results['unet'] = unet_data['history']
        except Exception as e:
            print(f"加载 UNet 模型时出错: {e}")
    
    # 加载U-Net+SE结果
    unet_se_path = os.path.join(save_dir, 'UNetSE_checkpoint.pth')
    if os.path.exists(unet_se_path):
        try:
            unet_se_data = torch.load(unet_se_path, weights_only=weights_only)
            results['unet_se'] = unet_se_data['history']
        except Exception as e:
            print(f"加载 UNetSE 模型时出错: {e}")
    
    # 加载U-Net+SE+ASPP结果
    unet_se_aspp_path = os.path.join(save_dir, 'UNetSEASPP_checkpoint.pth')
    if os.path.exists(unet_se_aspp_path):
        try:
            unet_se_aspp_data = torch.load(unet_se_aspp_path, weights_only=weights_only)
            results['unet_se_aspp'] = unet_se_aspp_data['history']
        except Exception as e:
            print(f"加载 UNetSEASPP 模型时出错: {e}")
    
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
        'sensitivity': [],
        'specificity': [],
        'f1': []
    }
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(data_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # 计算性能指标
            for j in range(len(images)):
                for c in range(masks.shape[1]):
                    true = masks[j, c].cpu().numpy()
                    pred = preds[j, c].cpu().numpy()
                    
                    # 放宽条件：即使真实掩码为空，也计算指标
                    # 这样可以评估模型对负样本的处理能力
                    dice = dice_coefficient(true, pred)
                    results['dice'].append(dice)
                    
                    iou = iou_score(true, pred)
                    results['iou'].append(iou)
                    
                    sens = sensitivity(true, pred)
                    results['sensitivity'].append(sens)
                    
                    spec = specificity(true, pred)
                    results['specificity'].append(spec)
                    
                    # F1分数
                    tp = np.sum(pred * true)
                    fp = np.sum(pred * (1 - true))
                    fn = np.sum((1 - pred) * true)
                    
                    precision = tp / (tp + fp + 1e-7)
                    recall = tp / (tp + fn + 1e-7)
                    f1 = 2 * precision * recall / (precision + recall + 1e-7)
                    
                    results['f1'].append(f1)
            
            # 保存可视化结果
            if save_dir and i < 5:  # 只保存前5个批次的结果
                for j in range(len(images)):
                    img = images[j].cpu().numpy().transpose(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min())  # 归一化到0-1
                    
                    # 创建图像
                    plt.figure(figsize=(16, 4))
                    
                    # 原始图像
                    # 在evaluate_segmentation函数中修改标题
                    plt.subplot(1, 4, 1)
                    plt.imshow(img)
                    plt.title('Original Image', fontsize=12, fontfamily='Arial')
                    plt.axis('off')
                    
                    # Ground Truth - 修复颜色范围问题
                    plt.subplot(1, 4, 2)
                    mask_vis = np.zeros((masks.shape[2], masks.shape[3], 3))
                    for c in range(masks.shape[1]):
                        # 确保颜色值在0-1范围内
                        if c == 0:
                            color = np.array([1.0, 0, 0])  # 红色
                        elif c == 1:
                            color = np.array([0, 1.0, 0])  # 绿色
                        else:
                            color = np.array([0, 0, 1.0])  # 蓝色
                        
                        # 使用布尔索引避免颜色值累加超出范围
                        mask_channel = masks[j, c].cpu().numpy()
                        mask_vis[mask_channel > 0.5] = color
                    
                    plt.imshow(mask_vis)
                    plt.title('真实标注', fontsize=12, fontfamily='Arial')
                    plt.axis('off')
                    
                    # 预测结果 - 修复颜色范围问题
                    plt.subplot(1, 4, 3)
                    pred_vis = np.zeros((preds.shape[2], preds.shape[3], 3))
                    for c in range(preds.shape[1]):
                        if c == 0:
                            color = np.array([1.0, 0, 0])  # 红色
                        elif c == 1:
                            color = np.array([0, 1.0, 0])  # 绿色
                        else:
                            color = np.array([0, 0, 1.0])  # 蓝色
                        
                        # 使用布尔索引避免颜色值累加超出范围
                        pred_channel = preds[j, c].cpu().numpy()
                        pred_vis[pred_channel > 0.5] = color
                    
                    plt.imshow(pred_vis)
                    plt.title('预测结果', fontsize=12, fontfamily='Arial')
                    plt.axis('off')
                    
                    # 简化版Grad-CAM可视化
                    plt.subplot(1, 4, 4)
                    # 使用最后一层特征图的平均值作为热图
                    with torch.enable_grad():
                        model.eval()
                        img_tensor = images[j:j+1].clone().requires_grad_(True)
                        output = model(img_tensor)
                        
                        # 获取最后一层特征图
                        heatmap = torch.abs(output[0]).sum(dim=0).cpu().detach().numpy()
                        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                        
                        # 将热图叠加到原始图像上
                        plt.imshow(img)
                        plt.imshow(heatmap, alpha=0.5, cmap='viridis')
                        plt.title('注意力图', fontsize=12, fontfamily='Arial')
                        plt.axis('off')
                    
                    plt.tight_layout()
                    # 修改保存文件名，避免中文
                    plt.savefig(os.path.join(save_dir, f'segmentation_result_{i}_{j}.png'), dpi=300)
                    plt.close()
    
    return results

def print_evaluation_results(results):
    """打印评估结果，并处理可能的空数组情况"""
    print("\n评估结果:")
    
    # 检查是否有有效的结果
    if not results['dice'] or len(results['dice']) == 0:
        print("警告: 没有找到有效的评估样本!")
        return
        
    # 安全计算平均值和标准差
    def safe_mean_std(values):
        if not values or len(values) == 0:
            return "N/A", "N/A"
        values = np.array(values)
        values = values[~np.isnan(values)]  # 移除 NaN 值
        if len(values) == 0:
            return "N/A", "N/A"
        return np.mean(values), np.std(values)
    
    # 计算并打印各项指标
    dice_mean, dice_std = safe_mean_std(results['dice'])
    iou_mean, iou_std = safe_mean_std(results['iou'])
    sens_mean, sens_std = safe_mean_std(results['sensitivity'])
    spec_mean, spec_std = safe_mean_std(results['specificity'])
    f1_mean, f1_std = safe_mean_std(results['f1'])
    
    if dice_mean != "N/A":
        print(f"平均Dice系数: {dice_mean:.4f} ± {dice_std:.4f}")
        print(f"平均IoU: {iou_mean:.4f} ± {iou_std:.4f}")
        print(f"平均敏感性: {sens_mean:.4f} ± {sens_std:.4f}")
        print(f"平均特异性: {spec_mean:.4f} ± {spec_std:.4f}")
        print(f"平均F1分数: {f1_mean:.4f} ± {f1_std:.4f}")
    else:
        print("无法计算评估指标，可能是测试集中没有足够的正样本。")

def plot_comparison_table(results_dict, save_path):
    """绘制模型比较表格"""
    models = list(results_dict.keys())
    metrics = ['dice', 'iou', 'sensitivity', 'specificity']
    
    # 准备数据
    data = np.zeros((len(models), len(metrics)))
    for i, model_name in enumerate(models):
        for j, metric in enumerate(metrics):
            metric_key = f'val_{metric}' if f'val_{metric}' in results_dict[model_name] else metric
            if metric_key in results_dict[model_name]:
                # 使用最后一个epoch的值
                data[i, j] = results_dict[model_name][metric_key][-1]
    
    # 创建DataFrame
    df = pd.DataFrame(data, index=models, columns=['Dice', 'IoU', '敏感性', '特异性'])
    
    # 绘制热图
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df, annot=True, fmt='.4f', cmap='YlGnBu', linewidths=.5, cbar_kws={'label': '性能指标值'})
    ax.set_title('表3-1: 不同模型分割性能比较', fontsize=14, fontfamily='Arial')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return df

def plot_training_history(history, save_path):
    """绘制训练历史并保存图表"""
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss History', fontsize=12, fontfamily='Arial')
    plt.xlabel('Epoch', fontsize=10, fontfamily='Arial')
    plt.ylabel('Loss', fontsize=10, fontfamily='Arial')
    plt.legend()
    
    # Dice系数曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_dice'], label='Train')
    plt.plot(history['val_dice'], label='Validation')
    plt.title('Dice Coefficient History', fontsize=12, fontfamily='Arial')
    plt.xlabel('Epoch', fontsize=10, fontfamily='Arial')
    plt.ylabel('Dice', fontsize=10, fontfamily='Arial')
    plt.legend()
    
    # IoU分数曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['train_iou'], label='Train')
    plt.plot(history['val_iou'], label='Validation')
    plt.title('IoU Score History', fontsize=12, fontfamily='Arial')
    plt.xlabel('Epoch', fontsize=10, fontfamily='Arial')
    plt.ylabel('IoU', fontsize=10, fontfamily='Arial')
    plt.legend()
    
    # 敏感性曲线
    plt.subplot(2, 2, 4)
    plt.plot(history['train_sensitivity'], label='Train')
    plt.plot(history['val_sensitivity'], label='Validation')
    plt.title('Sensitivity History', fontsize=12, fontfamily='Arial')
    plt.xlabel('Epoch', fontsize=10, fontfamily='Arial')
    plt.ylabel('Sensitivity', fontsize=10, fontfamily='Arial')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    # 设置输出目录
    output_dir = '/Users/very/源代码/Thesis_data/output/optimized_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置数据路径
    data_dir = '/Users/very/源代码/Thesis_data/processed_data/segmentation/processed'
    
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
    
    # 训练U-Net+SE+ASPP模型
    print("开始训练 U-Net+SE+ASPP 模型...")
    model = UNetSEASPP(n_channels=3, n_classes=5)
    
    # 训练U-Net+SE+ASPP模型
    print("开始训练 U-Net+SE+ASPP 模型...")
    model = UNetSEASPP(n_channels=3, n_classes=5)
    
    # 使用更合适的类别权重
    # 根据类别频率调整，给予罕见类别更高权重
    class_weights = [3.0, 2.5, 2.5, 2.0, 1.0]  # 增加稀有类别的权重
    criterion = CombinedLoss(dice_weight=0.8, boundary_weight=0.2, class_weights=class_weights)
    
    # 使用更低的初始学习率和更强的权重衰减
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    # 使用更耐心的学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.7, min_lr=1e-6
    )
    
    # 训练模型
    history, trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30
    )
    
    # 绘制训练历史
    history_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, history_path)
    print(f"训练历史已保存到: {history_path}")
    
    # 评估模型
    print("评估模型性能...")
    eval_dir = os.path.join(output_dir, 'evaluation_results')
    results = evaluate_segmentation(trained_model, val_loader, eval_dir)
    print_evaluation_results(results)
    
    # 加载之前训练的模型结果进行比较
    print("加载之前的训练结果进行比较...")
    try:
        # 尝试加载旧版本的模型结果
        old_results = load_training_results(weights_only=False)
    except:
        try:
            # 如果失败，尝试使用weights_only=True
            old_results = load_training_results(weights_only=True)
        except Exception as e:
            print(f"加载旧模型结果失败: {e}")
            old_results = {}
    
    # 添加当前模型的结果
    all_results = old_results.copy()
    all_results['unet_se_aspp'] = history
    
    # 生成比较表格
    if len(all_results) > 1:  # 至少有两个模型可以比较
        table_path = os.path.join(output_dir, 'model_comparison_table.png')
        df = plot_comparison_table(all_results, table_path)
        print(f"模型比较表已保存到: {table_path}")
        print("\n模型性能比较:")
        print(df)
    else:
        print("没有足够的模型结果进行比较。")
    
    print("\n优化完成! 所有结果已保存到:", output_dir)