import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

def monte_carlo_predict(model, image, n_samples=30):
    """进行Monte Carlo预测"""
    predictions = []
    model.enable_dropout()
    with torch.no_grad():
        for _ in range(n_samples):
            pred = torch.sigmoid(model(image))
            predictions.append(pred.cpu().numpy())
    return np.stack(predictions)

def calculate_uncertainty(predictions):
    """计算预测的不确定性（熵）"""
    mean_pred = np.mean(predictions, axis=0)
    entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-7), axis=1)
    return entropy

def ks_test(pred_dist, true_dist):
    """进行KS检验"""
    statistic, p_value = stats.ks_2samp(pred_dist.flatten(), true_dist.flatten())
    return statistic, p_value

def plot_segmentation_results(image, true_mask, pred_mask, uncertainty, save_path):
    """绘制分割结果和不确定性图"""
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(141)
    plt.imshow(image.transpose(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    
    # 真实标注
    plt.subplot(142)
    plt.imshow(true_mask.sum(axis=0), cmap='nipy_spectral')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # 预测结果
    plt.subplot(143)
    plt.imshow(pred_mask.sum(axis=0), cmap='nipy_spectral')
    plt.title('Prediction')
    plt.axis('off')
    
    # 不确定性热图
    plt.subplot(144)
    plt.imshow(uncertainty, cmap='hot')
    plt.colorbar()
    plt.title('Uncertainty')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_segmentation(model, dataloader, save_dir):
    """评估分割结果"""
    device = next(model.parameters()).device
    results = {
        'dice': [], 'iou': [],
        'precision': [], 'recall': [], 'f1': [],
        'ks_stats': [], 'ks_pvals': []
    }
    
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Monte Carlo预测
        predictions = monte_carlo_predict(model, images)
        mean_pred = np.mean(predictions, axis=0)
        pred_masks = (mean_pred > 0.5).astype(np.float32)
        
        # 计算不确定性
        uncertainty = calculate_uncertainty(predictions)
        
        # 计算评估指标
        for j in range(images.size(0)):
            # Dice和IoU
            dice = dice_coefficient(masks[j].cpu().numpy(), pred_masks[j])
            iou = iou_score(masks[j].cpu().numpy(), pred_masks[j])
            
            # 精确率、召回率、F1
            prec, rec, f1, _ = precision_recall_fscore_support(
                masks[j].cpu().numpy().flatten(),
                pred_masks[j].flatten(),
                average='weighted'
            )
            
            # KS检验
            ks_stat, ks_pval = ks_test(pred_masks[j], masks[j].cpu().numpy())
            
            results['dice'].append(dice)
            results['iou'].append(iou)
            results['precision'].append(prec)
            results['recall'].append(rec)
            results['f1'].append(f1)
            results['ks_stats'].append(ks_stat)
            results['ks_pvals'].append(ks_pval)
            
            # 保存可视化结果
            if i * dataloader.batch_size + j < 10:  # 只保存前10个样本的可视化结果
                plot_segmentation_results(
                    images[j].cpu().numpy(),
                    masks[j].cpu().numpy(),
                    pred_masks[j],
                    uncertainty[j],
                    os.path.join(save_dir, f'sample_{i}_{j}.png')
                )
    
    return results