import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_current_results():
    """加载本次训练结果"""
    save_dir = '/Users/very/源代码/Thesis_data/models/checkpoints'
    results = {}
    
    # 加载最新的训练结果
    for model_name in ['UNet', 'UNetSE', 'MCDropoutUNet']:
        checkpoint_path = os.path.join(save_dir, f'{model_name}_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            results[model_name.lower()] = checkpoint['history']
    
    return results

def display_results(results):
    """显示训练结果"""
    if not results:
        print("未找到训练结果")
        return
    
    print("\n=== 本次训练结果分析 ===")
    for model_name, history in results.items():
        print(f"\n{model_name.upper()} 模型:")
        print("-" * 50)
        
        # 计算关键指标
        best_val_dice = max(history['val_dice'])
        best_val_iou = max(history['val_iou'])
        best_epoch = history['val_dice'].index(best_val_dice) + 1
        
        print(f"最佳验证Dice (Epoch {best_epoch}): {best_val_dice:.4f}")
        print(f"最佳验证IoU: {best_val_iou:.4f}")
        print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
        print(f"最终验证损失: {history['val_loss'][-1]:.4f}")
        
        # 计算收敛速度
        convergence_threshold = 0.95 * best_val_dice
        for epoch, dice in enumerate(history['val_dice']):
            if dice >= convergence_threshold:
                print(f"收敛轮次 (95% 最佳性能): {epoch + 1}")
                break
        
        # 计算性能提升
        print("\n性能提升分析:")
        print(f"损失降低: {history['train_loss'][0] - history['train_loss'][-1]:.4f}")
        print(f"Dice提升: {history['train_dice'][-1] - history['train_dice'][0]:.4f}")
        print(f"IoU提升: {history['train_iou'][-1] - history['train_iou'][0]:.4f}")
        
        # 计算稳定性
        val_dice_std = np.std(history['val_dice'][-10:])
        print(f"\n模型稳定性 (最后10轮验证Dice标准差): {val_dice_std:.4f}")
        
        print("-" * 50)

def visualize_results(results):
    """可视化训练结果"""
    if not results:
        return
    
    vis_dir = '/Users/very/源代码/Thesis_data/visualization/current_results'
    os.makedirs(vis_dir, exist_ok=True)
    
    # 设置字体样式
    plt.style.use('seaborn')
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 创建三个子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 损失曲线
    ax = axes[0]
    for model_name, history in results.items():
        ax.plot(history['train_loss'], label=f'{model_name} Train')
        ax.plot(history['val_loss'], label=f'{model_name} Val')
    ax.set_title('Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Dice系数曲线
    ax = axes[1]
    for model_name, history in results.items():
        ax.plot(history['train_dice'], label=f'{model_name} Train')
        ax.plot(history['val_dice'], label=f'{model_name} Val')
    ax.set_title('Dice Coefficient')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.legend()
    ax.grid(True)
    
    # IoU分数曲线
    ax = axes[2]
    for model_name, history in results.items():
        ax.plot(history['train_iou'], label=f'{model_name} Train')
        ax.plot(history['val_iou'], label=f'{model_name} Val')
    ax.set_title('IoU Score')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    results = load_current_results()
    display_results(results)
    visualize_results(results)  # 添加这行