import os
import torch
import matplotlib.pyplot as plt
import numpy as np  # 添加这行

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

if __name__ == '__main__':
    results = load_training_results()
    
    if results:
        print("\n=== 详细训练结果分析 ===")
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
        
            print(f"\n最终性能:")
            print(f"训练损失: {history['train_loss'][-1]:.4f}")
            print(f"验证损失: {history['val_loss'][-1]:.4f}")
            print(f"训练Dice: {history['train_dice'][-1]:.4f}")
            print(f"验证Dice: {history['val_dice'][-1]:.4f}")
            print(f"训练IoU: {history['train_iou'][-1]:.4f}")
            print(f"验证IoU: {history['val_iou'][-1]:.4f}")
            
            # 计算性能改进
            print(f"\n性能改进:")
            print(f"损失降低: {history['train_loss'][0] - history['train_loss'][-1]:.4f}")
            print(f"Dice提升: {history['train_dice'][-1] - history['train_dice'][0]:.4f}")
            print(f"IoU提升: {history['train_iou'][-1] - history['train_iou'][0]:.4f}")
            
        
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