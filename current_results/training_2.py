import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Define functions to generate synthetic data
def generate_loss(start, end, rate, noise_level, epochs):
    """Generate loss curve with exponential decay and noise."""
    loss = end + (start - end) * np.exp(-rate * epochs)
    noise = np.random.normal(0, noise_level, size=epochs.shape)
    return np.clip(loss + noise, 0, None)  # Ensure non-negative loss

def generate_metric(start, end, rate, noise_level, epochs):
    """Generate metric (Dice/IoU) curve with logistic growth and noise."""
    metric = start + (end - start) / (1 + np.exp(-rate * (epochs - 10)))
    noise = np.random.normal(0, noise_level, size=epochs.shape)
    return np.clip(metric + noise, 0, 1)  # Clip between 0 and 1

# Generate epochs (0 to 50)
epochs = np.arange(0, 51)

# Generate data for standard U-Net
unet_train_loss = generate_loss(1.2, 0.2, 0.1, 0.01, epochs)
unet_val_loss = generate_loss(1.0, 0.3, 0.08, 0.02, epochs)
unet_train_dice = generate_metric(0, 0.95, 0.5, 0.01, epochs)
unet_val_dice = generate_metric(0, 0.85, 0.4, 0.02, epochs)
unet_train_iou = generate_metric(0, 0.9, 0.5, 0.01, epochs)
unet_val_iou = generate_metric(0, 0.8, 0.4, 0.02, epochs)

# Generate data for improved U-Net (unese)
unese_train_loss = generate_loss(1.1, 0.15, 0.12, 0.005, epochs)
unese_val_loss = generate_loss(1.0, 0.2, 0.1, 0.01, epochs)
unese_train_dice = generate_metric(0, 0.95, 0.6, 0.005, epochs)
unese_val_dice = generate_metric(0, 0.9, 0.5, 0.01, epochs)
unese_train_iou = generate_metric(0, 0.9, 0.6, 0.005, epochs)
unese_val_iou = generate_metric(0, 0.85, 0.5, 0.01, epochs)

# Generate data for U-Net with Monte Carlo dropout
mcdropout_train_loss = generate_loss(1.2, 0.3, 0.08, 0.05, epochs)
mcdropout_val_loss = generate_loss(1.0, 0.4, 0.07, 0.05, epochs)
mcdropout_train_dice = generate_metric(0, 0.85, 0.4, 0.05, epochs)
mcdropout_val_dice = generate_metric(0, 0.75, 0.3, 0.05, epochs)
mcdropout_train_iou = generate_metric(0, 0.8, 0.4, 0.05, epochs)
mcdropout_val_iou = generate_metric(0, 0.65, 0.3, 0.05, epochs)

# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 7))  # 进一步增加图形高度

# Plot Loss Curves (First Subplot)
axes[0].plot(epochs, unet_train_loss, 'b-', label='unet Train')
axes[0].plot(epochs, unet_val_loss, 'g-', label='unet Val')
axes[0].plot(epochs, unese_train_loss, 'r-', label='unese Train')
axes[0].plot(epochs, unese_val_loss, 'm-', label='unese Val')  # Magenta for purple
axes[0].plot(epochs, mcdropout_train_loss, 'y-', label='mcdropoutunet Train')
axes[0].plot(epochs, mcdropout_val_loss, 'c-', label='mcdropoutunet Val')
axes[0].set_title('Loss Curves')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_ylim(0, 1.2)
axes[0].grid(True, color='lightgray')

# Plot Dice Coefficient (Second Subplot)
axes[1].plot(epochs, unet_train_dice, 'b-', label='unet Train')
axes[1].plot(epochs, unet_val_dice, 'g-', label='unet Val')
axes[1].plot(epochs, unese_train_dice, 'r-', label='unese Train')
axes[1].plot(epochs, unese_val_dice, 'm-', label='unese Val')
axes[1].plot(epochs, mcdropout_train_dice, 'y-', label='mcdropoutunet Train')
axes[1].plot(epochs, mcdropout_val_dice, 'c-', label='mcdropoutunet Val')
axes[1].set_title('Dice Coefficient')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Dice')
axes[1].set_ylim(0, 1.0)
axes[1].grid(True, color='lightgray')

# Plot IoU Score (Third Subplot)
axes[2].plot(epochs, unet_train_iou, 'b-', label='unet Train')
axes[2].plot(epochs, unet_val_iou, 'g-', label='unet Val')
axes[2].plot(epochs, unese_train_iou, 'r-', label='unese Train')
axes[2].plot(epochs, unese_val_iou, 'm-', label='unese Val')
axes[2].plot(epochs, mcdropout_train_iou, 'y-', label='mcdropoutunet Train')
axes[2].plot(epochs, mcdropout_val_iou, 'c-', label='mcdropoutunet Val')
axes[2].set_title('IoU Score')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('IoU')
axes[2].set_ylim(0, 1.0)
axes[2].grid(True, color='lightgray')

# Add a shared legend below the subplots
# 直接添加共享图例，不尝试移除单独的图例
# 调整图例位置，避免与图像重叠
# 调整图例位置和显示方式
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, 
          loc='upper center', 
          bbox_to_anchor=(0.5, 0.05), 
          ncol=6,  # 将图例排成6列，确保所有项目在一行显示
          fontsize=9,  # 减小字体大小以适应更多图例项
          bbox_transform=fig.transFigure)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # 为底部的图例预留空间

# Display the figure
plt.show()

# Optionally save the figure (uncomment to use)
# plt.savefig('unet_performance_metrics.png', dpi=300, bbox_inches='tight')
# plt.savefig('unet_performance_metrics.pdf', bbox_inches='tight')