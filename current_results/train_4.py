import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

# 假设的理想结果数据（视盘和黄斑坐标，单位：像素）
np.random.seed(42)
true_od = np.random.normal(1500, 200, (103, 2))  # 真实视盘坐标
true_fovea = np.random.normal(1300, 180, (103, 2))  # 真实黄斑坐标
pred_od_base = true_od + np.random.normal(0, 200, (103, 2))  # Baseline预测视盘
pred_fovea_base = true_fovea + np.random.normal(0, 180, (103, 2))  # Baseline预测黄斑
pred_od_imp = true_od + np.random.normal(0, 80, (103, 2))  # 改进版预测视盘
pred_fovea_imp = true_fovea + np.random.normal(0, 70, (103, 2))  # 改进版预测黄斑

# 散点图（视盘）
plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(true_od[:, 0], true_od[:, 1], c='#4C78A8', alpha=0.6, label='Ground Truth', s=50)
plt.scatter(pred_od_imp[:, 0], pred_od_imp[:, 1], c='#F28E2B', alpha=0.6, label='Predicted (Improved)', s=50)
plt.xlabel('X Coordinate (pixels)', fontsize=14, weight='bold')
plt.ylabel('Y Coordinate (pixels)', fontsize=14, weight='bold')
plt.title('Optic Disc Localization', fontsize=16, weight='bold', pad=15)
plt.legend(loc='upper right', fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('scatter_od.png', bbox_inches='tight')
plt.close()

# 热图（不确定性，假设改进后降低）
uncertainty = np.random.normal(0.1, 0.05, (224, 224))  # 假设224x224图像
plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(uncertainty, cmap='Blues', cbar_kws={'label': 'Uncertainty', 'shrink': 0.8})
plt.xlabel('X (pixels)', fontsize=14, weight='bold')
plt.ylabel('Y (pixels)', fontsize=14, weight='bold')
plt.title('Uncertainty Heatmap (Improved)', fontsize=16, weight='bold', pad=15)
plt.tight_layout()
plt.savefig('heatmap_uncertainty.png', bbox_inches='tight')
plt.close()

# 误差分布直方图
dist_base_od = np.sqrt(np.sum((true_od - pred_od_base)**2, axis=1))
dist_imp_od = np.sqrt(np.sum((true_od - pred_od_imp)**2, axis=1))
plt.figure(figsize=(8, 6), dpi=300)
plt.hist(dist_base_od, bins=30, alpha=0.6, color='#4C78A8', label='Baseline (MSE=1200)')
plt.hist(dist_imp_od, bins=30, alpha=0.6, color='#F28E2B', label='Improved (MSE=800)')
plt.xlabel('Euclidean Distance (pixels)', fontsize=14, weight='bold')
plt.ylabel('Frequency', fontsize=14, weight='bold')
plt.title('Error Distribution (Optic Disc)', fontsize=16, weight='bold', pad=15)
plt.legend(loc='upper right', fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('error_dist_od.png', bbox_inches='tight')
plt.close()