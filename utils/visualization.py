import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_metrics(conf_matrix, roc_data, f1_score, confidence_intervals, 
                loss_history, class_recall, feature_importance):
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    for i in range(len(roc_data)):
        fpr, tpr, _ = roc_data[i]
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig('roc_curves.png')
    plt.close()
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 8))
    plt.plot(loss_history['train'], label='Train Loss')
    plt.plot(loss_history['val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()
    
    # 绘制SHAP值
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance (SHAP values)')
    plt.savefig('shap_values.png')
    plt.close()