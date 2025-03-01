import torch
import numpy as np
from models.bayesian_cnn import BayesianCNN
from dataset import CustomDataset
from torch.utils.data import DataLoader
from train import evaluate
import shap
from utils.visualization import plot_metrics

# 创建包装器类
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, target_class=0):
        super().__init__()
        self.model = model
        self.target_class = target_class
        
    def forward(self, x):
        output = self.model(x)
        return output[:, self.target_class]

def evaluate_trained_model():
    print("开始加载模型和数据...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_dataset = CustomDataset(mode='train')
    test_dataset = CustomDataset(mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)
    
    # 加载模型
    model = BayesianCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    print("\n进行测试集评估...")
    conf_matrix, f1, confidence_intervals, pred_probs = evaluate(model, test_loader, device)
    print(f"测试集 F1 分数: {f1:.4f}")
    
    # 计算并显示详细的评估指标
    class_recall = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    class_precision = conf_matrix.diagonal() / conf_matrix.sum(axis=0)
    
    print("\n详细评估结果:")
    print("混淆矩阵:")
    print(conf_matrix)
    print("\n各类别指标:")
    for i in range(5):
        print(f"类别 {i}:")
        print(f"- 召回率: {class_recall[i]:.4f}")
        print(f"- 精确率: {class_precision[i]:.4f}")
        print(f"- 样本数量: {conf_matrix.sum(axis=1)[i]}")
    
    print("\n计算SHAP值...")
    try:
        # 限制测试图像数量，减少计算量
        test_images = []
        max_images = 50  # 减少样本数量
        for images, _, _ in test_loader:
            test_images.append(images[:max_images])
            break  # 只取第一个batch
        test_images = torch.cat(test_images, dim=0)
        
        # 减少背景样本数量
        background = next(iter(train_loader))[0][:50].to(device)
        all_shap_values = []
        
        for target_class in range(5):
            print(f"计算第 {target_class+1}/5 个类别的SHAP值...")
            try:
                wrapped_model = ModelWrapper(model, target_class).to(device)
                explainer = shap.GradientExplainer(wrapped_model, background)
                print(f"- 创建解释器成功，开始计算SHAP值...")
                shap_values = explainer.shap_values(test_images.to(device))
                print(f"- 完成第 {target_class+1} 个类别的SHAP值计算")
                all_shap_values.append(shap_values)
            except Exception as e:
                print(f"计算第 {target_class+1} 个类别时出错: {str(e)}")
                all_shap_values.append(None)
        
        # 检查是否所有类别都计算成功
        if any(v is not None for v in all_shap_values):
            shap_values = np.array([v for v in all_shap_values if v is not None])
            print("\n生成可视化结果...")
            plot_metrics(conf_matrix, roc_curves, f1, confidence_intervals, 
                        None, class_recall, shap_values)
        else:
            print("所有类别的SHAP值计算都失败了，跳过可视化")
            
    except Exception as e:
        print(f"SHAP值计算过程出错: {str(e)}")
        print("跳过SHAP值计算，直接进行可视化...")
        plot_metrics(conf_matrix, roc_curves, f1, confidence_intervals, 
                    None, class_recall, None)
    
    print("评估完成！")

if __name__ == '__main__':
    evaluate_trained_model()