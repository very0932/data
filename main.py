import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.bayesian_cnn import BayesianCNN
from losses.focal_loss import MultiTaskLoss
from utils.visualization import plot_metrics
from train import train_epoch, evaluate
import shap
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning
from dataset import CustomDataset  # 添加数据集导入

# 过滤 numba 相关警告
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)
warnings.filterwarnings('ignore', category=NumbaWarning)

def main():
    print("程序开始执行...")
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("\n正在加载数据集...")
    try:
        train_dataset = CustomDataset(mode='train')
        print(f"训练集大小: {len(train_dataset)}")
        
        val_dataset = CustomDataset(mode='val')
        print(f"验证集大小: {len(val_dataset)}")
        
        test_dataset = CustomDataset(mode='test')
        print(f"测试集大小: {len(test_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)
        print("数据加载器创建成功")
        
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise
    
    # 初始化模型
    print("\n初始化模型...")
    model = BayesianCNN(num_classes=5).to(device)
    
    # 计算类别权重
    train_labels = torch.tensor([label for _, label, _ in train_dataset])
    class_counts = torch.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = class_weights.to(device)
    
    print("类别权重:", class_weights)
    
    # 初始化损失函数（不传入类别权重）
    criterion = MultiTaskLoss()
    
    # 使用较小的学习率和权重衰减
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    # 调整学习率调度器的参数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.2,  # 更温和的学习率衰减
        patience=10,  # 更长的等待周期
        verbose=True
    )
    
    # 训练循环
    num_epochs = 100
    loss_history = {'train': [], 'val': []}
    best_f1 = 0
    patience = 10
    no_improve = 0
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        total_loss = 0
        for batch_idx, (images, labels, masks) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            cls_outputs = model(images)  # 获取分类输出
            
            # 确保输出维度正确
            if cls_outputs.shape[0] != labels.shape[0]:
                cls_outputs = cls_outputs.view(labels.shape[0], -1)
            
            # 计算损失
            losses = criterion(
                cls_pred=cls_outputs,
                seg_pred=torch.zeros_like(masks),
                cls_target=labels,
                seg_target=masks
            )
            
            # 如果损失函数返回多个值，取第一个作为总损失
            if isinstance(losses, (list, tuple)):
                total_loss_batch = sum(losses)
            else:
                total_loss_batch = losses
                
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            
        avg_loss = total_loss / len(train_loader)
        loss_history['train'].append(avg_loss)
        
        # 验证
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels, masks in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    masks = masks.to(device)
                    
                    cls_outputs = model(images)
                    if cls_outputs.shape[0] != labels.shape[0]:
                        cls_outputs = cls_outputs.view(labels.shape[0], -1)
                        
                    losses = criterion(
                        cls_pred=cls_outputs,
                        seg_pred=torch.zeros_like(masks),
                        cls_target=labels,
                        seg_target=masks
                    )
                    
                    if isinstance(losses, (list, tuple)):
                        val_loss += sum(losses).item()
                    else:
                        val_loss += losses.item()
    
                avg_val_loss = val_loss / len(val_loader)
                conf_matrix, f1, confidence_intervals, pred_probs = evaluate(model, val_loader, device)
                print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, F1-Score: {f1:.4f}')
                
                # 更新学习率
                scheduler.step(f1)
                
                if f1 > best_f1:
                    best_f1 = f1
                    no_improve = 0
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print("\n触发早停，开始最终评估...")
                        
                        # 加载最佳模型进行测试
                        model.load_state_dict(torch.load('best_model.pth'))
                        model.eval()
                        
                        # 最终评估
                        print("正在进行测试集评估...")
                        conf_matrix, f1, confidence_intervals, pred_probs = evaluate(model, test_loader, device)
                        print(f"测试集 F1 分数: {f1:.4f}")
                        
                        # 计算类别召回率
                        class_recall = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
                        print("各类别召回率:", class_recall)
                        
                        # 计算SHAP值
                        print("\n开始计算SHAP值...")
                        test_images = []
                        for images, _, _ in test_loader:
                            test_images.append(images)
                            if len(test_images) * test_loader.batch_size >= 100:
                                break
                        test_images = torch.cat(test_images, dim=0)[:100]
                        
                        background = next(iter(train_loader))[0][:100].to(device)
                        all_shap_values = []
                        
                        for target_class in range(5):
                            print(f"计算第 {target_class+1}/5 个类别的SHAP值...")
                            wrapped_model = ModelWrapper(model, target_class).to(device)
                            explainer = shap.GradientExplainer(wrapped_model, background)
                            shap_values = explainer.shap_values(test_images.to(device))
                            all_shap_values.append(shap_values)
                        
                        # 合并所有类别的SHAP值
                        shap_values = np.array(all_shap_values)
                        
                        print("\n生成可视化结果...")
                        # 可视化结果
                        plot_metrics(conf_matrix, roc_curves, f1, confidence_intervals, 
                                   loss_history, class_recall, shap_values)
                        
                        print("\n评估完成！")
                        return  # 直接结束程序

if __name__ == '__main__':
    try:
        print("开始执行主程序...")
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()