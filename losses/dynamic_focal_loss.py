import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DynamicFocalLoss(nn.Module):
    """
    动态Focal Loss实现
    根据类别样本量自适应调整γ参数，稀有类别获得更高的γ值
    """
    def __init__(self, class_counts=[200, 150, 100, 50, 16], base_gamma=2.0, alpha=None, reduction='mean'):
        super(DynamicFocalLoss, self).__init__()
        self.base_gamma = base_gamma
        self.reduction = reduction
        
        if alpha is None:
            total_samples = sum(class_counts)
            self.alpha = torch.tensor([total_samples / (len(class_counts) * count) for count in class_counts])
            self.alpha = self.alpha / self.alpha.sum()
        else:
            self.alpha = torch.tensor(alpha)
            
        max_count = max(class_counts)
        self.gammas = torch.tensor([self.base_gamma * (1 + np.log(max_count / count)) for count in class_counts])
        
        print(f"动态Focal Loss初始化: \n类别权重: {self.alpha} \n动态γ值: {self.gammas}")
        
    def forward(self, inputs, targets):
        device = inputs.device
        self.alpha = self.alpha.to(device)
        self.gammas = self.gammas.to(device)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        probs = F.softmax(inputs, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1)).squeeze(1)
        
        batch_gammas = torch.index_select(self.gammas, 0, targets)
        focal_weights = (1 - probs) ** batch_gammas
        batch_alphas = torch.index_select(self.alpha, 0, targets)
        
        loss = batch_alphas * focal_weights * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss