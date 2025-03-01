import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
            
        return focal_loss.mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.seg_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, cls_pred, seg_pred, cls_target, seg_target):
        cls_loss = self.focal_loss(cls_pred, cls_target)
        seg_loss = self.seg_loss(seg_pred, seg_target)
        
        # 动态权重
        total_loss = cls_loss + seg_loss
        return total_loss, cls_loss, seg_loss