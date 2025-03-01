import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

class DiceLoss(nn.Module):
    """
    Dice损失函数
    用于处理分割任务中的类别不平衡问题
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # 使用reshape代替view，解决内存不连续问题
        batch_size = targets.size(0)
        probs = probs.reshape(batch_size, -1)
        targets = targets.reshape(batch_size, -1)
        
        # 计算交集
        intersection = (probs * targets).sum(dim=1)
        
        # 计算Dice系数
        dice = (2.0 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        
        # 返回Dice损失
        return 1.0 - dice.mean()

class BoundaryLoss(nn.Module):
    """
    边界损失函数
    基于预测掩码与真实掩码边界之间的距离
    """
    def __init__(self, theta=1.0):
        super(BoundaryLoss, self).__init__()
        self.theta = theta
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # 计算边界
        # 使用Sobel算子近似计算梯度
        def get_boundary(tensor):
            # 创建Sobel算子
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)
            
            batch_size, channels, height, width = tensor.shape
            boundaries = torch.zeros_like(tensor)
            
            for b in range(batch_size):
                for c in range(channels):
                    # 提取单个通道
                    channel = tensor[b, c:c+1]
                    
                    # 应用Sobel算子
                    grad_x = F.conv2d(channel, sobel_x, padding=1)
                    grad_y = F.conv2d(channel, sobel_y, padding=1)
                    
                    # 计算梯度幅度
                    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
                    
                    # 阈值处理得到边界
                    boundaries[b, c] = (grad_magnitude > 0.5).float()[0]
            
            return boundaries
        
        # 获取预测和真实边界
        pred_boundary = get_boundary(probs)
        target_boundary = get_boundary(targets)
        
        # 计算边界距离损失
        # 使用二值交叉熵作为边界损失的近似
        boundary_loss = F.binary_cross_entropy(pred_boundary, target_boundary, reduction='mean')
        
        return boundary_loss

class CombinedLoss(nn.Module):
    """
    组合损失函数
    结合Dice损失和边界损失
    """
    def __init__(self, dice_weight=0.7, boundary_weight=0.3, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.class_weights = class_weights
        
    def forward(self, logits, targets):
        # 基本损失计算
        dice_loss = self.dice_loss(logits, targets)
        boundary_loss = self.boundary_loss(logits, targets)
        
        # 应用类别权重（如果提供）
        if self.class_weights is not None:
            # 确保权重张量在正确的设备上
            device = logits.device
            weights = torch.tensor(self.class_weights, device=device)
            
            # 对每个类别应用权重
            weighted_dice = 0
            for c in range(logits.shape[1]):
                # 确保张量连续性
                class_logits = logits[:, c:c+1].contiguous()
                class_targets = targets[:, c:c+1].contiguous()
                class_dice = self.dice_loss(class_logits, class_targets)
                weighted_dice += weights[c] * class_dice
            
            # 使用加权平均替代原始dice_loss
            dice_loss = weighted_dice / weights.sum()
        
        # 组合损失
        combined_loss = self.dice_weight * dice_loss + self.boundary_weight * boundary_loss
        
        return combined_loss