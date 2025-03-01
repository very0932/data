import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, mode='train', base_dir='/Users/very/源代码/Thesis_data/processed_data'):
        self.mode = mode
        self.base_dir = base_dir
        
        # 检查数据目录是否存在
        if not os.path.exists(base_dir):
            raise ValueError(f"数据目录不存在: {base_dir}")
            
        # 读取预处理后的数据
        try:
            # 检查并加载图像和标签
            images_path = os.path.join(base_dir, 'grading/processed', f'{mode}_images.npy')
            labels_path = os.path.join(base_dir, 'grading/processed', f'{mode}_dr_labels.npy')
            
            if not os.path.exists(images_path):
                raise FileNotFoundError(f"找不到图像文件: {images_path}")
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"找不到标签文件: {labels_path}")
            
            # 加载并预处理图像数据
            self.images = np.load(images_path).astype(np.float32)
            if len(self.images.shape) != 4:
                raise ValueError(f"图像数据维度错误: {self.images.shape}, 应为 (N, H, W, C)")
                
            # 调整图像维度顺序并归一化
            self.images = np.transpose(self.images, (0, 3, 1, 2))
            self.images = np.clip(self.images / 255.0, 0, 1)
            
            # 加载并验证标签
            self.dr_labels = np.load(labels_path)
            if len(self.dr_labels) != len(self.images):
                raise ValueError(f"标签数量({len(self.dr_labels)})与图像数量({len(self.images)})不匹配")
            
            print(f"成功加载 {mode} 预处理数据：图像 {self.images.shape}, 标签 {self.dr_labels.shape}")
            
            # 加载掩码数据
            try:
                masks_path = os.path.join(base_dir, 'grading/processed', f'{mode}_masks.npy')
                if not os.path.exists(masks_path):
                    raise FileNotFoundError(f"找不到掩码文件: {masks_path}")
                    
                masks = np.load(masks_path)
                print(f"成功加载掩码数据：{masks.shape}")
                
                # 处理掩码数量不足的情况
                if len(masks) < len(self.images):
                    print(f"警告：掩码数量({len(masks)})小于图像数量({len(self.images)})，生成随机掩码填充")
                    additional_masks = np.random.rand(
                        len(self.images) - len(masks),
                        masks.shape[1],
                        masks.shape[2],
                        masks.shape[3]
                    )
                    masks = np.concatenate([masks, additional_masks], axis=0)
                
                self.masks = torch.FloatTensor(masks)
            except Exception as e:
                print(f"警告：加载掩码数据失败，使用随机掩码: {str(e)}")
                self.masks = torch.rand(len(self.images), 5, 224, 224)
                
        except Exception as e:
            raise ValueError(f"初始化数据集失败: {str(e)}")
    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        try:
            # 获取图像和标签
            image = torch.from_numpy(self.images[idx]).float()
            dr_label = torch.tensor(self.dr_labels[idx], dtype=torch.long)
            mask = self.masks[idx]  # 已经是 torch.FloatTensor
            
            # 验证数据
            if torch.isnan(image).any():
                raise ValueError(f"图像包含 NaN 值")
            if torch.isnan(mask).any():
                raise ValueError(f"掩码包含 NaN 值")
                
            return image, dr_label, mask
            
        except Exception as e:
            print(f"获取数据失败 (idx={idx}): {str(e)}")
            raise