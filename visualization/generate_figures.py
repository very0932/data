import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models.unet_se import UNet
from models.unet_se_aspp import UNetSEASPP
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

# 设置字体和样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

def load_model_results():
    """加载所有模型的训练结果"""
    save_dir = '/Users/very/源代码/Thesis_data/models/checkpoints'
    results = {}
    
    # 加载U-Net结果
    unet_path = os.path.join(save_dir, 'UNet_checkpoint.pth')
    if os.path.exists(unet_path):
        unet_data = torch.load(unet_path)
        results['U-Net'] = unet_data['history']
    
    # 加载U-Net+SE结果
    unet_se_path = os.path.join(save_dir, 'UNetSE_checkpoint.pth')
    if os.path.exists(unet_se_path):
        unet_se_data = torch.load(unet_se_path)
        results['U-Net+SE'] = unet_se_data['history']
    
    # 加载U-Net+SE+ASPP结果
    unet_se_aspp_path = os.path.join(save_dir, 'UNetSEASPP_checkpoint.pth')
    if os.path.exists(unet_se_aspp_path):
        unet_se_aspp_data = torch.load(unet_se_aspp_path)
        results['U-Net+SE+ASPP'] = unet_se_aspp_data['history']
    
    return results

def evaluate_models(test_loader, device):
    """评估所有模型的性能"""
    save_dir = '/Users/very/源代码/Thesis_data/models/checkpoints'
    models = {}
    metrics = {}
    
    # 加载U-Net模型
    unet_path = os.path.join(save_dir, 'UNet_checkpoint.pth')
    if os.path.exists(unet_path):
        unet = UNet(n_channels=3, n_classes=5, use_se=False)
        unet.load_state_dict(torch.load(unet_path)['model_state_dict'])
        unet.to(device)
        models['U-Net'] = unet
    
    # 加载U-Net+SE模型
    unet_se_path = os.path.join(save_dir, 'UNetSE_checkpoint.pth')
    if os.path.exists(unet_se_path):
        unet_se = UNet(n_channels=3, n_classes=5, use_se=True)
        unet_se.load_state_dict(torch.load(unet_se_path)['model_state_dict'])
        unet_se.to(device)
        models['U-Net+SE'] = unet_se
    
    # 加载U-Net+SE+ASPP模型
    unet_se_aspp_path = os.path.join(save_dir, 'UNetSEASPP_checkpoint.pth')
    if os.path.exists(unet_se_aspp_path):
        unet_se_aspp = UNetSEASPP(n_channels=3, n_classes=5)
        unet_se_aspp.load_state_dict(torch.load(unet_se_aspp_path)['model_state_dict'])
        unet_se_aspp.to(device)
        models['U-Net+SE+ASPP'] = unet_se_aspp
    
    # 评估每个模型
    for name, model in models.items():
        model.eval()
        dice_scores = []
        iou_scores = []
        sensitivity = []
        specificity = []
        
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model