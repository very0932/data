import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

def load_sample_images():
    """加载各类别的示例图像"""
    image_dir = '/Users/very/源代码/Thesis_data/processed_data/images/train'
    grading_df = pd.read_csv('/Users/very/源代码/Thesis_data/processed_data/grading_labels.csv')
    
    # 为每个DR和DME等级选择一个示例图像
    samples = {}
    for dr_grade in range(5):  # DR分为0-4级
        for dme_grade in range(3):  # DME分为0-2级
            mask = (grading_df['DR_grade'] == dr_grade) & (grading_df['DME_grade'] == dme_grade)
            if mask.any():
                img_id = grading_df[mask].iloc[0]['image_id']
                img_path = os.path.join(image_dir, f"{img_id}.jpg")
                if os.path.exists(img_path):
                    samples[(dr_grade, dme_grade)] = img_path
    
    return samples

def plot_sample_images(samples):
    """绘制各类别样本图"""
    plt.figure(figsize=(15, 10))
    plt.suptitle('样本图像示例 (DR等级 vs DME等级)', fontsize=16)
    
    for i, (key, img_path) in enumerate(samples.items()):
        dr_grade, dme_grade = key
        img = Image.open(img_path)
        plt.subplot(5, 3, i+1)
        plt.imshow(img)
        plt.title(f'DR={dr_grade}, DME={dme_grade}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/very/源代码/Thesis_data/visualization/sample_images.png')
    plt.close()

def plot_sample_statistics():
    """绘制样本数量统计表"""
    grading_df = pd.read_csv('/Users/very/源代码/Thesis_data/processed_data/grading_labels.csv')
    
    # 创建交叉统计表
    cross_tab = pd.crosstab(grading_df['DR_grade'], grading_df['DME_grade'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('DR和DME等级样本数量分布')
    plt.xlabel('DME等级')
    plt.ylabel('DR等级')
    
    plt.tight_layout()
    plt.savefig('/Users/very/源代码/Thesis_data/visualization/sample_statistics.png')
    plt.close()

def plot_multilabel_samples():
    """绘制多标签眼底图示例"""
    image_dir = '/Users/very/源代码/Thesis_data/processed_data/images/train'
    grading_df = pd.read_csv('/Users/very/源代码/Thesis_data/processed_data/grading_labels.csv')
    segmentation_df = pd.read_csv('/Users/very/源代码/Thesis_data/processed_data/segmentation_labels.csv')
    
    # 选择同时具有分级和分割标注的样本
    merged_df = pd.merge(grading_df, segmentation_df, on='image_id')
    sample_id = merged_df.iloc[0]['image_id']
    
    # 加载图像和对应的掩码
    img_path = os.path.join(image_dir, f"{sample_id}.jpg")
    mask_path = os.path.join('/Users/very/源代码/Thesis_data/processed_data/segmentation/masks/train', 
                            f"{sample_id}_mask.npy")
    
    img = Image.open(img_path)
    mask = np.load(mask_path)
    
    # 处理5通道掩码数据
    # 将5个通道合并为一个可视化掩码
    combined_mask = np.zeros((mask.shape[0], mask.shape[1]))
    for i in range(mask.shape[2]):
        combined_mask += mask[:,:,i] * (i + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 系统
    plt.rcParams['axes.unicode_minus'] = False
    
    # 原始图像
    plt.subplot(131)
    plt.imshow(img)
    plt.title('原始图像')
    plt.axis('off')
    
    # 分割掩码
    plt.subplot(132)
    plt.imshow(combined_mask, cmap='nipy_spectral')  # 使用彩色映射
    plt.title('分割掩码')
    plt.axis('off')
    
    # 叠加显示
    plt.subplot(133)
    plt.imshow(img)
    plt.imshow(combined_mask, alpha=0.3, cmap='nipy_spectral')
    plt.title('叠加显示')
    plt.axis('off')
    
    plt.suptitle(f'多标签示例 (DR={merged_df.iloc[0]["DR_grade"]}, DME={merged_df.iloc[0]["DME_grade"]})')
    plt.tight_layout()
    plt.savefig('/Users/very/源代码/Thesis_data/visualization/multilabel_sample.png')
    plt.close()

def create_grading_labels():
    """创建分级标签文件"""
    base_dir = '/Users/very/源代码/Thesis_data/processed_data'
    image_dir = os.path.join(base_dir, 'images/train')
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # 创建数据框
    data = []
    for img_file in image_files:
        image_id = img_file.replace('.jpg', '')
        # 这里我们随机分配DR和DME等级作为示例
        # 实际应用中应该使用真实的标注数据
        data.append({
            'image_id': image_id,
            'DR_grade': np.random.randint(0, 5),  # DR分为0-4级
            'DME_grade': np.random.randint(0, 3)  # DME分为0-2级
        })
    
    # 保存为CSV
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(base_dir, 'grading_labels.csv'), index=False)

def create_segmentation_labels():
    """创建分割标签文件"""
    base_dir = '/Users/very/源代码/Thesis_data/processed_data'
    segmentation_file = os.path.join(base_dir, 'segmentation_labels.csv')
    
    # 如果文件已存在，则跳过创建
    if os.path.exists(segmentation_file):
        print("segmentation_labels.csv 已存在，跳过创建步骤")
        return
        
    masks_dir = os.path.join(base_dir, 'segmentation/masks/train')
    
    # 获取所有掩码文件
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('_mask.npy')]
    
    # 创建数据框
    data = []
    for mask_file in mask_files:
        image_id = mask_file.replace('_mask.npy', '')
        data.append({
            'image_id': image_id,
            'mask_path': os.path.join('segmentation/masks/train', mask_file)
        })
    
    # 保存为CSV
    df = pd.DataFrame(data)
    df.to_csv(segmentation_file, index=False)
    print(f"已创建 {segmentation_file}")

if __name__ == '__main__':
    # 创建可视化输出目录
    visualization_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualization')
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 创建标签文件
    create_grading_labels()
    create_segmentation_labels()
    
    # 更新所有保存路径
    sample_images_path = os.path.join(visualization_dir, 'sample_images.png')
    sample_statistics_path = os.path.join(visualization_dir, 'sample_statistics.png')
    multilabel_sample_path = os.path.join(visualization_dir, 'multilabel_sample.png')
    
    # 生成三张图
    samples = load_sample_images()
    plot_sample_images(samples)
    plot_sample_statistics()
    plot_multilabel_samples()