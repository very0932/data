import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class IDRiDDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        IDRiD数据集加载器
        
        Args:
            root: 数据集根目录
            split: 'train', 'val', 或 'test'
            transform: 图像预处理
        """
        self.root = root
        self.transform = transform
        self.split = split
        
        # 加载标签文件
        csv_path = os.path.join(root, f'{split}.csv')
        if not os.path.exists(csv_path):
            self._prepare_split_csv()
            
        self.data = pd.read_csv(csv_path)
        self.img_dir = os.path.join(root, 'images')
        
        print(f"加载 {split} 集: {len(self.data)} 个样本")
        
    def _prepare_split_csv(self):
        orig_csv = os.path.join(self.root, 'labels.csv')
        if not os.path.exists(orig_csv):
            raise FileNotFoundError(f"找不到标签文件: {orig_csv}")
            
        df = pd.read_csv(orig_csv)
        
        # 按类别分层抽样
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()
        
        for grade in range(5):
            grade_df = df[df['grade'] == grade]
            n = len(grade_df)
            
            train_idx = int(0.7 * n)
            val_idx = int(0.85 * n)
            
            train_df = pd.concat([train_df, grade_df.iloc[:train_idx]])
            val_df = pd.concat([val_df, grade_df.iloc[train_idx:val_idx]])
            test_df = pd.concat([test_df, grade_df.iloc[val_idx:]])
        
        train_df.to_csv(os.path.join(self.root, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.root, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.root, 'test.csv'), index=False)
        
        if self.split == 'train':
            self.data = train_df
        elif self.split == 'val':
            self.data = val_df
        else:
            self.data = test_df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        grade = self.data.iloc[idx]['grade']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, grade, None