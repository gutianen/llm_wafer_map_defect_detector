import os
import cv2
import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import json
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from ultralytics import YOLO
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
import gc
import warnings

# ==== 画图设置 ====
plt.rcParams['font.size'] = 14
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# ==== 初始化设置 ====
warnings.filterwarnings('ignore')  # 忽略警告信息
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭tokenizer并行功能
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# 配置参数
config = {
    'training_name': 'wafer_detection_traning',
    'wafer_data_dir': './dataset/57_wafer_detection',  # 数据集路径
    'yolo_data_dir': './dataset/57_wafer_detection/yolo_data',  # YOLO数据路径
    'yolo_model_name': 'yolov8n.pt',
    'clip_model_name': '/root/models/clip-vit-base-patch32',
    'defect_classes': ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none'],
    'defect_descriptions': {
        'Center': "中心区域集中缺陷，通常由工艺中心异常引起",
        'Donut': "环形缺陷图案，可能由于旋转不均匀导致",
        'Edge-Loc': "边缘局部缺陷，通常与夹具或边缘处理相关",
        'Edge-Ring': "边缘环状缺陷，可能由于边缘刻蚀不均匀",
        'Loc': "局部随机缺陷，可能由污染或局部工艺问题引起",
        'Near-full': "近全表面缺陷，表明系统性工艺问题",
        'Random': "随机分布缺陷，通常与颗粒污染相关",
        'Scratch': "刮擦缺陷，机械损伤或操作问题",
        'none': "无缺陷，良品晶圆"
    },
    'test_size': 0.2,  # 保留配置，实际使用已有划分
    'yolo_epochs': 80,
    'yolo_img_size': 416,
    'yolo_batch_size': 64,
    'yolo_learning_rate': 0.005,
    'yolo_patience': 25,
    'yolo_num_of_workers': 4,
    'yolo_enhance_mosaic': 1.0,  # 启用mosaic增强
    'yolo_enhance_hsv_h': 0.015,  # 色调扰动
    'yolo_enhance_hsv_s': 0.7,  # 饱和度扰动
    'yolo_enhance_hsv_v': 0.4,  # 明度扰动
    'clip_learning_rate': 1e-4,
    'clip_epochs': 50,
    'clip_batch_size': 64,
    'clip_drop_out': 0.3,
    'clip_early_stop_patience': 20,
    'clip_weight_decay': 1e-4,
    'clip_num_of_workers': 4,
    'clip_lr_reduce_patience': 10,
    'clip_lr_reduce_factor': 0.85,
    'clip_tune_model_file_name': 'clip_tune_model.pth'
}


class WaferDataset:
    """晶圆数据集分析类"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.classes = config['defect_classes']
        self.class_map = {cls: idx for idx, cls in enumerate(self.classes)}

    def analyze_dataset(self):
        """分析训练、测试、验证集的类别分布"""
        train_counts = {}
        test_counts = {}
        valid_counts = {}

        # 分析三个数据集的类别分布
        for split in ['train', 'test', 'valid']:
            for cls in self.classes:
                # 构建标签文件路径
                label_path = os.path.join(self.data_path, split, 'labels')
                if not os.path.exists(label_path):
                    continue

                # 统计属于当前类别的标签文件数量
                count = 0
                for label_file in os.listdir(label_path):
                    if label_file.endswith('.txt'):
                        with open(os.path.join(label_path, label_file), 'r') as f:
                            lines = f.readlines()
                            # 跳过注释行，找到第一个标注行
                            annotation_line = None
                            for line in lines:
                                stripped_line = line.strip()
                                if not stripped_line.startswith('#') and stripped_line:
                                    annotation_line = stripped_line
                                    break

                            if annotation_line:  # 如果有标注内容
                                class_id = int(annotation_line.split()[0])
                                class_name = self.classes[class_id] if 0 <= class_id < len(self.classes) else 'unknown'
                                if class_name == cls:
                                    count += 1
                            else:  # 空标签文件或只有注释对应'none'类别
                                if cls == 'none':
                                    count += 1

                if split == 'train':
                    train_counts[cls] = count
                elif split == 'test':
                    test_counts[cls] = count
                else:
                    valid_counts[cls] = count

        return train_counts, test_counts, valid_counts

    def visualize_distribution(self, train_counts, test_counts, valid_counts):
        """可视化三个数据集的类别分布"""
        plt.figure(figsize=(20, 6))

        plt.subplot(1, 3, 1)
        sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()))
        plt.title('训练集类别分布')
        plt.xticks(rotation=45)
        plt.ylabel('样本数量')

        plt.subplot(1, 3, 2)
        sns.barplot(x=list(test_counts.keys()), y=list(test_counts.values()))
        plt.title('测试集类别分布')
        plt.xticks(rotation=45)
        plt.ylabel('样本数量')

        plt.subplot(1, 3, 3)
        sns.barplot(x=list(valid_counts.keys()), y=list(valid_counts.values()))
        plt.title('验证集类别分布')
        plt.xticks(rotation=45)
        plt.ylabel('样本数量')

        plt.tight_layout()
        plt.savefig(f"{config['training_name']}_dataset_distribution.png", dpi=300)
        plt.show()

    def load_metadata(self, split):
        """加载指定数据集分割（train/test/valid）的工艺参数和缺陷特征"""
        metadata = []  # 存储 (工艺参数文本, 缺陷特征文本, 缺陷类型标签)
        src_label_dir = os.path.join(self.data_path, split, 'labels')

        if not os.path.exists(src_label_dir):
            return metadata

        for label_file in os.listdir(src_label_dir):
            if label_file.endswith('.txt'):
                process_params = ""
                defect_features = ""
                class_id = -1  # 默认无效值

                # 读取标签文件内容
                with open(os.path.join(src_label_dir, label_file), 'r') as f:
                    lines = f.readlines()

                # 解析注释行中的工艺参数和缺陷特征
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line.startswith('# process_params:'):
                        process_params = stripped_line.replace('# process_params:', '').strip()
                    elif stripped_line.startswith('# defect_features:'):
                        defect_features = stripped_line.replace('# defect_features:', '').strip()
                    elif not stripped_line.startswith('#') and stripped_line:
                        # 解析缺陷类型标签（非注释行的第一列）
                        class_id = int(stripped_line.split()[0])

                # 处理无缺陷情况（none类）
                if class_id == -1:  # 无标注行，对应'none'类
                    class_id = self.class_map['none']

                metadata.append({
                    'process_text': process_params,
                    'defect_text': defect_features,
                    'label': class_id
                })

        return metadata


class YOLODataLoader:
    """YOLO数据集处理类 - 适配新的数据集结构"""

    def __init__(self, wafer_data_path, yolo_data_path):
        self.source_path = wafer_data_path  # 指向包含train/test/valid的根目录
        self.yolo_path = yolo_data_path
        self.classes = config['defect_classes']

    def create_yolo_annotations(self):
        """创建YOLO格式的数据集，直接使用已有划分和标注"""
        # 创建必要的目录
        for split in ['train', 'test', 'valid']:
            os.makedirs(os.path.join(self.yolo_path, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(self.yolo_path, 'labels', split), exist_ok=True)

        # 处理三个数据集分割
        for split in ['train', 'test', 'valid']:
            # 源图像和标签路径
            src_img_dir = os.path.join(self.source_path, split, 'images')
            src_label_dir = os.path.join(self.source_path, split, 'labels')

            if not os.path.exists(src_img_dir) or not os.path.exists(src_label_dir):
                print(f"警告: 未找到{split}集的图像或标签目录，已跳过")
                continue

            # 处理每个图像文件
            for img_file in os.listdir(src_img_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    # 构建图像和标签路径
                    img_name = os.path.splitext(img_file)[0]
                    src_img_path = os.path.join(src_img_dir, img_file)
                    src_label_path = os.path.join(src_label_dir, f"{img_name}.txt")

                    # 目标路径
                    dest_img_path = os.path.join(self.yolo_path, 'images', split, img_file)
                    dest_label_path = os.path.join(self.yolo_path, 'labels', split, f"{img_name}.txt")

                    # 创建图像软链接
                    if not os.path.exists(dest_img_path):
                        os.symlink(os.path.abspath(src_img_path), dest_img_path)

                    # 复制或创建标签文件，只保留标注内容，去除注释行
                    if os.path.exists(src_label_path):
                        # 读取源标签文件并过滤注释行
                        with open(src_label_path, 'r') as f:
                            lines = f.readlines()

                        # 只保留非注释行
                        annotation_lines = []
                        for line in lines:
                            stripped_line = line.strip()
                            if not stripped_line.startswith('#') and stripped_line:
                                annotation_lines.append(stripped_line)

                        # 写入目标标签文件
                        with open(dest_label_path, 'w') as f:
                            f.write('\n'.join(annotation_lines))
                    else:
                        # 如果没有标签文件，创建空文件（表示无缺陷）
                        with open(dest_label_path, 'w') as f:
                            pass

        # 创建YOLO配置文件
        self.create_yaml_config()

    def create_yaml_config(self):
        """创建YOLO训练所需的yaml配置文件"""
        config_yaml = {
            'path': os.path.abspath(self.yolo_path),
            'train': 'images/train',
            'val': 'images/valid',  # 使用valid作为验证集
            'test': 'images/test',  # 新增测试集配置
            'nc': len(self.classes),
            'names': self.classes
        }

        with open(os.path.join(self.yolo_path, 'wafer_dataset.yaml'), 'w') as f:
            yaml.dump(config_yaml, f)


class YOLODefectDetector:
    """YOLO缺陷检测模型类"""

    def __init__(self, device):
        self.device = device
        self.model = YOLO(config['yolo_model_name'])
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_map50': [], 'val_map50': []
        }

    def train(self, data_yaml, epochs=100, img_size=512, batch_size=16, learning_rate=0.01, patience=10,
              num_of_workers=6, enhance_mosaic=1.0, enhance_hsv_h=0.01, enhance_hsv_s=0.5, enhance_hsv_v=0.5):
        """训练YOLO模型"""
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            lr0=learning_rate,
            patience=patience,
            workers=num_of_workers,
            save=True,
            device=self.device,
            augment=True,
            rect=True,  # 矩形训练
            cos_lr=True,  # 余弦学习率
            verbose=False,
            amp=True,  # 启用自动混合精度
            mosaic=enhance_mosaic,  # 启用mosaic增强
            hsv_h=enhance_hsv_h,  # 色调扰动
            hsv_s=enhance_hsv_s,  # 饱和度扰动
            hsv_v=enhance_hsv_v  # 明度扰动
        )

        # 记录训练历史
        if hasattr(results, 'results') and results.results:
            for epoch_result in results.results:
                self.training_history['train_loss'].append(epoch_result.get('train/loss', 0))
                self.training_history['val_loss'].append(epoch_result.get('val/loss', 0))
                self.training_history['train_map50'].append(epoch_result.get('metrics/mAP50', 0))
                self.training_history['val_map50'].append(epoch_result.get('metrics/mAP50(B)', 0))

        return results

    def evaluate(self, data_yaml, split='test'):
        """在指定数据集上评估YOLO模型"""
        results = self.model.val(
            data=data_yaml,
            device=self.device,
            split=split  # 可以指定评估test或valid集
        )
        print(f"YOLO模型{split}集评估结果：")
        print(f"mAP50: {results.box.map50:.3f}, mAP50-95: {results.box.map:.3f}")
        return results

    def visualize_training(self):
        """可视化YOLO训练过程"""
        if not self.training_history['train_loss']:
            print("没有训练历史数据")
            return

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_loss'], label='训练损失', linewidth=2)
        plt.plot(self.training_history['val_loss'], label='验证损失', linewidth=2)
        plt.title('YOLO训练损失', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['train_map50'], label='训练mAP50', linewidth=2)
        plt.plot(self.training_history['val_map50'], label='验证mAP50', linewidth=2)
        plt.title('YOLO mAP50指标', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('mAP50')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{config['training_name']}_yolo_loss_curves.png", dpi=300, bbox_inches='tight')
        plt.show()

    def detect_defects(self, image_path, conf_threshold=0.5):
        """检测图像中的缺陷区域"""
        results = self.model(image_path, conf=conf_threshold)
        defect_regions = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    defect_regions.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self.model.names[class_id] if hasattr(self.model,
                                                                            'names') else f'class_{class_id}'
                    })
        return defect_regions


class ClipDataset(Dataset):
    """CLIP模型的数据集类，支持多模态数据"""

    def __init__(self, patches, process_texts, defect_texts, labels, processor, augmenter=None):
        self.patches = patches
        self.process_texts = process_texts  # 工艺参数文本列表
        self.defect_texts = defect_texts  # 缺陷特征文本列表
        self.labels = labels
        self.processor = processor
        self.augmenter = augmenter

    def __getitem__(self, idx):
        patch = self.patches[idx]
        if self.augmenter:
            patch = self.augmenter(patch)
        # 处理图像
        image_inputs = self.processor(images=patch, return_tensors="pt")['pixel_values'].squeeze()

        # 返回图像、工艺文本、缺陷文本、标签
        return (image_inputs,
                self.process_texts[idx],
                self.defect_texts[idx],
                torch.tensor(self.labels[idx], dtype=torch.long))

    def __len__(self):
        return len(self.patches)


class CLIPDefectClassifier:
    """CLIP缺陷分类器类，支持多模态特征融合"""
    clip_model_file_path = None
    checkpoint = None
    early_stop = None

    def __init__(self, device, model_name, yolo_model_path=None, clip_model_file_path=None, learning_rate=1e-4,
                 lr_reduce_factor=0.85, lr_reduce_patience=5, weight_decay=1e-4, early_stop_patience=10):
        self.device = device
        # 加载CLIP模型，保留视觉和文本编码器
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

        # 1. 缺陷类型分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 仅使用图像特征预测缺陷类型
            nn.ReLU(),
            nn.Dropout(config['clip_drop_out']),
            nn.Linear(256, len(config['defect_classes']))
        ).to(self.device)

        # 2. 工艺参数预测头（文本特征匹配）
        self.process_projector = nn.Sequential(
            nn.Linear(512, 512),  # 将图像特征投影到文本特征空间
            nn.LayerNorm(512)
        ).to(self.device)

        # 3. 缺陷特征预测头（文本特征匹配）
        self.defect_feature_projector = nn.Sequential(
            nn.Linear(512, 512),  # 将图像特征投影到文本特征空间
            nn.LayerNorm(512)
        ).to(self.device)

        # 预先生成训练集中的工艺和缺陷特征文本向量（用于匹配）
        self.process_text_features = self.preprocess_text_features('process')  # 工艺文本向量库
        self.defect_text_features = self.preprocess_text_features('defect')  # 缺陷特征文本向量库

        # 多任务损失函数（缺陷类型分类损失 + 文本匹配损失）
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # 用于图像特征与文本特征的匹配损失
        self.mse_loss = nn.MSELoss()
        # 优化器需包含新增的投影头参数
        self.optimizer = optim.AdamW(
            list(self.classifier.parameters()) +
            list(self.process_projector.parameters()) +
            list(self.defect_feature_projector.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=lr_reduce_factor,
            patience=lr_reduce_patience
        )
        self.defect_classes = config['defect_classes']
        self.defect_descriptions = config['defect_descriptions']
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        self.yolo_detector = YOLO(yolo_model_path) if yolo_model_path else None
        self.text_prompts = self.create_text_prompts()
        self.text_features = self.encode_text_prompts()
        self.early_stop = EarlyStopping(patience=early_stop_patience)
        # 数据增强
        self.augmenter = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
        ])
        # 断点续训
        if not clip_model_file_path:
            self.clip_model_file_path = config['clip_tune_model_file_name']
        else:
            self.clip_model_file_path = clip_model_file_path
        if os.path.exists(self.clip_model_file_path):
            self.load_model(self.clip_model_file_path)

    def forward(self, image_features):
        """前向传播：同时输出缺陷类型、工艺特征、缺陷特征"""
        # 1. 缺陷类型预测
        defect_logits = self.classifier(image_features)

        # 2. 工艺参数特征投影（图像→文本空间）
        process_proj = self.process_projector(image_features)
        # process_proj /= process_proj.norm(dim=-1, keepdim=True)
        process_proj = process_proj / process_proj.norm(dim=-1, keepdim=True)  # 修复这里

        # 3. 缺陷特征投影（图像→文本空间）
        defect_feat_proj = self.defect_feature_projector(image_features)
        # defect_feat_proj /= defect_feat_proj.norm(dim=-1, keepdim=True)
        defect_feat_proj = defect_feat_proj / defect_feat_proj.norm(dim=-1, keepdim=True)

        return defect_logits, process_proj, defect_feat_proj

    def preprocess_text_features(self, text_type):
        """预编码训练集中的工艺参数或缺陷特征文本，构建向量库"""
        # 加载训练集的工艺/缺陷文本（需从WaferDataset的metadata中获取）
        wafer_dataset = WaferDataset(config['wafer_data_dir'])
        train_metadata = wafer_dataset.load_metadata('train')

        # 提取对应类型的文本
        texts = []
        if text_type == 'process':
            texts = [m['process_text'] for m in train_metadata if m['process_text']]
            self.train_process_texts = texts  # 保存工艺文本列表
        else:
            texts = [m['defect_text'] for m in train_metadata if m['defect_text']]
            self.train_defect_texts = texts  # 保存缺陷特征文本列表

        # 编码文本为向量
        # 分批次处理文本，避免内存溢出
        batch_size = 32  # 调整批次大小
        features_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.clip_processor.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(self.device)

            with torch.no_grad():
                batch_features = self.clip_model.get_text_features(**inputs)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)
            features_list.append(batch_features)

        # 拼接所有批次的特征
        return torch.cat(features_list, dim=0)

    def create_text_prompts(self):
        """创建文本提示词"""
        prompts = []
        for cls in self.defect_classes:
            class_prompts = [
                f"a wafer map showing {cls} defect pattern",
                f"semiconductor wafer with {cls} defects",
                f"{cls} pattern on silicon wafer",
                f"wafer inspection result: {cls}",
                f"defect type {cls} in wafer manufacturing"
            ]
            prompts.extend(class_prompts)
        return prompts

    def encode_text_prompts(self):
        """编码文本提示词"""
        text_inputs = self.clip_processor.tokenizer(
            text=self.text_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        ).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def load_model(self, load_path):
        """加载预训练模型权重"""
        self.checkpoint = torch.load(load_path, map_location=self.device)
        self.classifier.load_state_dict(self.checkpoint['classifier_state_dict'])
        self.process_projector.load_state_dict(self.checkpoint['process_projector_state_dict'])
        self.defect_feature_projector.load_state_dict(self.checkpoint['defect_feature_projector_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.training_history = self.checkpoint['training_history']
        print(f"模型已从{load_path}加载")

    def save_model(self, save_path, epoch):
        """保存分类器模型权重"""
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'process_projector_state_dict': self.process_projector.state_dict(),
            'defect_feature_projector_state_dict': self.defect_feature_projector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'current_epoch': epoch  # 记录当前epoch
        }, save_path)
        print(f"模型已保存至: {save_path}")

    def parse_label_file(self, label_path):
        """解析标签文件，提取工艺参数、缺陷特征描述和边界框"""
        process_params = ""
        defect_features = ""
        bbox = None

        if not os.path.exists(label_path):
            return process_params, defect_features, bbox

        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 解析注释行
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('# process_params:'):
                process_params = stripped_line[len('# process_params:'):].strip()
            elif stripped_line.startswith('# defect_features:'):
                defect_features = stripped_line[len('# defect_features:'):].strip()

        # 解析标注行（边界框）
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line.startswith('#') and stripped_line:
                parts = stripped_line.split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = parts
                    return process_params, defect_features, (class_id, x_center, y_center, width, height)

        return process_params, defect_features, bbox

    def extract_defect_patches(self, img_path, true_class=None):
        """从图像中提取缺陷区域，返回图像patch和解析的文本信息"""
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法读取图像 {img_path}，已跳过")
            return [], "", ""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取图像文件名和路径信息
        img_dir = os.path.dirname(img_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 构建对应的标签文件路径
        label_path = os.path.join(os.path.dirname(img_dir), 'labels', f"{img_name}.txt")

        # 解析标签文件获取工艺参数、缺陷特征和边界框
        process_param, defect_feature, bbox_data = self.parse_label_file(label_path)


        # 处理边界框
        bbox = None
        if bbox_data:
            class_id, x_center, y_center, width, height = bbox_data
            h, w = image_rgb.shape[:2]

            # 转换YOLO格式到像素坐标
            x_center = float(x_center) * w
            y_center = float(y_center) * h
            width = float(width) * w
            height = float(height) * h

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # 确保坐标在有效范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            bbox = (x1, y1, x2, y2)

        # 如果有真实类别信息且为'none'，或者没有检测到边界框
        if (true_class == 'none') or (bbox is None):
            # 使用中心区域作为patch
            h, w = image_rgb.shape[:2]
            x1, y1 = max(0, w // 2 - 112), max(0, h // 2 - 112)
            x2, y2 = min(w, w // 2 + 112), min(h, h // 2 + 112)
            patch = image_rgb[y1:y2, x1:x2]
            patch = Image.fromarray(patch).resize((224, 224))
        else:
            # 使用标签文件中的边界框
            x1, y1, x2, y2 = bbox
            patch = image_rgb[y1:y2, x1:x2]
            if patch.size > 0:
                patch = Image.fromarray(patch).resize((224, 224))
            else:
                # 如果边界框无效，使用中心区域
                h, w = image_rgb.shape[:2]
                x1, y1 = max(0, w // 2 - 112), max(0, h // 2 - 112)
                x2, y2 = min(w, w // 2 + 112), min(h, h // 2 + 112)
                patch = image_rgb[y1:y2, x1:x2]
                patch = Image.fromarray(patch).resize((224, 224))

        # combined_text = f"工艺参数: {process_params}。缺陷特征: {defect_features}"
        return patch, process_param, defect_feature

    def load_clip_dataset(self, data_path, split):
        """加载指定分割的CLIP数据集，包含图像patch和文本信息"""
        patches = []
        process_texts = []  # 存储工艺参数
        defect_texts = []  # 缺陷特征描述
        labels = []

        # 构建图像和标签目录路径
        img_dir = os.path.join(data_path, split, 'images')
        label_dir = os.path.join(data_path, split, 'labels')

        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            print(f"警告: 未找到{split}集的图像或标签目录，已跳过")
            return patches, process_texts, defect_texts, labels

        # 处理每个图像
        for img_file in os.listdir(img_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_name = os.path.splitext(img_file)[0]
                img_path = os.path.join(img_dir, img_file)
                label_path = os.path.join(label_dir, f"{img_name}.txt")

                # 确定类别
                class_id = 0  # 默认值
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        # 查找第一个非注释行作为标注
                        annotation_line = None
                        for line in lines:
                            stripped_line = line.strip()
                            if not stripped_line.startswith('#') and stripped_line:
                                annotation_line = stripped_line
                                break

                        if annotation_line:
                            # 从标签文件获取类别ID
                            class_id = int(annotation_line.split()[0])
                            # 确保类别ID有效
                            if class_id < 0 or class_id >= len(self.defect_classes):
                                class_id = 0
                        else:
                            # 空标签文件或只有注释对应'none'类别
                            class_id = self.defect_classes.index('none')
                except Exception as e:
                    print(f"处理标签文件{label_path}时出错: {e}，使用默认类别")
                    class_id = 0

                # 提取缺陷区域和文本信息
                img_patch, process_text, defect_text = self.extract_defect_patches(
                    img_path,
                    self.defect_classes[class_id]
                )

                patches.append(img_patch)
                process_texts.append(process_text)
                defect_texts.append(defect_text)
                labels.append(class_id)

                # 对少数类进行数据增强
                if split == 'train' and Counter(labels)[class_id] < 100:
                    for _ in range(3):
                        augmented_patch = self.augmenter(img_patch)
                        patches.append(augmented_patch)
                        process_texts.append(process_text)
                        defect_texts.append(defect_text)
                        labels.append(class_id)

        return patches, process_texts, defect_texts, labels

    def fine_tune(self, data_path, epochs=10, batch_size=32):
        """微调CLIP模型，使用多模态特征融合"""
        if not self.yolo_detector and not os.path.exists(os.path.join(data_path, 'train', 'labels')):
            raise ValueError("请提供YOLO模型路径或确保标签文件存在以提取缺陷patch")

        # 加载训练集和验证集（包含图像和文本）
        train_patches, train_process_texts, train_defect_texts, train_labels = self.load_clip_dataset(data_path, 'train')
        val_patches, val_process_texts, val_defect_texts, val_labels = self.load_clip_dataset(data_path, 'valid')

        print(f"CLIP训练集样本数: {len(train_patches)}")
        print(f"CLIP验证集样本数: {len(val_patches)}")

        # 创建数据加载器
        train_dataset = ClipDataset(train_patches, train_process_texts, train_defect_texts, train_labels, self.clip_processor, self.augmenter)
        val_dataset = ClipDataset(val_patches, val_process_texts, val_defect_texts, val_labels, self.clip_processor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config['clip_num_of_workers']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config['clip_num_of_workers']
        )

        # 计算类别权重（处理不平衡）
        class_counts = Counter(train_labels)
        weights = torch.FloatTensor([len(train_labels) / (len(self.defect_classes) * class_counts.get(i, 1))
                                     for i in range(len(self.defect_classes))]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        scaler = GradScaler()  # 混合精度训练

        best_val_accuracy = 0.0
        best_val_loss = float('inf')
        if (self.checkpoint):
            start_epoch = self.checkpoint.get('current_epoch', 0)
        else:
            start_epoch = 0
        train_start = datetime.now()
        print(f"训练开始时间: {train_start.strftime('%Y-%m-%d %H:%M:%S')}")
        # 开始训练
        for epoch in range(start_epoch, epochs):
            epoch_start_time = datetime.now()
            # 训练阶段
            self.clip_model.eval()  # 冻结CLIP模型
            self.classifier.train()
            self.process_projector.train()
            self.defect_feature_projector.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                images, process_texts, defect_texts, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                with autocast():
                    # 获取图像特征
                    with torch.no_grad():
                        image_features = self.clip_model.get_image_features(images)
                        image_features /= image_features.norm(dim=-1, keepdim=True)

                        # 获取文本特征（工艺参数+缺陷特征）
                        process_inputs = self.clip_processor.tokenizer(
                            process_texts, return_tensors="pt", padding=True, truncation=True, max_length=77
                        ).to(self.device)
                        true_process_feats = self.clip_model.get_text_features(**process_inputs)
                        true_process_feats /= true_process_feats.norm(dim=-1, keepdim=True)

                        defect_inputs = self.clip_processor.tokenizer(
                            defect_texts, return_tensors="pt", padding=True, truncation=True, max_length=77
                        ).to(self.device)
                        true_defect_feats = self.clip_model.get_text_features(**defect_inputs)
                        true_defect_feats /= true_defect_feats.norm(dim=-1, keepdim=True)

                    # 模型前向传播
                    defect_logits, process_proj, defect_feat_proj = self.forward(image_features)

                    # 计算多任务损失
                    loss = self.cross_entropy_loss(defect_logits, labels)  # 缺陷类型损失
                    loss_process = self.mse_loss(process_proj, true_process_feats)  # 工艺匹配损失
                    loss_defect_feat = self.mse_loss(defect_feat_proj, true_defect_feats)  # 缺陷特征匹配损失

                    # 总损失（可调整权重）
                    total_loss_batch = loss + 0.5 * loss_process + 0.5 * loss_defect_feat

                # 反向传播
                scaler.scale(total_loss_batch).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # 统计精度
                _, predicted = torch.max(defect_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                train_loss += total_loss_batch.item()


            train_acc = correct / total
            train_loss = train_loss / total

            # 验证阶段
            self.classifier.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    images, process_texts, defect_texts, labels = batch
                    images = images.to(self.device)
                    # process_texts = process_texts.to(self.device)
                    # defect_texts = defect_texts.to(self.device)
                    labels = labels.to(self.device)

                    with autocast():
                        # 获取图像特征
                        image_features = self.clip_model.get_image_features(images)
                        image_features /= image_features.norm(dim=-1, keepdim=True)

                        # 获取文本特征
                        process_inputs = self.clip_processor.tokenizer(
                            process_texts, return_tensors="pt", padding=True, truncation=True, max_length=77
                        ).to(self.device)
                        true_process_feats = self.clip_model.get_text_features(**process_inputs)
                        true_process_feats /= true_process_feats.norm(dim=-1, keepdim=True)

                        defect_inputs = self.clip_processor.tokenizer(
                            defect_texts, return_tensors="pt", padding=True, truncation=True, max_length=77
                        ).to(self.device)
                        true_defect_feats = self.clip_model.get_text_features(**defect_inputs)
                        true_defect_feats /= true_defect_feats.norm(dim=-1, keepdim=True)

                        defect_logits, process_proj, defect_feat_proj = self.forward(image_features)

                        loss = self.cross_entropy_loss(defect_logits, labels)  # 缺陷类型损失
                        loss_process = self.mse_loss(process_proj, true_process_feats)  # 工艺匹配损失
                        loss_defect_feat = self.mse_loss(defect_feat_proj, true_defect_feats)  # 缺陷特征匹配损失

                        # 总损失（可调整权重）
                        total_loss_batch = loss + 0.5 * loss_process + 0.5 * loss_defect_feat

                    _, predicted = torch.max(defect_logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_loss += total_loss_batch.item()

            val_acc = correct / total
            val_loss = val_loss / total
            # 学习率调整
            self.scheduler.step(val_acc)

            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)

            # 保留原有最佳模型保存逻辑
            if val_acc > best_val_accuracy or (val_loss < best_val_loss * 0.98):
                best_val_accuracy = max(val_acc, best_val_accuracy)
                best_val_loss = min(val_loss, best_val_loss)
                print(f"发现最佳模型！ 准确率：{best_val_accuracy * 100:.2f}%, 损失：{best_val_loss:.6f}")
                self.save_model(config['clip_tune_model_file_name'], epoch)

            # 保留原有打印逻辑
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch + 1}/{epochs}] '
                  f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc * 100:.2f}% | '
                  f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc * 100:.2f}% | '
                  f'学习率: {current_lr: .6f} | '
                  f'训练耗时: {(datetime.now() - epoch_start_time).total_seconds():.3f}s | '
                  f'早停计数: {self.early_stop.counter}/{self.early_stop.patience}')

            # 早停判定
            if self.early_stop(val_acc, val_loss):
                print(f"\n早停触发! 连续{self.early_stop.patience}个epoch没有改进")
                break

            gc.collect()  # 清理CPU内存
            torch.cuda.empty_cache()  # 清理未使用的GPU缓存

        train_end = datetime.now()
        print("训练结束时间：", train_end.strftime('%Y-%m-%d %H:%M:%S'))
        print(f"\n全部训练完成! 总时间: {(train_end - train_start).total_seconds():.3f}s")
        print(f"最佳准确率: {best_val_accuracy * 100:.2f}%")
        print(f"最佳损失: {best_val_loss:.4f}")

        # 加载最佳模型
        self.load_model(self.clip_model_file_path)
        return self.training_history

    def evaluate(self, data_path, split='test'):
        """在测试集上评估CLIP模型"""
        test_patches, process_texts, defect_texts, test_labels = self.load_clip_dataset(data_path, split)
        test_dataset = ClipDataset(test_patches, process_texts, defect_texts, test_labels, self.clip_processor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['clip_batch_size'],
            shuffle=False,
            num_workers=config['clip_num_of_workers']
        )

        self.classifier.eval()
        all_preds = []
        all_processes = []
        all_defect_featrues = []
        all_labels = []
        results = []  # 存储包含详细信息的结果

        with torch.no_grad():
            batch_idx = 0  # 手动维护批次索引
            for batch in test_loader:
                images,  process_texts, defect_texts, labels = batch
                images = images.to(self.device)
                # process_texts = process_texts.to(self.device)
                # defect_texts = defect_texts.to(self.device)
                labels = labels.to(self.device)

                image_features = self.clip_model.get_image_features(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # 获取文本特征（工艺参数+缺陷特征）
                process_inputs = self.clip_processor.tokenizer(
                    process_texts, return_tensors="pt", padding=True, truncation=True, max_length=77
                ).to(self.device)
                true_process_feats = self.clip_model.get_text_features(**process_inputs)
                true_process_feats /= true_process_feats.norm(dim=-1, keepdim=True)

                defect_inputs = self.clip_processor.tokenizer(
                    defect_texts, return_tensors="pt", padding=True, truncation=True, max_length=77
                ).to(self.device)
                true_defect_feats = self.clip_model.get_text_features(**defect_inputs)
                true_defect_feats /= true_defect_feats.norm(dim=-1, keepdim=True)

                # 模型前向传播
                defect_logits, process_proj, defect_feat_proj = self.forward(image_features)
                _, predicted = torch.max(defect_logits.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_processes.extend(process_proj.cpu().numpy())
                all_defect_featrues.extend(defect_feat_proj.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                batch_process_texts = process_texts[batch_idx * config['clip_batch_size']: (batch_idx + 1) * config['clip_batch_size']]
                batch_defect_texts = defect_texts[batch_idx * config['clip_batch_size']: (batch_idx + 1) * config['clip_batch_size']]
                for pred, process_proj, defect_proj, true_label, true_process, true_defect in zip(
                        predicted.cpu().numpy(),
                        process_proj.cpu().numpy(),
                        defect_feat_proj.cpu().numpy(),
                        labels.cpu().numpy(),
                        batch_process_texts,
                        batch_defect_texts
                ):
                    results.append({
                        "预测类别": self.defect_classes[pred],
                        "真实类别": self.defect_classes[true_label],
                        "预测工艺（向量）": process_proj.tolist(),
                        "预测缺陷（向量）": defect_proj.tolist(),
                        "真实工艺": true_process,  # 显示真实工艺文本，而非ID
                        "真实缺陷": true_defect  # 显示真实缺陷文本，而非ID
                    })

                batch_idx += 1

        # 生成分类报告
        print(f"\nCLIP{split}集分类报告:")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=self.defect_classes
        ))

        print("\nCLIP混淆矩阵:")
        print(confusion_matrix(all_labels, all_preds))

        # 绘制混淆矩阵
        self.visualize_confusion_matrix(all_labels, all_preds, split)

        return all_labels, all_preds

    def visualize_training(self):
        """可视化CLIP训练过程"""
        if not self.training_history['train_loss']:
            print("没有训练历史数据")
            return

        plt.figure(figsize=(15, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_loss'], label='训练损失', linewidth=2)
        plt.plot(self.training_history['val_loss'], label='验证损失', linewidth=2)
        plt.title('CLIP训练损失', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['train_acc'], label='训练准确率', linewidth=2)
        plt.plot(self.training_history['val_acc'], label='验证准确率', linewidth=2)
        plt.title('CLIP准确率', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{config['training_name']}_clip_training_curves.png", dpi=300)
        plt.show()

    def visualize_confusion_matrix(self, labels, preds, split):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.defect_classes,
            yticklabels=self.defect_classes
        )
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.title(f'CLIP模型{split}集混淆矩阵')
        plt.tight_layout()
        plt.savefig(f"{config['training_name']}_clip_confusion_matrix_{split}.png", dpi=300)
        plt.show()

    def predict(self, image_path, top_k=3):
        """预测单张图像的缺陷，返回包含详多模态特征细信息的结果"""
        self.classifier.eval()
        self.process_projector.eval()
        self.defect_feature_projector.eval()

        # 提取缺陷区域和文本信息(缺陷、工艺）
        patch, process_text, defect_text = self.extract_defect_patches(image_path)
        if not patch:
            return None, None, None

        # 处理所有patch并取多数投票
        prediction = None
        predicted_process = None
        predicted_defect = None

        with torch.no_grad():
            # 预处理图像
            # image_inputs = self.clip_processor(images=patch, return_tensors="pt")['pixel_values'].to(self.device)
            image_inputs = self.clip_processor(images=patch, return_tensors="pt")
            # 将字典中的所有张量移动到设备（仅移动，不改变结构）
            for key in image_inputs:
                image_inputs[key] = image_inputs[key].to(self.device)

            image_features = self.clip_model.get_image_features(**image_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 投影到文本空间
            process_proj = self.process_projector(image_features)
            defect_feat_proj = self.defect_feature_projector(image_features)

            # 1. 匹配工艺参数（从训练集工艺文本库中找最相似）
            process_similarity = torch.matmul(process_proj, self.process_text_features.T)  # (1, N)
            top_process_idx = torch.topk(process_similarity, k=top_k, dim=1).indices.squeeze().tolist()
            # 处理top_k=1时返回 scalar 的情况
            if isinstance(top_process_idx, int):
                top_process_idx = [top_process_idx]

            # 2. 匹配缺陷特征（从训练集缺陷文本库中找最相似）
            defect_similarity = torch.matmul(defect_feat_proj, self.defect_text_features.T)  # (1, N)
            top_defect_idx = torch.topk(defect_similarity, k=top_k, dim=1).indices.squeeze().tolist()
            # 处理top_k=1时返回 scalar 的情况
            if isinstance(top_process_idx, int):
                top_process_idx = [top_process_idx]


            # 3. 获取训练集中对应的文本（需提前缓存训练集文本列表）
            # （这里假设已缓存train_process_texts和train_defect_texts）
            predicted_process = [self.train_process_texts[i] for i in top_process_idx]
            predicted_defect = [self.train_defect_texts[i] for i in top_defect_idx]

            # 分类
            defect_logits, process_proj, defect_feat_proj = self.forward(image_features)
            _, predicted = torch.max(defect_logits.data, 1)
            prediction = predicted.item()

            # 计算相似度分数并排序
            process_similarity_scores = torch.matmul(process_proj, self.process_text_features.T).squeeze().cpu().numpy()
            defect_similarity_scores = torch.matmul(defect_feat_proj,
                                                    self.defect_text_features.T).squeeze().cpu().numpy()

            # 按分数排序并返回top_k结果
            process_sorted = sorted(zip(predicted_process, process_similarity_scores), key=lambda x: x[1], reverse=True)
            defect_sorted = sorted(zip(predicted_defect, defect_similarity_scores), key=lambda x: x[1], reverse=True)

            predicted_process = [f"工艺: {text} (相似度: {score:.3f})" for text, score in process_sorted]
            predicted_defect = [f"缺陷: {text} (相似度: {score:.3f})" for text, score in defect_sorted]

        return prediction, predicted_process, predicted_defect




class WaferDefectSystem:
    """整合YOLO和CLIP模型的晶圆缺陷检测系统"""

    def __init__(self, yolo_model_path=None, clip_model_path=None):
        self.device = device
        self.yolo_detector = YOLODefectDetector(self.device)
        self.clip_classifier = None

        # 如果提供了模型路径，则加载已有模型
        if yolo_model_path and os.path.exists(yolo_model_path):
            self.yolo_detector.model = YOLO(yolo_model_path)
            print(f"已加载YOLO模型: {yolo_model_path}")

        if clip_model_path and os.path.exists(clip_model_path):
            self.clip_classifier = CLIPDefectClassifier(
                self.device,
                config['clip_model_name'],
                yolo_model_path=yolo_model_path,
                clip_model_file_path=clip_model_path
            )

    def train_yolo(self):
        """训练YOLO模型"""
        # 准备YOLO数据集
        yolo_dataloader = YOLODataLoader(config['wafer_data_dir'], config['yolo_data_dir'])
        yolo_dataloader.create_yolo_annotations()

        # 训练YOLO模型
        data_yaml = os.path.join(config['yolo_data_dir'], 'wafer_dataset.yaml')
        print("开始训练YOLO模型...")
        results = self.yolo_detector.train(
            data_yaml,
            epochs=config['yolo_epochs'],
            img_size=config['yolo_img_size'],
            batch_size=config['yolo_batch_size'],
            learning_rate=config['yolo_learning_rate'],
            patience=config['yolo_patience'],
            num_of_workers=config['yolo_num_of_workers'],
            enhance_mosaic=config['yolo_enhance_mosaic'],
            enhance_hsv_h=config['yolo_enhance_hsv_h'],
            enhance_hsv_s=config['yolo_enhance_hsv_s'],
            enhance_hsv_v=config['yolo_enhance_hsv_v']
        )

        # 可视化训练结果
        self.yolo_detector.visualize_training()

        # 获取实际的模型保存路径 - 从训练结果中提取
        yolo_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        print(f"YOLO模型训练完成，最佳模型保存于: {yolo_model_path}")

        return yolo_model_path

    def train_clip(self, yolo_model_path):
        """训练CLIP分类器"""
        # 初始化CLIP分类器
        self.clip_classifier = CLIPDefectClassifier(
            self.device,
            config['clip_model_name'],
            yolo_model_path=yolo_model_path,
            learning_rate=config['clip_learning_rate'],
            lr_reduce_factor=config['clip_lr_reduce_factor'],
            lr_reduce_patience=config['clip_lr_reduce_patience'],
            weight_decay=config['clip_weight_decay'],
            early_stop_patience=config['clip_early_stop_patience']
        )

        # 微调CLIP模型
        print("开始微调CLIP模型...")
        self.clip_classifier.fine_tune(
            config['wafer_data_dir'],
            epochs=config['clip_epochs'],
            batch_size=config['clip_batch_size']
        )

        # 可视化训练结果
        self.clip_classifier.visualize_training()

        return config['clip_tune_model_file_name']

    def evaluate_system(self):
        """评估整个系统性能"""
        if not self.yolo_detector or not self.clip_classifier:
            print("请先训练或加载YOLO和CLIP模型")
            return

        # 评估YOLO模型
        data_yaml = os.path.join(config['yolo_data_dir'], 'wafer_dataset.yaml')
        self.yolo_detector.evaluate(data_yaml, split='test')

        # 评估CLIP模型
        self.clip_classifier.evaluate(config['wafer_data_dir'], split='test')

    def detect_and_classify(self, image_path):
        """检测并分类晶圆缺陷"""
        if not self.yolo_detector or not self.clip_classifier:
            print("请先训练或加载YOLO和CLIP模型")
            return None

        # 使用YOLO检测缺陷区域
        defects = self.yolo_detector.detect_defects(image_path)
        print(f"检测到{len(defects)}个缺陷区域")

        # 使用CLIP分类整体缺陷类型
        prediction, predicted_process, predicted_defect = self.clip_classifier.predict(image_path)
        if prediction is not None:
            class_name = self.clip_classifier.defect_classes[prediction]
            print(f"检测缺陷分类: {class_name}")
            print(f"检测缺陷特征: {predicted_defect}")
            print(f"检测缺陷工艺: {predicted_process}")

class EarlyStopping:
    """早停机制类"""

    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc, val_loss):
        score = val_acc  # 使用准确率作为主要判断指标

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop



def main():
    """主函数：训练模型并评估性能"""
    # 初始化系统
    system = WaferDefectSystem()

    # 分析数据集分布
    print("分析数据集分布...")
    wafer_dataset = WaferDataset(config['wafer_data_dir'])
    train_counts, test_counts, valid_counts = wafer_dataset.analyze_dataset()
    wafer_dataset.visualize_distribution(train_counts, test_counts, valid_counts)

    # 训练YOLO模型
    yolo_model_path = system.train_yolo()

    # 训练CLIP模型
    clip_model_path = system.train_clip(yolo_model_path)

    # 评估系统
    system = WaferDefectSystem(yolo_model_path, clip_model_path)
    system.evaluate_system()

    # 示例预测
    test_image_dir = os.path.join(config['wafer_data_dir'], 'test', 'images')
    if os.path.exists(test_image_dir) and os.listdir(test_image_dir):
        sample_image = os.path.join(test_image_dir, os.listdir(test_image_dir)[0])
        print(f"\n对样本图像 {sample_image} 进行预测:")
        system.detect_and_classify(sample_image)



if __name__ == "__main__":
    main()
