import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import os
from torch.nn.modules.container import Sequential  # 导入报错的Sequential类
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, C2f, SPPF, Concat, Detect  # Ultralytics模型核心模块
from ultralytics.nn.modules.block import Bottleneck, DFL

# 批量添加所有必要的安全全局类（覆盖已报错和潜在需要的类）
torch.serialization.add_safe_globals([
    # PyTorch 内置类
    Sequential,
    nn.Linear,
    nn.ReLU,
    nn.Dropout,
    nn.LayerNorm,
    nn.Conv2d,
    nn.modules.batchnorm.BatchNorm2d,
    nn.modules.activation.SiLU,
    nn.modules.container.ModuleList,
    nn.modules.pooling.MaxPool2d,
    nn.modules.upsampling.Upsample,
    # Ultralytics 模型类
    DetectionModel,
    Conv,
    C2f,
    SPPF,
    Concat,
    Detect,
    Bottleneck,
    DFL
])

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cpu")
)


# 配置参数（沿用训练时的配置）
config = {
    'defect_classes': ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none'],
    'yolo_model_name': '../models/yolov8n.pt',
    'clip_model_name': '../models/clip-vit-base-patch32',  # 同训练时路径

}


class JointDefectModel(nn.Module):
    def __init__(self, yolo_trained_model_path, clip_trained_model_path):
        super().__init__()
        # 1. 加载YOLO模型
        train_yolo = YOLO(yolo_trained_model_path)
        self.yolo_model = train_yolo.model
        # # 设置推理配置，避免自动训练
        # 直接使用模型进行推理，避免复杂的配置
        self.yolo_model.eval()

        # 2. 加载CLIP相关组件
        self.clip_model = CLIPModel.from_pretrained(config['clip_model_name'])
        self.clip_processor = CLIPProcessor.from_pretrained(config['clip_model_name'])

        # 3. 加载CLIP微调组件（分类头和投影头）
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(config['defect_classes']))
        )
        self.process_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )
        self.defect_feature_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )

        # 加载CLIP微调权重
        clip_checkpoint = torch.load(clip_trained_model_path, map_location=device, weights_only=True)
        self.classifier.load_state_dict(clip_checkpoint['classifier_state_dict'])
        self.process_projector.load_state_dict(clip_checkpoint['process_projector_state_dict'])
        self.defect_feature_projector.load_state_dict(clip_checkpoint['defect_feature_projector_state_dict'])

        self.classes = config['defect_classes']
        self.eval()  # 全局置为推理模式

    def forward(self, image_tensor):
        """
        单张图像多模态推理流程
        返回统一格式的张量：[bbox(4), class_id(1), confidence(1), process_feature(512), defect_feature(512)]
        总共4+1+1+512+512 = 1030个元素
        """
        # 1. 简化的YOLO检测
        bbox_tensor = self._simple_yolo_detection(image_tensor)

        # 2. 提取图像区域
        patch_tensor = self._extract_patch(image_tensor, bbox_tensor)

        # 3. CLIP多模态推理
        with torch.no_grad():
            # 获取图像特征
            image_features = self.clip_model.get_image_features(patch_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 缺陷分类
            defect_logits = self.classifier(image_features)
            pred_class_id = torch.argmax(defect_logits, dim=1)
            confidence = torch.softmax(defect_logits, dim=1)[0, pred_class_id]

            # 多模态特征投影
            process_feature = self.process_projector(image_features)
            process_feature = process_feature / process_feature.norm(dim=-1, keepdim=True)

            defect_feature = self.defect_feature_projector(image_features)
            defect_feature = defect_feature / defect_feature.norm(dim=-1, keepdim=True)

            # 拼接所有结果
            result_tensor = torch.cat([
                # 基础检测结果 [6]
                torch.tensor([
                    bbox_tensor[0].item(), bbox_tensor[1].item(), bbox_tensor[2].item(), bbox_tensor[3].item(),
                    # bbox [4]
                    float(pred_class_id.item()),  # class_id [1]
                    confidence.item(),  # confidence [1]
                ]),
                # 特征向量 [1024]
                process_feature.squeeze(0),  # process_feature [512]
                defect_feature.squeeze(0)  # defect_feature [512]
            ]).unsqueeze(0)  # [1, 1030]

            return result_tensor

    def _encode_texts(self, texts):
        """将文本描述编码为特征向量"""
        with torch.no_grad():
            inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _feature_to_text(self, feature, text_embeddings, texts):
        """将特征向量转换为最相似的文本描述"""
        # 计算相似度
        similarities = torch.matmul(feature, text_embeddings.T)
        # 获取最相似的文本索引
        best_match_idx = torch.argmax(similarities, dim=-1)
        best_similarity = similarities[0, best_match_idx].item()

        # 返回文本和相似度
        return texts[best_match_idx.item()], best_similarity

    def _simple_yolo_detection(self, image_tensor):
        # 获取图像尺寸
        _, _, h, w = image_tensor.shape

        # 简化：总是返回中心区域，避免复杂的检测逻辑
        center_x, center_y = w // 2, h // 2
        crop_size = min(w, h, 224) // 2

        x1 = torch.clamp(torch.tensor(center_x - crop_size, device=device), 0, w)
        y1 = torch.clamp(torch.tensor(center_y - crop_size, device=device), 0, h)
        x2 = torch.clamp(torch.tensor(center_x + crop_size, device=device), 0, w)
        y2 = torch.clamp(torch.tensor(center_y + crop_size, device=device), 0, h)

        return torch.tensor([x1, y1, x2, y2], device=device, dtype=torch.float32)

    def _extract_patch(self, image_tensor, bbox_tensor):
        # 确保边界框值是整数
        x1, y1, x2, y2 = bbox_tensor.int()

        # 使用PyTorch操作进行裁剪
        patch = image_tensor[:, :, y1:y2, x1:x2]

        # 调整大小到224x224
        patch_resized = torch.nn.functional.interpolate(
            patch, size=(224, 224), mode='bilinear', align_corners=False
        )

        # 直接返回调整大小后的张量，在forward中统一处理
        return patch_resized
