
import torch


# ===================== 特征匹配工具类 =====================
class FeatureMatcher:
    def __init__(self, process_descriptions, defect_descriptions, clip_model, clip_processor, device):
        self.process_descriptions = process_descriptions
        self.defect_descriptions = defect_descriptions
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device

        # 预编码文本描述的特征向量
        self.process_text_embeddings = self._encode_texts(process_descriptions)
        self.defect_text_embeddings = self._encode_texts(defect_descriptions)

    def _encode_texts(self, texts):
        """将文本描述编码为特征向量"""
        with torch.no_grad():
            inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def feature_to_text(self, feature, text_embeddings, texts):
        """将特征向量转换为最相似的文本描述"""
        # 计算相似度
        similarities = torch.matmul(feature, text_embeddings.T)
        # 获取最相似的文本索引
        best_match_idx = torch.argmax(similarities, dim=-1)
        best_similarity = similarities[0, best_match_idx].item()

        # 返回文本和相似度
        return texts[best_match_idx.item()], best_similarity

    def match_features(self, process_feature, defect_feature):
        """匹配工艺和缺陷特征"""
        process_text, process_similarity = self.feature_to_text(
            process_feature, self.process_text_embeddings, self.process_descriptions
        )
        defect_text, defect_similarity = self.feature_to_text(
            defect_feature, self.defect_text_embeddings, self.defect_descriptions
        )

        return process_text, process_similarity, defect_text, defect_similarity