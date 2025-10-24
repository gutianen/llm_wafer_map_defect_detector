# test_predict.py：单独测试 /predict 接口（预测核心逻辑）
import base64
import io
from PIL import Image
import torch
import sys
from pathlib import Path
# 将项目根目录（app.py 所在目录）添加到 Python 路径
project_root = Path(__file__).parent.parent  # __file__ 是当前 test_healthz.py，parent.parent 是项目根目录
sys.path.append(str(project_root))
from fastapi.testclient import TestClient
from jose import jwt
from app import app, JWT_SECRET, ALGORITHM, JWT_ISSUER, JWT_AUDIENCE, DEVICE

from test_public import image_to_base64, generate_test_token

client = TestClient(app)



# ---------------------- 单独测试用例 ----------------------
def test_api_predict():
    """测试：有效Token + 有效图片 → 正常返回预测结果"""
    print("=== 测试有效输入预测 ===")
    # 1. 准备测试数据
    img_b64 = image_to_base64("wafer_image_index_49185.jpg")
    test_token = generate_test_token(sub="predict-test-user", role="predictor")

    # 2. 发送预测请求
    response = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {test_token}"},
        json={"image_b64": img_b64}  # 符合PredictReq的JSON结构
    )

    # 3. 验证结果
    assert response.status_code == 200, f"预期200，实际{response.status_code}"
    data = response.json()

    # 验证返回结构
    assert "defect_class" in data, "返回数据缺少'pred'字段"
    assert "confidence" in data, "返回数据缺少'confidence'字段"
    assert "process_description" in data, "返回数据缺少'process_description'字段"
    assert "process_similarity" in data, "返回数据缺少'process_similarity'字段"
    assert "defect_description" in data, "返回数据缺少'defect_description'字段"
    assert "defect_similarity" in data, "返回数据缺少'defect_similarity'字段"

    print(f"✓ 预测类别：{data['defect_class']}")
    print(f"✓ 预测检测置信度：{data['confidence']}")
    print(f"✓ 预测工艺描述：{data['process_description']}")
    print(f"✓ 预测工艺描述相似度：{data['process_similarity']}")
    print(f"✓ 预测缺陷描述：{data['defect_description']}")
    print(f"✓ 预测缺陷描述相似度：{data['defect_similarity']}")



# 单独执行所有测试
if __name__ == "__main__":
    # 先验证图片转换逻辑（依赖基础）
    # test_image_transform_correctness()
    # 再测试预测接口核心功能
    test_api_predict()
    # test_predict_with_invalid_image()
    # test_predict_without_token()
    print("\n=== /predict 接口测试通过 ===")
