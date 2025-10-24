# -*- coding: utf-8 -*-
import os, io, base64
from typing import List
from time import perf_counter

from fastapi import FastAPI, HTTPException, Response, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import BaseModel
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
from prometheus_client.registry import CollectorRegistry
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_matcher import FeatureMatcher


# ===================== 配置 =====================
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cpu")
)

APP_TITLE = os.getenv("APP_TITLE", "wafermap-defect-detection-torchscript")
APP_VERSION = os.getenv("APP_VERSION", "1.0")
APP_DIR = Path(__file__).parent
MODEL_PATH = os.getenv("MODEL_PATH", str(APP_DIR / "torchscript" / "wafermap_defect_ts_model.pt"))
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "../../models/clip-vit-base-patch32")

# JWT：默认 HS256；生产可切 RS256 并提供公钥
ALGORITHM = os.getenv("JWT_ALG", "HS256")            # HS256 / RS256
JWT_ISSUER = os.getenv("JWT_ISS", "admin-auth")
JWT_AUDIENCE = os.getenv("JWT_AUD", "admin-api")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")                 # HS256 时必须
JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY")         # RS256 时必须（PEM 公钥字符串）

config = {
    'defect_classes': ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none'],
    'process_descriptions' : [
            "高温氧化工艺，温度850-950℃，压力45-55Pa，材料单晶硅，蚀刻时间50-70s，旋转速度300-500rpm",
            "化学气相沉积工艺，温度800-900℃，压力50-60Pa，材料多晶硅，蚀刻时间60-80s，旋转不均匀度5-15%",
            "离子注入工艺，温度880-980℃，压力40-50Pa，材料硅锗，蚀刻时间40-60s，夹具压力20-30N",
            "光刻工艺，温度900-1000℃，压力35-45Pa，材料单晶硅，蚀刻时间70-90s，边缘蚀刻速率1.2-1.8μm/min",
            "蚀刻工艺，温度820-920℃，压力55-65Pa，材料多晶硅，蚀刻时间55-75s，洁净度等级Class 100-1000",
            "金属化工艺，温度780-880℃，压力60-70Pa，材料硅外延片，蚀刻时间80-100s，薄膜均匀度3-7%",
            "清洗工艺，温度850-950℃，压力45-55Pa，材料单晶硅，蚀刻时间50-70s，良率95-99%",
            "退火工艺，温度800-900℃，压力50-60Pa，材料多晶硅，蚀刻时间60-80s，颗粒密度1-5个/cm²",
            "机械抛光工艺，温度880-980℃，压力40-50Pa，材料硅锗，蚀刻时间40-60s，机械手速度50-100mm/s"
    ],
    'defect_descriptions' : [
            "中心区域密集点缺陷，直径约2-5mm，呈圆形分布，边缘无异常，可能与中心加热不均相关",
            "环形缺陷图案，内直径10-15mm，外直径20-25mm，环宽2-3mm，环内无缺陷，与旋转速度波动相关",
            "边缘局部区域缺陷，位于晶圆边缘向内5mm范围内，呈点状聚集，可能与夹具压力不均相关",
            "晶圆边缘环形缺陷带，宽度3-5mm，沿整个圆周分布，可能与边缘刻蚀速率异常相关",
            "局部区域缺陷，面积约5-15mm²，形状不规则，分布位置随机，可能与洁净度不足相关",
            "近满片缺陷，覆盖晶圆70-90%区域，缺陷密度由中心向边缘递减，可能由工艺参数严重偏离导致",
            "无明显缺陷，表面平整，符合工艺标准，各项指标正常，良品晶圆",
            "随机分布的点缺陷，密度1-5个/cm²，大小不一，无明显规律，可能与颗粒污染相关",
            "单条划痕缺陷，长度10-20mm，宽度0.1-0.3mm，呈直线状，方向随机，可能与机械臂移动异常相关"
    ]
}

# ===================== 方法 & 类定义 =====================
class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        device = str(DEVICE)
        start = perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            dur = perf_counter() - start
            LAT.labels(route=route_path, device=device).observe(dur)
            REQ.labels(route=route_path, status=status, device=device).inc()

class PredictReq(BaseModel):
    image_b64: str  # base64 PNG/JPG

class PredictResp(BaseModel):
    bbox: List[float]
    defect_class: str
    confidence: float
    process_description: str  # 工艺参数描述
    process_similarity: float  # 工艺描述相似度
    defect_description: str  # 缺陷特征描述
    defect_similarity: float  # 缺陷描述相似度

security = HTTPBearer(auto_error=True)

def _get_verify_key() -> str:
    if ALGORITHM == "HS256":
        if not JWT_SECRET:
            raise RuntimeError("缺少 JWT_SECRET（HS256）")
        return JWT_SECRET
    elif ALGORITHM == "RS256":
        if not JWT_PUBLIC_KEY:
            raise RuntimeError("缺少 JWT_PUBLIC_KEY（RS256 公钥 PEM）")
        return JWT_PUBLIC_KEY
    else:
        raise RuntimeError(f"不支持算法: {ALGORITHM}")

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        # 注意：python-jose 不支持 leeway 位置参数；这里使用默认严格校验
        claims = jwt.decode(
            token,
            _get_verify_key(),
            algorithms=[ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={
                "verify_signature": True,
                "verify_aud": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_nbf": True,
            },
        )
        return claims
    except JWTError as e:
        # 统一返回 401，便于前端识别鉴权失败
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

def preprocess_image(image_bytes):
    """图像预处理：转为YOLO输入格式"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((416, 416))  # 与导出时的输入尺寸一致
    image_np = np.array(image) / 255.0  # 归一化
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0)
    return image_tensor

# ===================== 加载 TorchScript 模型 =====================
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"MODEL_PATH 不存在: {MODEL_PATH}")
model = torch.jit.load(MODEL_PATH, map_location=DEVICE).eval().to(DEVICE)

# 注意：这里需要实际的CLIP模型来编码文本
# 由于TorchScript模型不包含原始CLIP模型，我们需要单独加载
from transformers import CLIPProcessor, CLIPModel

clip_model_original = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
feature_matcher = FeatureMatcher(config['process_descriptions'], config['defect_descriptions'], clip_model_original, clip_processor, DEVICE)
print("✓ 特征匹配器初始化成功")

# ===================== Prometheus 指标 =====================
custom_registry = CollectorRegistry()
REQ = Counter(
    "requests_total",  # 1. 指标名称（全局唯一标识）
    "Total inference requests",  # 2. 指标描述（说明指标含义，便于理解）
    labelnames=["route", "status", "device"],  # 3. 标签（接口路由、HTTP状态码、服务运行设备）
    registry=custom_registry  # 4. 指标注册表（管理指标的容器）
)
LAT = Histogram(
    "request_latency_seconds",  # 1. 指标名称
    "Latency of inference",  # 2. 指标描述
    labelnames=["route", "device"],  # 3. 标签（接口路由、HTTP状态码、服务运行设备）
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),  # 4. 桶（将 “延迟” 按指定区间（桶）分箱，统计每个区间的请求数量）
    registry=custom_registry  # 5. 指标注册表
)

# ===================== 应用与中间件 =====================
app = FastAPI(title=APP_TITLE, version=APP_VERSION)
app.add_middleware(PrometheusMiddleware)

# ===================== 路由 =====================
@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": str(DEVICE), "model_path": MODEL_PATH}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(custom_registry), media_type=CONTENT_TYPE_LATEST)

@app.get("/whoami")
def whoami(claims: dict = Depends(verify_jwt)):
    # 便于验证鉴权链路
    return {"sub": claims.get("sub"), "role": claims.get("role")}

@app.post("/predict", response_model=PredictResp)
def predict(req: PredictReq, claims: dict = Depends(verify_jwt)):
    try:
        img_bytes = base64.b64decode(req.image_b64);
        x = preprocess_image(img_bytes).to(DEVICE)  # [1,3,416,416]
        # 推理
        with torch.no_grad():
            result = model(x)
        print(f"result形状: {result.shape}, 类型: {type(result)}")
        print(f"result内容: {result}")

        # 解析张量输出 [1, 1030]
        # 0-3: bbox, 4: class_id, 5: confidence
        # 6-517: process_feature(512), 518-1029: defect_feature(512)

        # 基础检测结果
        bbox = result[0, :4].tolist()
        class_id = int(result[0, 4].item())
        confidence = result[0, 5].item()

        # 特征向量
        process_feature_tensor = result[0, 6:518].unsqueeze(0)  # [1, 512]
        defect_feature_tensor = result[0, 518:1030].unsqueeze(0)  # [1, 512]

        # 获取对应的缺陷类型
        defect_class = config['defect_classes'][class_id]

        # 特征匹配（如果特征匹配器可用）
        process_description = ""
        process_similarity = 0.0
        defect_description = ""
        defect_similarity = 0.0

        if feature_matcher is not None:
            try:
                process_description, process_similarity, defect_description, defect_similarity = feature_matcher.match_features(process_feature_tensor, defect_feature_tensor)
                print(f"特征匹配成功: 工艺相似度={process_similarity:.3f}, 缺陷相似度={defect_similarity:.3f}")
            except Exception as e:
                print(f"特征匹配失败: {e}")

        # 构造响应
        resp = PredictResp(
            bbox=bbox,
            defect_class=defect_class,
            confidence=confidence,
            process_description=process_description,
            process_similarity=process_similarity,
            defect_description=defect_description,
            defect_similarity=defect_similarity
        )

        return resp
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 配置端口、主机、工作进程数
    uvicorn.run(
        "app:app",  # 指定应用路径
        host="0.0.0.0",
        port=8802,       # 直接在代码中设置端口
        workers=1,       # 工作进程数
        reload=False,      # 开发环境热重载（生产环境关闭）
    )