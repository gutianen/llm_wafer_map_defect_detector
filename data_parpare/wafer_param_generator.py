import os
import random
from tqdm import tqdm  # 移除未使用的shutil导入

# 配置路径
root_dir = "../dataset/57_wafer_detection"  # 数据集根目录
classes = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'None', 'Random', 'Scratch']
class_id_map = {cls: idx for idx, cls in enumerate(classes)}

# 1. 工艺参数范围（按缺陷类型）
process_params_ranges = {
    "Center": {  # 中心区域缺陷对应的工艺参数范围
        "temperature": (850, 950),  # 温度范围，单位为摄氏度（℃）
        "pressure": (45, 55),        # 压力范围，单位可能为帕斯卡（Pa）或其他压力单位
        "material": ["单晶硅"],      # 所使用的材料为单晶硅
        "etch_time": (50, 70),       # 蚀刻时间范围，单位通常为秒（s）
        "rotation_speed": (300, 500) # 旋转速度范围，单位通常为转每分钟（rpm）
    },
    "Donut": {  # 环形缺陷对应的工艺参数范围
        "temperature": (800, 900),   # 温度范围，单位为摄氏度（℃）
        "pressure": (50, 60),        # 压力范围，单位可能为帕斯卡（Pa）或其他压力单位
        "material": ["多晶硅"],      # 所使用的材料为多晶硅
        "etch_time": (60, 80),       # 蚀刻时间范围，单位通常为秒（s）
        "rotation_unevenness": (5, 15) # 旋转不均匀度范围，单位为百分比（%）
    },
    "Edge-Loc": {  # 边缘局部缺陷对应的工艺参数范围
        "temperature": (880, 980),   # 温度范围，单位为摄氏度（℃）
        "pressure": (40, 50),        # 压力范围，单位可能为帕斯卡（Pa）或其他压力单位
        "material": ["硅锗"],        # 所使用的材料为硅锗
        "etch_time": (40, 60),       # 蚀刻时间范围，单位通常为秒（s）
        "clamp_pressure": (20, 30)   # 夹具压力范围，单位为牛顿（N）
    },
    "Edge-Ring": {  # 边缘环形缺陷对应的工艺参数范围
        "temperature": (900, 1000),  # 温度范围，单位为摄氏度（℃）
        "pressure": (35, 45),        # 压力范围，单位可能为帕斯卡（Pa）或其他压力单位
        "material": ["单晶硅"],      # 所使用的材料为单晶硅
        "etch_time": (70, 90),       # 蚀刻时间范围，单位通常为秒（s）
        "edge_etch_rate": (1.2, 1.8) # 边缘蚀刻速率范围，单位为微米每分钟（μm/min）
    },
    "Loc": {  # 局部缺陷对应的工艺参数范围
        "temperature": (820, 920),   # 温度范围，单位为摄氏度（℃）
        "pressure": (55, 65),        # 压力范围，单位可能为帕斯卡（Pa）或其他压力单位
        "material": ["多晶硅"],      # 所使用的材料为多晶硅
        "etch_time": (55, 75),       # 蚀刻时间范围，单位通常为秒（s）
        "cleanliness_class": ["Class 100", "Class 1000"] # 洁净度等级（Class 100：每立方英尺空气中粒径≥0.5μm的粒子数≤100）
    },
    "Near-full": {  # 近满缺陷对应的工艺参数范围
        "temperature": (780, 880),   # 温度范围，单位为摄氏度（℃）
        "pressure": (60, 70),        # 压力范围，单位可能为帕斯卡（Pa）或其他压力单位
        "material": ["硅外延片"],    # 所使用的材料为硅外延片
        "etch_time": (80, 100),      # 蚀刻时间范围，单位通常为秒（s）
        "film_uniformity": (3, 7)    # 薄膜均匀度范围，单位为百分比（%）
    },
    "None": {  # 无缺陷对应的工艺参数范围
        "temperature": (850, 950),   # 温度范围，单位为摄氏度（℃）
        "pressure": (45, 55),        # 压力范围，单位可能为帕斯卡（Pa）或其他压力单位
        "material": ["单晶硅"],      # 所使用的材料为单晶硅
        "etch_time": (50, 70),       # 蚀刻时间范围，单位通常为秒（s）
        "yield_rate": (95, 99)       # 良率范围，单位为百分比（%）
    },
    "Random": {  # 随机缺陷对应的工艺参数范围
        "temperature": (800, 900),   # 温度范围，单位为摄氏度（℃）
        "pressure": (50, 60),        # 压力范围，单位可能为帕斯卡（Pa）或其他压力单位
        "material": ["多晶硅"],      # 所使用的材料为多晶硅
        "etch_time": (60, 80),       # 蚀刻时间范围，单位通常为秒（s）
        "particle_density": (1, 5)   # 颗粒密度范围，单位为每平方厘米粒子数（个/cm²）
    },
    "Scratch": {  # 划痕缺陷对应的工艺参数范围
        "temperature": (880, 980),   # 温度范围，单位为摄氏度（℃）
        "pressure": (40, 50),        # 压力范围，单位可能为帕斯卡（Pa）或其他压力单位
        "material": ["硅锗"],        # 所使用的材料为硅锗
        "etch_time": (40, 60),       # 蚀刻时间范围，单位通常为秒（s）
        "robot_speed": (50, 100)     # 机械手速度范围，单位为毫米每秒（mm/s）
    }
}


# 2. 缺陷特征描述（按类型）
defect_features = {
    "Center": [
        "中心区域密集点缺陷，直径约2-5mm，呈圆形分布，边缘无异常",
        "晶圆中心存在聚集性缺陷，可能由中心加热不均导致，密度约10-20个/mm²",
        "中心区域出现成片缺陷，呈不规则块状，面积约10-30mm²",
        "圆心附近检测到密集缺陷群，可能与旋转轴偏移相关，边缘清晰",
        "中心20%区域内分布大量微小缺陷，直径<0.5mm，外围无明显缺陷"
    ],
    "Donut": [
        "环形缺陷图案，内直径10-15mm，外直径20-25mm，环宽2-3mm，环内无缺陷",
        "中间无缺陷、外围环形分布缺陷，环区域缺陷密度高，可能与旋转速度波动相关",
        "甜甜圈状缺陷，环形区域呈连续分布，宽度均匀，环中心与晶圆中心重合",
        "环形缺陷带，位于距中心10-20mm处，由密集小点组成，环内区域正常",
        "同心圆状缺陷，主环宽度3-5mm，可能伴随1-2个次级细环，与刻蚀时间过长有关"
    ],
    "Edge-Loc": [
        "边缘局部区域缺陷，位于晶圆边缘向内5mm范围内，呈点状聚集",
        "边缘特定位置(约3点钟方向)出现缺陷簇，面积约5-10mm²，形状不规则",
        "晶圆边缘局部凸起区域伴随缺陷，可能与夹具压力不均相关，位置固定",
        "边缘1-2处局部缺陷，呈线性分布，长度5-8mm，深度较浅",
        "边缘局部区域出现密集缺陷点，与夹具接触位置重合，可能由接触污染导致"
    ],
    "Edge-Ring": [
        "晶圆边缘环形缺陷带，宽度3-5mm，沿整个圆周分布，连续性好",
        "距边缘2-5mm处形成闭合环形缺陷，密度均匀，可能与边缘刻蚀速率异常相关",
        "边缘环形缺陷，环内缺陷呈放射状分布，与晶圆旋转方向一致",
        "边缘完整环形缺陷带，伴随轻微颜色差异，可能由边缘曝光不均导致",
        "边缘环形缺陷，宽度随圆周轻微变化(±1mm)，缺陷密度由内向外递增"
    ],
    "Loc": [
        "局部区域缺陷，面积约5-15mm²，形状不规则，分布位置随机",
        "单个局部缺陷簇，包含10-30个小点缺陷，分布集中，周围区域正常",
        "局部块状缺陷，边界清晰，内部缺陷密度高，可能与洁净度不足相关",
        "小范围局部缺陷，呈椭圆形，长轴约8mm，短轴约5mm，位置不固定",
        "局部区域出现混合型缺陷，包含点缺陷和线缺陷，面积约10mm²"
    ],
    "Near-full": [
        "近满片缺陷，覆盖晶圆70-90%区域，缺陷密度由中心向边缘递减",
        "大面积缺陷分布，除边缘5mm外几乎全覆盖，缺陷呈均匀分布",
        "近满片缺陷，伴随明显的区域划分，不同区域缺陷密度差异显著",
        "覆盖80%以上面积的密集缺陷，可能由工艺参数严重偏离标准导致",
        "近满片缺陷，中心区域缺陷密度最高，向边缘逐渐降低，无明显空白区"
    ],
    "None": [
        "无明显缺陷，表面平整，符合工艺标准，各项指标正常",
        "晶圆质量良好，无可见缺陷或异常图案，表面光洁度高",
        "未检测到任何缺陷，边缘光滑，中心区域无异常，符合A级标准",
        "无缺陷特征，表面均匀一致，无污点、划痕或异常纹理",
        "各项检测指标均在合格范围内，无任何缺陷或工艺异常迹象"
    ],
    "Random": [
        "随机分布的点缺陷，密度1-5个/cm²，大小不一，无明显规律",
        "全片随机分布缺陷，数量约20-50个，位置无关联性，可能与颗粒污染相关",
        "随机分布的微小缺陷，直径<0.3mm，分布均匀，无聚集现象",
        "随机分布的混合缺陷，包含点缺陷和短线缺陷，分布无规律",
        "全片随机分布的稀疏缺陷，数量较少(5-15个)，大小和形状各异"
    ],
    "Scratch": [
        "单条划痕缺陷，长度10-20mm，宽度0.1-0.3mm，呈直线状，方向随机",
        "多条平行划痕，间距2-5mm，长度5-15mm，可能与机械臂移动异常相关",
        "交叉划痕缺陷，形成十字或网状，深度较浅，边缘有轻微凸起",
        "弧形划痕，长度15-25mm，位于晶圆边缘区域，可能与传送过程接触导致",
        "细微划痕，长度3-8mm，不易察觉，需放大观察，数量1-3条"
    ]
}


def generate_process_params(cls):
    """生成指定缺陷类型的工艺参数字符串（带单位）"""
    params = process_params_ranges[cls]
    param_str = []
    for k, v in params.items():
        if isinstance(v, tuple):  # 数值范围参数（需加单位）
            val = random.uniform(v[0], v[1])
            # 根据参数类型设置格式和单位
            if k == "temperature":
                param_str.append(f"{k}={val:.1f}℃")
            elif k == "pressure":
                param_str.append(f"{k}={val:.1f}Pa")
            elif k == "edge_etch_rate":
                param_str.append(f"{k}={val:.1f}μm/min")
            elif k in ["rotation_unevenness", "film_uniformity", "yield_rate"]:
                param_str.append(f"{k}={val:.1f}%")
            elif k == "rotation_speed":
                param_str.append(f"{k}={int(val)}rpm")
            elif k == "etch_time":
                param_str.append(f"{k}={int(val)}s")
            elif k == "clamp_pressure":
                param_str.append(f"{k}={int(val)}N")
            elif k == "particle_density":
                param_str.append(f"{k}={val:.1f}个/cm²")
            elif k == "robot_speed":
                param_str.append(f"{k}={int(val)}mm/s")
        else:  # 列表选值参数（如材料、洁净度等级）
            param_str.append(f"{k}={random.choice(v)}")
    return "process_params: " + ", ".join(param_str)


def update_label_files():
    """批量更新label文件，添加工艺参数和缺陷描述（避免重复添加）"""
    for split in ["train", "test", "valid"]:
        label_dir = os.path.join(root_dir, split, "labels")
        if not os.path.exists(label_dir):
            print(f"警告: 未找到{split}集的labels目录，已跳过")
            continue

        # 遍历当前分割的所有label文件
        for label_file in tqdm(os.listdir(label_dir), desc=f"Processing {split} labels"):
            if not label_file.endswith(".txt"):
                continue
            label_path = os.path.join(label_dir, label_file)

            # 读取原有label内容
            with open(label_path, "r") as f:
                lines = f.readlines()

            # 关键：判断是否已存在工艺参数注释（避免重复添加）
            has_process_comment = any(
                line.strip().startswith("# process_params:")
                for line in lines
            )
            if has_process_comment:
                # 已存在注释，跳过并提示
                print(f"跳过: {label_path} 已包含工艺参数注释，无需重复添加")
                continue

            # 1. 提取缺陷类型（核心逻辑：优先从标注获取，无标注则为None）
            cls = "None"  # 默认无缺陷
            if lines and not lines[0].startswith("#"):  # 非注释行视为标注
                try:
                    class_id = int(lines[0].split()[0])
                    if 0 <= class_id < len(classes):
                        cls = classes[class_id]
                except (IndexError, ValueError):
                    # 处理格式异常的标注（如空行、非整数类ID）
                    print(f"警告: {label_path} 标注格式异常，默认按'None'处理")

            # 2. 生成工艺参数和缺陷描述注释
            process_line = f"# {generate_process_params(cls)}\n"
            feature_line = f"# defect_features: {random.choice(defect_features[cls])}\n"

            # 3. 写入新内容（注释行 + 原有标注）
            with open(label_path, "w") as f:
                f.write(process_line)    # 先写工艺参数
                f.write(feature_line)    # 再写缺陷描述
                f.writelines(lines)      # 最后写原有标注
            # 仅在成功添加时打印（减少冗余）
            print(f"完成: {label_path} 已添加工艺参数和缺陷描述")


if __name__ == "__main__":
    update_label_files()
    print("\n" + "="*50)
    print("所有标签文件处理完成！")
    print(f"成功添加工艺参数和缺陷描述的文件：需查看上述'完成'日志")
    print(f"跳过的文件：已包含注释的文件（见'跳过'日志）")
    print("="*50)
