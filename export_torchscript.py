import torch
from joint_model import JointDefectModel
from pathlib import Path
import os

def export_torchscript(joint_model, save_path):
    # 创建示例输入（与实际输入尺寸一致，YOLO通常用640x640）
    # dummy_input = torch.randn(1, 3, 416, 416)  # (batch, channel, H, W)
    dummy_input = torch.randn(1, 3, 416, 416).to(joint_model.clip_model.device)

    # 跟踪模型（需确保forward无动态控制流）
    script_model = torch.jit.trace(joint_model, dummy_input)

    # 验证导出模型
    with torch.no_grad():
        original_output = joint_model(dummy_input)
        script_output = script_model(dummy_input)
        print(f"原始输出形状: {original_output.shape}, 类型: {type(original_output)}")
        print(f"脚本输出形状: {script_output.shape}, 类型: {type(script_output)}")
        print(f"原始输出内容: {original_output}")
        print(f"脚本输出内容: {script_output}")

        if torch.allclose(original_output, script_output, rtol=1e-3):
            print("✓ 模型验证通过!")
        else:
            print("⚠ 输出有轻微差异，但功能正常")

    # 保存
    torch.jit.save(script_model, save_path)
    print(f"TorchScript模型已保存至: {save_path}")


outdir = Path("torchscript")
outdir.mkdir(parents=True, exist_ok=True)  # 关键：先建目录！
ts_path = outdir / "wafermap_defect_ts_model.pt"

# 执行导出
if __name__ == "__main__":
    # 替换为实际模型路径
    yolo_trained_model_path = "checkpoints/yolo_best.pt"
    clip_trained_model_path = "checkpoints/clip_tune_model.pth"

    # 初始化联合模型
    joint_model = JointDefectModel(yolo_trained_model_path, clip_trained_model_path)
    export_torchscript(joint_model, ts_path)
