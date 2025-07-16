import os
import torch
import pytorch_fid.fid_score as fid_score
import lpips
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# 真实图像和生成图像的路径
real_images_path = r""
generated_images_path = r""

# 规范化路径
real_images_path = os.path.abspath(os.path.normpath(real_images_path))
generated_images_path = os.path.abspath(os.path.normpath(generated_images_path))

print(f"检查路径格式: {real_images_path}")
print(f"检查路径格式: {generated_images_path}")

# 检查路径是否存在
if not os.path.exists(real_images_path):
    raise FileNotFoundError(f"错误: 真实图像路径不存在 -> {real_images_path}")

if not os.path.exists(generated_images_path):
    raise FileNotFoundError(f"错误: 生成图像路径不存在 -> {generated_images_path}")

# 获取所有图片文件
real_files = sorted([f for f in os.listdir(real_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
fake_files = sorted([f for f in os.listdir(generated_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if not real_files:
    raise ValueError(f"错误: 真实图像文件夹中没有有效的图片 -> {real_images_path}")

if not fake_files:
    raise ValueError(f"错误: 生成图像文件夹中没有有效的图片 -> {generated_images_path}")

print(f"真实图像文件夹包含 {len(real_files)} 张图片")
print(f"生成图像文件夹包含 {len(fake_files)} 张图片")

# 确保真实和生成的图片数量一致
if len(real_files) != len(fake_files):
    raise ValueError("错误: 真实和生成的图片数量不匹配")

# 选择计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 计算 FID 分数
print("计算 FID ...")
fid_value = fid_score.calculate_fid_given_paths(
    [real_images_path, generated_images_path],
    batch_size=8,
    device=device,
    dims=2048,
    num_workers=0
)
print(f"FID 分数: {fid_value}")

# 初始化 LPIPS 计算模型
lpips_model = lpips.LPIPS(net='alex').to(device)

# 计算 LPIPS 和 SSIM
lpips_scores = []
ssim_scores = []

for real_file, fake_file in zip(real_files, fake_files):
    real_img_path = os.path.join(real_images_path, real_file)
    fake_img_path = os.path.join(generated_images_path, fake_file)

    # 读取并转换图像格式
    real_img = cv2.imread(real_img_path)
    fake_img = cv2.imread(fake_img_path)

    # 确保图像大小一致
    if real_img.shape != fake_img.shape:
        fake_img = cv2.resize(fake_img, (real_img.shape[1], real_img.shape[0]))

    # 计算 SSIM
    real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    fake_gray = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(real_gray, fake_gray, data_range=255)
    ssim_scores.append(ssim_score)

    # 计算 LPIPS
    real_tensor = torch.tensor(real_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
    fake_tensor = torch.tensor(fake_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1

    real_tensor, fake_tensor = real_tensor.to(device), fake_tensor.to(device)
    lpips_score = lpips_model(real_tensor, fake_tensor).item()
    lpips_scores.append(lpips_score)

# 计算平均 LPIPS 和 SSIM
mean_lpips = np.mean(lpips_scores)
mean_ssim = np.mean(ssim_scores)

print(f"平均 LPIPS 分数: {mean_lpips}")
print(f"平均 SSIM 分数: {mean_ssim}")
