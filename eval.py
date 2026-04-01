"""
消融實驗單張圖片推論腳本 (Ablation Single-Image Inference)
==========================================================

本腳本提供三種消融模式的單張圖片推論功能，
可對指定圖片同時展示三種模式的異常檢測結果。

模式說明：
  Mode A - 僅重建損失 (Recon-Only): 以重建誤差圖作為異常分數
  Mode B - + 判別一致性 (Recon + Disc): 以判別子網路輸出作為異常分數
  Mode C - 完整版 (Full Pipeline): 加入平均池化平滑後處理

執行方式：
  python eval.py --image_path ./mvtec/bottle/test/broken_large/000.png --obj_name bottle
"""

import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork


# =======================
# 模型載入
# =======================
def load_student_models(obj_name, checkpoint_dir, device):
    """
    載入已訓練好的學生模型權重。

    Args:
        obj_name (str): 物件類別名稱
        checkpoint_dir (str): 權重檔案所在目錄
        device (str): 運算裝置

    Returns:
        tuple: (recon_model, seg_model)
    """
    recon_model = ReconstructiveSubNetwork(
        in_channels=3, out_channels=3, base_width=64
    )
    recon_path = os.path.join(checkpoint_dir, f"{obj_name}_best_recon.pckl")
    if not os.path.exists(recon_path):
        raise FileNotFoundError(f"❌ 未找到重建模型權重: {recon_path}")
    recon_model.load_state_dict(torch.load(recon_path, map_location=device))
    recon_model.to(device)
    recon_model.eval()

    seg_model = DiscriminativeSubNetwork(
        in_channels=6, out_channels=2, base_channels=32
    )
    seg_path = os.path.join(checkpoint_dir, f"{obj_name}_best_seg.pckl")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"❌ 未找到分割模型權重: {seg_path}")
    seg_model.load_state_dict(torch.load(seg_path, map_location=device))
    seg_model.to(device)
    seg_model.eval()

    return recon_model, seg_model


# =======================
# 圖片前處理
# =======================
def preprocess_image(image_path, img_dim=256):
    """
    載入並前處理單張圖片，與訓練時的測試集處理方式一致。

    Args:
        image_path (str): 圖片路徑
        img_dim (int): 目標尺寸 (寬和高)

    Returns:
        tuple: (image_tensor [1, 3, H, W], original_image_np [H, W, 3])
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"❌ 無法載入圖片: {image_path}")
    image = cv2.resize(image, dsize=(img_dim, img_dim))

    # 與 data_loader 一致：除以 255 歸一化到 [0, 1]
    original_display = image[:, :, ::-1].copy() / 255.0  # BGR → RGB，歸一化
    image_float = image.astype(np.float32) / 255.0  # [H, W, 3] BGR
    image_tensor = torch.from_numpy(
        np.transpose(image_float, (2, 0, 1))  # [3, H, W]
    ).unsqueeze(0)  # [1, 3, H, W]

    return image_tensor, original_display


# =======================
# 消融推論與視覺化
# =======================
def ablation_inference_single(image_path, obj_name, checkpoint_dir, device, save_path=None):
    """
    對單張圖片執行三種消融模式的推論，並視覺化比較結果。

    Args:
        image_path (str): 輸入圖片路徑
        obj_name (str): 物件類別名稱
        checkpoint_dir (str): 權重檔案目錄
        device (str): 運算裝置
        save_path (str, optional): 結果儲存路徑。若為 None 則直接顯示。
    """
    # 1. 載入模型
    recon_model, seg_model = load_student_models(obj_name, checkpoint_dir, device)

    # 2. 前處理圖片
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # 3. 三種模式推論
    with torch.no_grad():
        # 重建輸出 (所有模式共用)
        recon_output = recon_model(image_tensor)

        # --- Mode A: 僅重建誤差 ---
        l2_error = ((recon_output - image_tensor) ** 2).mean(dim=1, keepdim=True)
        mode_a_map = l2_error[0, 0].cpu().numpy()

        # --- Mode B: 重建 + 判別 ---
        joined_input = torch.cat((recon_output, image_tensor), dim=1)
        seg_output = seg_model(joined_input)
        seg_probs = torch.softmax(seg_output, dim=1)
        mode_b_map = seg_probs[0, 1].cpu().numpy()

        # --- Mode C: 完整版 + 平均池化 ---
        smoothed = F.avg_pool2d(
            seg_probs[:, 1:, :, :], kernel_size=21, stride=1, padding=21 // 2
        )
        mode_c_map = smoothed[0, 0].cpu().numpy()

        # 重建圖像 (用於顯示)
        recon_display = recon_output[0].permute(1, 2, 0).cpu().numpy()
        recon_display = np.clip(recon_display, 0, 1)

    # 4. 視覺化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"消融實驗 - {obj_name} - {os.path.basename(image_path)}",
        fontsize=16, fontweight="bold"
    )

    # 第一行：原圖、重建圖、重建誤差差異
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("原始圖像", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(recon_display[:, :, ::-1])  # BGR → RGB
    axes[0, 1].set_title("重建圖像", fontsize=12)
    axes[0, 1].axis("off")

    # 原圖與重建的差異
    diff_display = np.abs(original_image - recon_display[:, :, ::-1])
    axes[0, 2].imshow(diff_display)
    axes[0, 2].set_title("重建差異 |Input - Recon|", fontsize=12)
    axes[0, 2].axis("off")

    # 第二行：三種異常分數圖
    im_a = axes[1, 0].imshow(mode_a_map, cmap="hot", vmin=0)
    axes[1, 0].set_title(
        f"Mode A: 僅 L_recon\nMax: {mode_a_map.max():.4f}",
        fontsize=11
    )
    axes[1, 0].axis("off")
    plt.colorbar(im_a, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_b = axes[1, 1].imshow(mode_b_map, cmap="hot", vmin=0, vmax=1)
    axes[1, 1].set_title(
        f"Mode B: + L_s_dist\nMax: {mode_b_map.max():.4f}",
        fontsize=11
    )
    axes[1, 1].axis("off")
    plt.colorbar(im_b, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im_c = axes[1, 2].imshow(mode_c_map, cmap="hot", vmin=0, vmax=1)
    axes[1, 2].set_title(
        f"Mode C: Full (Warmup)\nMax: {mode_c_map.max():.4f}",
        fontsize=11
    )
    axes[1, 2].axis("off")
    plt.colorbar(im_c, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
        print(f"💾 視覺化結果已儲存: {save_path}")
    else:
        plt.show()
    plt.close()


# =======================
# 程式進入點
# =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="消融實驗單張圖片推論 - 三種模式異常檢測結果比較"
    )
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="要測試的圖片路徑"
    )
    parser.add_argument(
        "--obj_name", type=str, required=True,
        help="物件類別名稱 (如 bottle, capsule 等)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default="./save_files/checkpoints",
        help="學生模型權重檔案所在目錄"
    )
    parser.add_argument(
        "--save_path", type=str, default=None,
        help="結果儲存路徑 (若不指定則直接顯示)"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=-2,
        help="GPU ID (-2: 自動選擇, -1: CPU)"
    )

    args = parser.parse_args()

    # 裝置選擇
    if args.gpu_id == -2:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = []
            for i in range(gpu_count):
                torch.cuda.set_device(i)
                gpu_memory.append((i, torch.cuda.memory_allocated(i)))
            args.gpu_id = min(gpu_memory, key=lambda x: x[1])[0] if gpu_memory else -1
        else:
            args.gpu_id = -1

    device = f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu"
    print(f"🖥️  運算裝置: {device}")

    ablation_inference_single(
        image_path=args.image_path,
        obj_name=args.obj_name,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        save_path=args.save_path,
    )