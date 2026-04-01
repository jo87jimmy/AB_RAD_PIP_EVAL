"""
消融實驗推論腳本 (Ablation Study Inference Script)
===================================================

本腳本實現了三種消融模式的推論，用於評估各模組對異常檢測效能的貢獻：

模式說明：
  Mode A - 僅重建損失 (Recon-Only):
    只使用重建子網路的重建誤差 (L2 + SSIM) 作為異常分數。
    完全跳過判別子網路，直接以像素級重建誤差衡量異常。
    用途：驗證重建子網路本身的檢測能力。

  Mode B - 重建 + 判別一致性 (Recon + Discriminative):
    使用完整的兩階段管線：重建 → 判別分割。
    以判別子網路的 softmax 輸出作為異常分數。
    用途：驗證加入判別子網路後的效能提升。

  Mode C - 完整版 + 平均池化平滑 (Full Pipeline):
    在 Mode B 基礎上，對異常分數圖進行 21×21 平均池化平滑。
    這是訓練時搭配動態權重 Warmup 後的完整推論管線。
    用途：驗證後處理平滑對最終 AUROC 的貢獻。

執行方式：
  python main.py --obj_id 1          # 測試 bottle 類別
  python main.py --obj_id -1         # 測試全部 15 個類別
  python main.py --obj_id 1 --gpu_id 0  # 指定 GPU
"""

import os
import argparse
import random

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data_loader import MVTecDRAEMTestDataset
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import SSIM


# =======================
# 隨機種子設定
# =======================
def setup_seed(seed):
    """設定隨機種子，確保實驗可重現"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =======================
# GPU 選擇工具
# =======================
def get_available_gpu():
    """自動選擇記憶體使用率最低的 GPU"""
    if not torch.cuda.is_available():
        return -1
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return -1
    gpu_memory = []
    for i in range(gpu_count):
        torch.cuda.set_device(i)
        memory_allocated = torch.cuda.memory_allocated(i)
        gpu_memory.append((i, memory_allocated))
    available_gpu = min(gpu_memory, key=lambda x: x[1])[0]
    return available_gpu


# =======================
# 模型載入
# =======================
def load_student_models(obj_name, checkpoint_dir, device):
    """
    載入已訓練好的學生模型權重。

    Args:
        obj_name (str): 物件類別名稱 (如 'bottle')
        checkpoint_dir (str): 權重檔案所在目錄
        device (str): 運算裝置 ('cuda' 或 'cpu')

    Returns:
        tuple: (recon_model, seg_model) 重建子網路與判別子網路
    """
    # ---- 重建子網路 ----
    recon_model = ReconstructiveSubNetwork(
        in_channels=3, out_channels=3, base_width=64
    )
    recon_weights_path = os.path.join(checkpoint_dir, f"{obj_name}_best_recon.pckl")
    if not os.path.exists(recon_weights_path):
        raise FileNotFoundError(
            f"❌ 未找到重建模型權重: {recon_weights_path}"
        )
    recon_model.load_state_dict(
        torch.load(recon_weights_path, map_location=device)
    )
    recon_model.to(device)
    recon_model.eval()

    # ---- 判別子網路 ----
    seg_model = DiscriminativeSubNetwork(
        in_channels=6, out_channels=2, base_channels=32
    )
    seg_weights_path = os.path.join(checkpoint_dir, f"{obj_name}_best_seg.pckl")
    if not os.path.exists(seg_weights_path):
        raise FileNotFoundError(
            f"❌ 未找到分割模型權重: {seg_weights_path}"
        )
    seg_model.load_state_dict(
        torch.load(seg_weights_path, map_location=device)
    )
    seg_model.to(device)
    seg_model.eval()

    return recon_model, seg_model


# =======================
# 消融推論策略
# =======================
class AblationMode:
    """
    消融實驗模式定義。

    每個模式對應一種異常分數計算方式，
    用於衡量不同模組對最終效能 (AUROC) 的貢獻。
    """
    RECON_ONLY = "recon_only"          # Mode A: 僅使用重建誤差
    RECON_PLUS_DISC = "recon_plus_disc"  # Mode B: 重建 + 判別分割
    FULL_PIPELINE = "full_pipeline"      # Mode C: 完整版 (含平均池化平滑)


def compute_anomaly_score_recon_only(recon_model, image_batch, device):
    """
    Mode A: 僅重建損失 (Recon-Only)

    只使用重建子網路，以像素級重建誤差作為異常分數。
    計算方式為 L2 誤差圖取通道平均後的最大值。

    原因：
      此模式「拔掉」了判別子網路，
      純粹依賴重建網路的還原能力來偵測異常。
      若異常區域無法被良好重建，則重建誤差會較高。

    Args:
        recon_model: 重建子網路
        image_batch: 輸入圖像批次 [B, 3, H, W]
        device: 運算裝置

    Returns:
        tuple: (image_scores, pixel_scores_flat)
          - image_scores: 每張圖的影像級異常分數 list
          - pixel_scores_flat: 每張圖的像素級異常分數 (展平) list
    """
    with torch.no_grad():
        # 通過重建子網路產生重建圖像
        recon_output = recon_model(image_batch)

        # 計算像素級 L2 重建誤差，沿通道維度 (dim=1) 取平均
        # 結果形狀: [B, 1, H, W]
        l2_error_map = ((recon_output - image_batch) ** 2).mean(dim=1, keepdim=True)

        image_scores = []
        pixel_scores_flat = []

        for i in range(l2_error_map.shape[0]):
            error_map = l2_error_map[i, 0].cpu().numpy()  # [H, W]
            # 影像級分數：取整張圖的最大重建誤差
            image_scores.append(float(np.max(error_map)))
            # 像素級分數：展平供 AUROC 計算
            pixel_scores_flat.append(error_map.flatten())

    return image_scores, pixel_scores_flat


def compute_anomaly_score_recon_plus_disc(recon_model, seg_model, image_batch, device):
    """
    Mode B: 重建 + 判別一致性 (Recon + Discriminative)

    使用完整的兩階段管線：
      1. 重建子網路產生重建圖像
      2. 將重建圖像與原圖拼接後送入判別子網路
      3. 以 softmax 後的異常類別機率 (channel 1) 作為異常分數

    原因：
      此模式加入了判別子網路，可觀察到判別網路對異常定位能力的提升。
      判別子網路在訓練時學到了「重建差異 → 異常遮罩」的映射關係。

    Args:
        recon_model: 重建子網路
        seg_model: 判別子網路
        image_batch: 輸入圖像批次 [B, 3, H, W]
        device: 運算裝置

    Returns:
        tuple: (image_scores, pixel_scores_flat)
    """
    with torch.no_grad():
        recon_output = recon_model(image_batch)
        # 將重建圖像與原始圖像在通道維度拼接，形成 6 通道輸入
        joined_input = torch.cat((recon_output, image_batch), dim=1)
        seg_output = seg_model(joined_input)
        # softmax 後取異常類別 (index=1) 的機率
        seg_probs = torch.softmax(seg_output, dim=1)

        image_scores = []
        pixel_scores_flat = []

        for i in range(seg_probs.shape[0]):
            anomaly_map = seg_probs[i, 1].cpu().numpy()  # [H, W]
            # 影像級分數：取異常機率的最大值
            image_scores.append(float(np.max(anomaly_map)))
            pixel_scores_flat.append(anomaly_map.flatten())

    return image_scores, pixel_scores_flat


def compute_anomaly_score_full_pipeline(recon_model, seg_model, image_batch, device):
    """
    Mode C: 完整版 + 平均池化平滑 (Full Pipeline)

    在 Mode B 的基礎上，增加 21×21 的平均池化平滑處理。
    這是訓練時搭配動態權重 Warmup 策略後的完整推論管線。

    原因：
      平均池化平滑可以消除分割輸出中的高頻雜訊，
      使異常得分圖更加平滑，降低假陽性。
      此模式代表完整系統的最終效能。

    Args:
        recon_model: 重建子網路
        seg_model: 判別子網路
        image_batch: 輸入圖像批次 [B, 3, H, W]
        device: 運算裝置

    Returns:
        tuple: (image_scores, pixel_scores_flat)
    """
    with torch.no_grad():
        recon_output = recon_model(image_batch)
        joined_input = torch.cat((recon_output, image_batch), dim=1)
        seg_output = seg_model(joined_input)
        seg_probs = torch.softmax(seg_output, dim=1)

        # 對異常類別機率圖進行 21×21 平均池化平滑
        # kernel_size=21, stride=1, padding=10 確保輸出尺寸不變
        anomaly_channel = seg_probs[:, 1:, :, :]  # [B, 1, H, W]
        smoothed = F.avg_pool2d(
            anomaly_channel, kernel_size=21, stride=1, padding=21 // 2
        )

        image_scores = []
        pixel_scores_flat = []

        for i in range(smoothed.shape[0]):
            smooth_map = smoothed[i, 0].cpu().numpy()  # [H, W]
            # 影像級分數：平滑後的最大值
            image_scores.append(float(np.max(smooth_map)))
            # 像素級使用原始 (未平滑) 的分割輸出，保留定位精度
            raw_map = seg_probs[i, 1].cpu().numpy()
            pixel_scores_flat.append(raw_map.flatten())

    return image_scores, pixel_scores_flat


# =======================
# 單類別消融測試
# =======================
def run_ablation_for_category(obj_name, args, device):
    """
    對單一 MVTec 類別執行三種消融模式的推論與 AUROC 計算。

    Args:
        obj_name (str): 物件類別名稱
        args: 命令列參數
        device (str): 運算裝置

    Returns:
        dict: 三種模式的 Image-level / Pixel-level AUROC 結果
    """
    print(f"\n{'='*60}")
    print(f"  📦 類別: {obj_name}")
    print(f"{'='*60}")

    # 1. 載入模型
    try:
        recon_model, seg_model = load_student_models(
            obj_name, args.checkpoint_dir, device
        )
    except FileNotFoundError as e:
        print(f"  {e}")
        return None

    # 2. 準備測試資料集
    test_path = os.path.join(args.mvtec_root, obj_name, "test")
    if not os.path.exists(test_path):
        print(f"  ❌ 測試資料路徑不存在: {test_path}")
        return None

    dataset = MVTecDRAEMTestDataset(
        root_dir=test_path, resize_shape=[256, 256]
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )
    print(f"  📊 測試集大小: {len(dataset)} 張圖片")

    # 3. 收集 Ground Truth 標籤
    all_gt_labels = []     # 影像級: 0=正常, 1=異常
    all_gt_masks = []      # 像素級: 展平的 mask

    # 4. 各模式的預測結果容器
    results_per_mode = {
        AblationMode.RECON_ONLY: {"img_scores": [], "pix_scores": []},
        AblationMode.RECON_PLUS_DISC: {"img_scores": [], "pix_scores": []},
        AblationMode.FULL_PIPELINE: {"img_scores": [], "pix_scores": []},
    }

    # 5. 逐批次推論
    for sample_batched in dataloader:
        image_batch = sample_batched["image"].to(device)
        gt_label = sample_batched["has_anomaly"].numpy()[0, 0]  # 0 or 1
        gt_mask = sample_batched["mask"].numpy()[0]  # [1, H, W]

        all_gt_labels.append(gt_label)
        all_gt_masks.append(gt_mask.flatten())

        # --- Mode A: Recon-Only ---
        img_s, pix_s = compute_anomaly_score_recon_only(
            recon_model, image_batch, device
        )
        results_per_mode[AblationMode.RECON_ONLY]["img_scores"].extend(img_s)
        results_per_mode[AblationMode.RECON_ONLY]["pix_scores"].extend(pix_s)

        # --- Mode B: Recon + Disc ---
        img_s, pix_s = compute_anomaly_score_recon_plus_disc(
            recon_model, seg_model, image_batch, device
        )
        results_per_mode[AblationMode.RECON_PLUS_DISC]["img_scores"].extend(img_s)
        results_per_mode[AblationMode.RECON_PLUS_DISC]["pix_scores"].extend(pix_s)

        # --- Mode C: Full Pipeline ---
        img_s, pix_s = compute_anomaly_score_full_pipeline(
            recon_model, seg_model, image_batch, device
        )
        results_per_mode[AblationMode.FULL_PIPELINE]["img_scores"].extend(img_s)
        results_per_mode[AblationMode.FULL_PIPELINE]["pix_scores"].extend(pix_s)

    # 6. 計算 AUROC
    all_gt_labels = np.array(all_gt_labels)
    all_gt_masks_flat = np.concatenate(all_gt_masks)

    category_results = {}
    mode_display_names = {
        AblationMode.RECON_ONLY: "Mode A: 僅重建損失 (L_recon)",
        AblationMode.RECON_PLUS_DISC: "Mode B: + 判別一致性 (L_s_dist)",
        AblationMode.FULL_PIPELINE: "Mode C: 完整版 (動態權重 Warmup)",
    }

    for mode, data in results_per_mode.items():
        img_scores = np.array(data["img_scores"])
        pix_scores = np.concatenate(data["pix_scores"])

        # Image-level AUROC
        try:
            img_auroc = roc_auc_score(all_gt_labels, img_scores)
        except ValueError:
            img_auroc = float("nan")

        # Pixel-level AUROC
        try:
            pix_auroc = roc_auc_score(
                all_gt_masks_flat.astype(int), pix_scores
            )
        except ValueError:
            pix_auroc = float("nan")

        category_results[mode] = {
            "image_auroc": img_auroc,
            "pixel_auroc": pix_auroc,
        }
        print(
            f"  {mode_display_names[mode]:<45} | "
            f"Image AUROC: {img_auroc:.4f} | "
            f"Pixel AUROC: {pix_auroc:.4f}"
        )

    return category_results


# =======================
# 消融實驗表格輸出
# =======================
def print_ablation_table(all_results):
    """
    以表格形式輸出所有類別的消融實驗結果，
    並計算平均 AUROC。

    Args:
        all_results (dict): {obj_name: {mode: {image_auroc, pixel_auroc}}} 結構的結果字典
    """
    modes = [
        AblationMode.RECON_ONLY,
        AblationMode.RECON_PLUS_DISC,
        AblationMode.FULL_PIPELINE,
    ]
    mode_short_names = {
        AblationMode.RECON_ONLY: "L_recon Only",
        AblationMode.RECON_PLUS_DISC: "+ L_s_dist",
        AblationMode.FULL_PIPELINE: "Full (Warmup)",
    }

    # ---- 影像級 AUROC 表格 ----
    print("\n")
    print("=" * 80)
    print("  📊 消融實驗結果 - Image-level AUROC")
    print("=" * 80)

    # 表頭
    header = f"{'Category':<15}"
    for mode in modes:
        header += f" | {mode_short_names[mode]:>15}"
    print(header)
    print("-" * 80)

    # 各類別數據
    avg_scores = {mode: [] for mode in modes}
    for obj_name, results in all_results.items():
        if results is None:
            continue
        row = f"{obj_name:<15}"
        for mode in modes:
            score = results[mode]["image_auroc"]
            row += f" | {score:>15.4f}"
            if not np.isnan(score):
                avg_scores[mode].append(score)
        print(row)

    # 平均值
    print("-" * 80)
    avg_row = f"{'Average':<15}"
    for mode in modes:
        scores = avg_scores[mode]
        avg = np.mean(scores) if scores else float("nan")
        avg_row += f" | {avg:>15.4f}"
    print(avg_row)
    print("=" * 80)

    # ---- 像素級 AUROC 表格 ----
    print("\n")
    print("=" * 80)
    print("  📊 消融實驗結果 - Pixel-level AUROC")
    print("=" * 80)

    print(header)  # 重用同樣的表頭
    print("-" * 80)

    avg_scores_px = {mode: [] for mode in modes}
    for obj_name, results in all_results.items():
        if results is None:
            continue
        row = f"{obj_name:<15}"
        for mode in modes:
            score = results[mode]["pixel_auroc"]
            row += f" | {score:>15.4f}"
            if not np.isnan(score):
                avg_scores_px[mode].append(score)
        print(row)

    print("-" * 80)
    avg_row = f"{'Average':<15}"
    for mode in modes:
        scores = avg_scores_px[mode]
        avg = np.mean(scores) if scores else float("nan")
        avg_row += f" | {avg:>15.4f}"
    print(avg_row)
    print("=" * 80)


# =======================
# 結果存檔
# =======================
def save_results_to_csv(all_results, output_path):
    """
    將消融實驗結果儲存為 CSV 檔案，方便後續分析或繪圖。

    Args:
        all_results (dict): 結果字典
        output_path (str): CSV 輸出路徑
    """
    modes = [
        AblationMode.RECON_ONLY,
        AblationMode.RECON_PLUS_DISC,
        AblationMode.FULL_PIPELINE,
    ]
    mode_labels = {
        AblationMode.RECON_ONLY: "L_recon_Only",
        AblationMode.RECON_PLUS_DISC: "Plus_L_s_dist",
        AblationMode.FULL_PIPELINE: "Full_Warmup",
    }

    lines = []
    # CSV 表頭
    header_parts = ["Category"]
    for mode in modes:
        header_parts.append(f"{mode_labels[mode]}_Image_AUROC")
        header_parts.append(f"{mode_labels[mode]}_Pixel_AUROC")
    lines.append(",".join(header_parts))

    # 資料列
    for obj_name, results in all_results.items():
        if results is None:
            continue
        parts = [obj_name]
        for mode in modes:
            parts.append(f"{results[mode]['image_auroc']:.6f}")
            parts.append(f"{results[mode]['pixel_auroc']:.6f}")
        lines.append(",".join(parts))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n💾 結果已儲存至: {output_path}")


# =======================
# 主流程
# =======================
def main(obj_names, args):
    """
    消融實驗主流程。

    依序對每個目標類別執行三種消融模式的推論，
    計算 Image-level 與 Pixel-level AUROC，
    最後以表格和 CSV 形式輸出結果。

    Args:
        obj_names (list[str]): 待測試的物件類別名稱列表
        args: 命令列參數
    """
    setup_seed(111)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  運算裝置: {device}")
    print(f"🔄 開始消融實驗，共 {len(obj_names)} 個類別")

    all_results = {}

    for obj_name in obj_names:
        results = run_ablation_for_category(obj_name, args, device)
        all_results[obj_name] = results

    # 輸出匯總表格
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if valid_results:
        print_ablation_table(valid_results)

        # 儲存 CSV
        output_dir = "./ablation_results"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "ablation_auroc_results.csv")
        save_results_to_csv(valid_results, csv_path)
    else:
        print("\n❌ 沒有任何類別成功完成測試，請檢查權重檔案路徑。")

    print("\n🎉 消融實驗完成！")


# =======================
# 程式進入點
# =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="消融實驗推論腳本 - 評估各模組對異常檢測效能的貢獻"
    )
    parser.add_argument(
        "--obj_id", type=int, required=True,
        help="物件類別 ID (-1 表示全部類別)"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=-2,
        help="GPU ID (-2: 自動選擇, -1: CPU)"
    )
    parser.add_argument(
        "--mvtec_root", type=str, default="./mvtec",
        help="MVTec 資料集根目錄路徑"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default="./save_files/checkpoints",
        help="學生模型權重檔案所在目錄"
    )

    args = parser.parse_args()

    # 自動選擇 GPU
    if args.gpu_id == -2:
        args.gpu_id = get_available_gpu()
        print(f"自動選擇 GPU: {args.gpu_id}")

    # 定義所有 MVTec AD 類別
    obj_batch = [
        ["capsule"], ["bottle"], ["carpet"], ["leather"], ["pill"],
        ["transistor"], ["tile"], ["cable"], ["zipper"],
        ["toothbrush"], ["metal_nut"], ["hazelnut"], ["screw"],
        ["grid"], ["wood"],
    ]

    if int(args.obj_id) == -1:
        picked_classes = [
            "capsule", "bottle", "carpet", "leather", "pill",
            "transistor", "tile", "cable", "zipper",
            "toothbrush", "metal_nut", "hazelnut", "screw",
            "grid", "wood",
        ]
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    # 執行消融實驗
    if args.gpu_id == -1:
        main(picked_classes, args)
    else:
        with torch.cuda.device(args.gpu_id):
            main(picked_classes, args)
