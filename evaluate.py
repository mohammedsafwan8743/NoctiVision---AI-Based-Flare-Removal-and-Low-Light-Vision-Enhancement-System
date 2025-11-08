import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

# Paths
gt_folder = r'C:\Users\sweet\Downloads\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\dataset\Flare7Kpp\test_data\real\gt'          # Folder containing gt_000000.png
pred_folder = r'C:\Users\sweet\Downloads\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\training outputdeflare'      # Folder containing 000000_deflare.png

# Initialize accumulators
psnr_scores = []
ssim_scores = []
pixel_auroc_scores = []
sample_auroc_scores = []
pixel_aupr_scores = []

# Loop through predicted images
for pred_name in os.listdir(pred_folder):
    if not pred_name.endswith('_deflare.png'):
        continue

    # Extract index from prediction filename
    index = pred_name.replace('_deflare.png', '')
    gt_name = f'gt_{index.zfill(6)}.png'  # Ensure 6-digit padding

    pred_path = os.path.join(pred_folder, pred_name)
    gt_path = os.path.join(gt_folder, gt_name)

    if not os.path.exists(gt_path):
        print(f"GT not found for {pred_name}")
        continue

    # Load images
    pred_img = cv2.imread(pred_path)
    gt_img = cv2.imread(gt_path)

    # Resize if shapes don't match
    if pred_img.shape != gt_img.shape:
        gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]))

    # Convert to RGB
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    # Compute PSNR and SSIM
    psnr_val = psnr(gt_img, pred_img, data_range=255)
    ssim_val = ssim(gt_img, pred_img, data_range=255, channel_axis=-1)

    # Flatten images for AUROC and AUPR calculations
    gt_flat = gt_img.flatten()
    pred_flat = pred_img.flatten()

    # Calculate Pixel AUROC: Binary thresholding for each pixel (assuming flares are brighter)
    gt_binary = (gt_flat > np.mean(gt_flat)).astype(int)
    pred_binary = (pred_flat > np.mean(pred_flat)).astype(int)

    pixel_auroc = roc_auc_score(gt_binary, pred_binary)
    pixel_auroc_scores.append(pixel_auroc)

    # Sample AUROC: Calculate AUROC for each image as a whole (multi-class)
    # Convert the images to probabilities for AUROC calculation
    gt_prob = np.array([gt_flat])
    pred_prob = np.array([pred_flat])

    try:
        sample_auroc = roc_auc_score(gt_prob, pred_prob, multi_class='ovr', average='macro')
        sample_auroc_scores.append(sample_auroc)
    except ValueError:
        # In case of errors, we append NaN for this case
        sample_auroc_scores.append(np.nan)

    # Pixel AUPR: Area under the Precision-Recall curve for each pixel
    pixel_aupr = average_precision_score(gt_binary, pred_binary)
    pixel_aupr_scores.append(pixel_aupr)

    # Accumulate metrics for printing
    psnr_scores.append(psnr_val)
    ssim_scores.append(ssim_val)

    print(f"{pred_name} - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, Pixel AUROC: {pixel_auroc:.4f}, Pixel AUPRO: {pixel_aupr:.4f}")

# Print overall scores
print("\n--- Overall Evaluation ---")
print(f"Average PSNR: {np.mean(psnr_scores):.2f}")
print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
print(f"Average Pixel AUROC: {np.mean(pixel_auroc_scores):.4f}")
print(f"Average Pixel AUPRO: {np.mean(pixel_aupr_scores):.4f}")
