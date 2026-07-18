# Standard vs. Custom-Designed Spatial Filters for Brain Tumor MRI Enhancement

Comparison of standard spatial filters (Box, Median, Sobel, Laplacian, High-Boost) against two custom, content-adaptive filters for denoising and sharpening brain tumor MRI images. Evaluated with PSNR, SSIM, Edge Strength, a custom Edge Preservation Index, and downstream CNN classification accuracy.

**Course:** Image Processing Sessional (CSE 4106), Dept. of CSE, RUET <br>
**Authors:** Mst. Maysha Khanom Moon (2103068), Dina Shanjida (2103069)

---

## Problem

Standard filters apply one fixed operation to every pixel — smoothing removes noise but blurs tumor boundaries; sharpening restores edges but also amplifies noise. This project builds two filters that adapt their behavior per pixel based on local image statistics.

## Custom Filters

**1. Adaptive Tumor-Preserving Mean** — smooths low-variance (flat/noisy) regions, leaves high-variance (edge/boundary) pixels untouched.
```python
def adaptive_tumor_preserving_mean(img, ksize=5, var_threshold=50):
    img_f = img.astype(np.float32)
    mean = cv2.blur(img_f, (ksize, ksize))
    mean_sq = cv2.blur(img_f**2, (ksize, ksize))
    local_var = mean_sq - mean**2
    output = np.where(local_var < var_threshold, mean, img_f)
    return np.clip(output, 0, 255).astype(np.uint8)
```

**2. Adaptive High-Boost** — scales sharpening gain by local gradient magnitude (light on flat regions, strong on edges).
```python
def adaptive_high_boost(img, ksize=(5,5), k_min=0.5, k_max=2.5):
    img_f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, ksize, 0)
    mask = img_f - blurred
    gx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)
    adaptive_k = k_min + (k_max - k_min) * grad_norm
    boosted = img_f + adaptive_k * mask
    return np.clip(boosted, 0, 255).astype(np.uint8)
```

**Edge Preservation Index (EPI)** — IoU of Canny edge maps between clean and filtered images:
```python
def edge_preservation_index(original, filtered):
    orig_edges = cv2.Canny(original, 50, 150)
    filt_edges = cv2.Canny(filtered, 50, 150)
    intersection = np.logical_and(orig_edges, filt_edges).sum()
    union = np.logical_or(orig_edges, filt_edges).sum()
    return intersection / union if union > 0 else 0
```

---

## Dataset

**Brain Tumor MRI Dataset** — 7,200 grayscale images, 4 balanced classes.

```
Training/  glioma, meningioma, pituitary, notumor  (1400 each)
Testing/   glioma, meningioma, pituitary, notumor   (400 each)
```

Curated from [figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427), [SARTAJ](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri), and [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) (notumor class). Deduplicated, balanced, no train/test leakage.

> This project uses a **50 images/class subset** (200 total) for the filtering pipeline and CNN experiment.

---

## Notebook Workflow

1. Preprocessing
2. Setup — metrics + filter implementations
3. Degradation (Gaussian noise σ=15, salt-and-pepper, blur)
4. Degradation before/after visualization
5. Apply standard & custom filters
6. Quantitative evaluation (PSNR, SSIM, Edge Strength, EPI, runtime)
7. Summary table
8. Bar charts — standard vs. custom
9. Radar chart — normalized multi-metric comparison
10. Full custom-filtered dataset generation
11. CNN build & accuracy comparison (clean vs. custom-filtered)

---

## Results

| Degradation | Filter | PSNR | SSIM | Edge Str. | Edge Pres. |
|---|---|---|---|---|---|
| Gaussian | Box (Std) | 27.77 | 0.690 | 56.75 | 0.369 |
| Gaussian | Median (Std) | 27.98 | 0.713 | 62.05 | 0.345 |
| Gaussian | **Custom Smoothing** | 25.67 | 0.526 | 95.30 | 0.322 |
| Salt & Pepper | Box (Std) | 26.50 | 0.717 | 60.03 | 0.307 |
| Salt & Pepper | Median (Std) | 30.74 | 0.918 | 51.37 | 0.471 |
| Salt & Pepper | **Custom Smoothing** | 21.25 | 0.687 | 90.46 | **0.495** |
| Blurred | Sobel (Std) | 18.54 | 0.643 | 46.10 | 0.102 |
| Blurred | Laplacian (Std) | 22.32 | 0.677 | 44.20 | 0.136 |
| Blurred | High Boost (Std) | 24.96 | 0.768 | 34.10 | 0.100 |
| Blurred | **Custom Sharpening** | **25.08** | **0.773** | 34.18 | 0.097 |

**CNN classification accuracy:** clean baseline 70.0% vs. custom-filtered 68.3%.

**Findings:**
- Custom Sharpening wins outright — best PSNR/SSIM, less halo artifact than fixed-gain High-Boost.
- Custom Smoothing wins Edge Preservation on salt-and-pepper noise (0.495 vs. 0.471).
- Custom Smoothing underperforms on Gaussian noise (SSIM 0.526 vs. 0.69–0.71) — `var_threshold=50` is too low, so noise-inflated variance gets misread as edges.
- Custom filters run ~10–20× slower than standard filters (per-pixel computation).

---

## Requirements

```
opencv-python numpy pandas matplotlib scikit-image scikit-learn tensorflow
```

## How to Run

1. Upload the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) `Training/` folder to Google Drive.
2. Open `DIP_project_version_2.ipynb` in Colab, set `SOURCE_DIR` to your dataset path.
3. Run all cells. (For local use, replace the `google.colab.drive.mount()` cell with a local path.)

## `.gitignore`

```
dataset/
results/
.ipynb_checkpoints/
__pycache__/
```

## Future Work

- Soft variance-blending instead of a hard threshold
- Learned per-image parameters
- Combined smoothing + sharpening pipeline
- Full-dataset validation with a deeper CNN

## Acknowledgements

Dataset sources: [figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427), [SARTAJ](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri), [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection).

## License

Coursework project (CSE 4106, RUET). Shared for portfolio/learning purposes — attribution appreciated.
