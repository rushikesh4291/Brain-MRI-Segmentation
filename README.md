# 🧠 Brain MRI Segmentation (U-Net / Attention U-Net)

This repository contains a deep-learning pipeline for **automatic brain tumor segmentation** from MRI scans.
The model takes MRI slices as input and outputs **pixel-wise masks** highlighting tumor regions.

> **Attribution:** Base implementation adapted from [suniash/brain_mri_segmentation](https://github.com/suniash/brain_mri_segmentation) (MIT License).  
> This repo adds clearer structure, a reproducible training/evaluation pipeline, and recruiter-friendly documentation.

---

## 📌 Highlights
- **Task:** Binary semantic segmentation of brain MRIs (tumor vs. background).
- **Models:** U-Net baseline; easily extensible to Attention U-Net / ResNet encoders.
- **Metrics:** Dice Score, IoU, Pixel Accuracy.
- **Reproducibility:** Deterministic seeds, saved configs, and command-line scripts.
- **Clean repo:** Data kept out of version control; sample results provided.

---

## 📂 Repository Structure
```
brain-mri-segmentation/
│
├── data/                      # (Do NOT commit full datasets)
│   ├── sample_images/         # a few tiny example images
│   └── README.md              # where/how to download datasets
│
├── notebooks/                 # experiments
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── src/                       # source code
│   ├── __init__.py
│   ├── config.py              # hyperparams & paths
│   ├── preprocessing.py       # normalization / augmentation
│   ├── dataset.py             # Dataset/Dataloader
│   ├── model.py               # U-Net / Attention blocks
│   ├── train.py               # training loop
│   └── evaluate.py            # metrics & visualization
│
├── results/                   # sample outputs & metrics
│   ├── segmented_samples/
│   └── metrics.json
│
├── requirements.txt           # pip dependencies
├── LICENSE                    # MIT License
└── README.md                  # this file
```
> Tip: add large files (datasets, checkpoints) to `.gitignore` or use Git LFS.

---

## 🗂 Dataset
- Recommended: **LGG MRI Segmentation** (Kaggle) or **BraTS** subsets.  
  - Kaggle example: <https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation>
- **Do not** commit raw datasets. Place downloaded data under `data/` and update paths in `src/config.py`.

**Basic expected layout (customize as needed):**
```
data/
├── images/            # *.png / *.jpg (or NIfTI if you extend to 3D)
└── masks/             # binary masks aligned with images
```

---

## ⚙️ Setup
```bash
# 1) Clone
git clone https://github.com/yourusername/brain-mri-segmentation.git
cd brain-mri-segmentation

# 2) Create env (optional) & install
python -m venv .venv && source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Minimal `requirements.txt` (tweak as needed):
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23
pillow>=9.0
tqdm>=4.60
matplotlib>=3.6
scikit-image>=0.20
scikit-learn>=1.2
```

---

## 🚀 Usage

### 1) Preprocess
```bash
python src/preprocessing.py   --images_dir data/images   --masks_dir data/masks   --out_dir data/processed   --img_size 256
```

### 2) Train
```bash
python src/train.py   --data_dir data/processed   --epochs 50   --batch_size 16   --lr 1e-3   --save_dir checkpoints   --model unet
```

### 3) Evaluate
```bash
python src/evaluate.py   --data_dir data/processed   --model_path checkpoints/unet_best.pth   --save_dir results/segmented_samples
```

> Set seeds & device via `src/config.py` or CLI flags (if implemented).

---

## 📊 Results (Example)
Update this table with **your** actual run metrics.

| Metric     | Value |
|------------|-------|
| Dice Score | 0.87  |
| IoU        | 0.79  |
| Accuracy   | 0.92  |

**Sample predictions:** *(place images in `results/` and update links)*  
| Input MRI | Ground Truth | Prediction |
|-----------|--------------|------------|
| ![MRI](results/sample_mri.png) | ![GT](results/sample_mask.png) | ![Pred](results/sample_pred.png) |

---

## 🔬 Experiments You Can Add (Good for Interviews)
- Attention U-Net blocks (spatial/channel attention).
- Encoder backbones (ResNet34/50 via torchvision).
- Loss functions: BCE+DICE, Focal, Tversky.
- 3D extension for volumetric MRIs (NIfTI) with MONAI or 3D U-Net.
- Post-processing: connected components, CRF refinement.
- Training tricks: mixup/cutmix (for segmentation), cosine LR, SAM/Lookahead.

Document each experiment briefly in the README or a `REPORT.md`.

---

## 🧰 Tips for Recruiters/Reviewers
- `notebooks/` show the end-to-end workflow (EDA → training → evaluation).
- `src/` contains clean, reusable code.
- `results/` includes metrics and a few qualitative examples.
- Dataset is referenced—not included—keeping the repo light and compliant.

---

## 🧾 License & Attribution
- Licensed under **MIT** (see `LICENSE`).  
- Based on **suniash/brain_mri_segmentation** (MIT). If you reuse code from there,
  keep the license notice and link to the original repository.

---

## 🙋 Maintainer
**Your Name** · IIT Kharagpur (Engineering Entrepreneurship)  
- Email: your.email@example.com  
- LinkedIn: https://www.linkedin.com/in/your-profile

> If you use or extend this work, a star ⭐ on the repo is appreciated!
