# MMR-Status-Prediction

Deep learning system for predicting **dMMR (deficient Mismatch Repair)** status in colorectal cancer using **PET-CT imaging** and **clinical features**.

---

## 📁 Structure

```
.
├── train.py              # Training (single / fusion model)
├── inference.py          # Inference
├── dataset.py            # MMRDataset (CT / CP / CPC modes)
├── utils.py              # Seed, sampler, augmentation
├── models/               # Model architectures
│   ├── resnet18.py       # 3D ResNet18
│   ├── densenet3d.py     # 3D DenseNet
│   ├── attention.py      # Multi-Head Attention
│   ├── transformer.py    # 3D Vision Transformer
│   ├── fusion.py         # Multimodal fusion models
│   └── bottleneck.py     # Output fusion, domain adaptation
└── data_processing/      # Preprocessing pipeline
    ├── dicom_to_nii.py   # DICOM → NIfTI
    ├── pet_suv.py        # PET SUV calculation
    ├── segmentation.py   # Colon segmentation
    ├── registration.py   # CT-PET rigid registration
    ├── roi_extraction.py # ROI extraction
    ├── resample.py       # Image resampling
    ├── crop.py           # ROI-centered crop
    ├── build_dataset.py  # Build / split dataset
    └── pipeline.py       # Pipeline entry
```

---

## ⚡ Quick Start

### Install

```bash
pip install torch torchvision simpleitk pydicom nibabel numpy pandas scikit-learn openpyxl
pip install TotalSegmentator
```

### Data Format

Excel file with columns: `id`, `CT_Path`, `PET_Path`, `MMR_label` (0=pMMR, 1=dMMR), plus clinical features for CPC mode.

### Train

```python
from models.fusion import CPC_ResNet18_mutli_attention
from train import train_single_model

model = CPC_ResNet18_mutli_attention(in_channels=1, num_class=1)

train_single_model(
    model=model,
    train_csv='./train.xlsx',
    test_csv='./test.xlsx',
    mode='CPC',          # CT / CP / CPC
    epochs=400,
    batch_size=16,
    lr=0.0001,
    save_dir='./result',
    experiment_name='exp',
)
```

### Inference

```python
from models.fusion import CPC_ResNet18_mutli_attention
from inference import load_checkpoint, predict
from dataset import MMRDataset
from torch.utils.data import DataLoader

model = CPC_ResNet18_mutli_attention(in_channels=1, num_class=1)
model = load_checkpoint(model, './result/best.ckpt')

dataset = MMRDataset(csv_path='./test.xlsx', mode='CPC')
loader = DataLoader(dataset, batch_size=16)

results = predict(model, loader, device, mode='CPC')
# [{'id': ..., 'label': ..., 'probability': ..., 'prediction': ...}, ...]
```

---

## 🧠 Models

| Model                             | Mode | Description                         |
| --------------------------------- | ---- | ----------------------------------- |
| `Single_ResNet18_mutli_attention` | CT   | 3D ResNet18 + attention             |
| `CP_ResNet18`                     | CP   | Dual-branch late fusion             |
| `CP_ResNet18_mutli_attention`     | CP   | Dual-branch + cross-modal attention |
| `CPC_ResNet18`                    | CPC  | Dual-branch + clinical FC fusion    |
| `CPC_ResNet18_mutli_attention`    | CPC  | Dual-branch + attention + clinical  |
| `DenseNet`                        | Any  | 3D DenseNet                         |
| `VisionTransformer`               | Any  | 3D ViT                              |

---

## 📊 Metrics

Loss, Accuracy, Specificity, Sensitivity, AUC-ROC are logged to `./result/logs/` and checkpoints saved to `./result/checkpoints/`.

## 📄 License

Academic research use only.

