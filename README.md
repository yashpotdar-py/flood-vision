# 🌊 Flood Vision: Flood Mapping and Damage Segmentation System

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green?logo=nvidia)
![Status](https://img.shields.io/badge/Status-In_Development-yellow)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> **Flood Vision** - A deep learning–based computer vision system for flood mapping and damage assessment using aerial imagery.

---

## 🧭 Overview

**Flood Vision** is a computer vision system designed to automatically analyze aerial or drone footage from flood-affected regions.  
It uses **semantic segmentation** to identify and label key terrain types such as **water**, **flooded buildings**, **vegetation**, and **roads**, helping researchers and responders estimate affected areas efficiently.

The project focuses on **robust image segmentation** and **high-speed inference** on CUDA-enabled devices.

---

## ⚙️ Features Implemented

- ✅ 4-class **semantic segmentation model**
- ✅ **Weighted loss** for handling class imbalance
- ✅ **CUDA acceleration** for efficient GPU training
- ✅ **Video and image inference**
- ✅ **Color-coded visual outputs**
- ✅ **FloodNet dataset** integration
- ✅ Modular and extendable PyTorch architecture

---

## 🧪 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python 3.13 |
| **Framework** | PyTorch |
| **GPU Acceleration** | NVIDIA CUDA |
| **Data Augmentation** | Albumentations |
| **Video/Image Processing** | OpenCV |
| **Visualization** | Matplotlib |
| **Loss Function** | Weighted Cross-Entropy |

---

## 🗂️ Dataset

This project uses the **FloodNet** dataset for supervised training.

📚 **Dataset Link:**  
[https://github.com/BinaLab/FloodNet-Supervised_v1.0](https://github.com/BinaLab/FloodNet-Supervised_v1.0)

### 📖 Citation

```bibtex
@ARTICLE{9460988,
 author={Rahnemoonfar, Maryam and Chowdhury, Tashnim and Sarkar, Argho and Varshney, Debvrat and Yari, Masoud and Murphy, Robin Roberson},
 journal={IEEE Access}, 
 title={FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding}, 
 year={2021},
 volume={9},
 pages={89644-89654},
 doi={10.1109/ACCESS.2021.3090981}
}
````

---

## 🧰 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yashpotdar-py/flood-vision.git
cd flood-vision
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ (Optional) Verify CUDA availability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🏋️‍♂️ Training

Train the segmentation model with:

```bash
python -m train.train_segmentation_4class
```

The training script:

- Loads images and masks from the FloodNet dataset
- Applies Albumentations-based augmentations
- Uses a **weighted loss** for balanced learning
- Logs metrics like IoU and validation loss

---

## 🔍 Inference

Run segmentation on an image or video:

```bash
python -m inference.run_inference --input data/videos/test1.mp4 --output results/output.mp4
```

### Color Legend

| Class             | Color     | Description |
| ----------------- | --------- | --------------------|
| Water             | 🟦 Blue   | Flooded water areas|
| Flooded Buildings | 🟥 Red    | Buildings under water|
| Vegetation & Roads| 🟩 Green  | Trees, fields, grass, Streets and infrastructure|

The processed video will be saved in `results/output_videos/`.

---

<!-- ## 📸 Example Outputs

### Segmentation Example (Placeholder)

![Segmentation Result](docs/images/sample_segmentation.png)

### Validation Results (Placeholder)

![Training Graphs](docs/images/training_metrics.png)

*(Place your actual result screenshots inside `docs/images/`.)*

--- -->

## 📁 Project Structure

```plaintext
flood-vision/
│
├── data/
│   ├── images/
│   ├── masks/
│   └── videos/
│
├── train/
│   ├── train_segmentation_4class.py
│   └── utils/
│
├── inference/
│   ├── run_inference.py
│
├── models/
│   ├── unet.py
│   └── loss_utils.py
│
├── results/
│   └── output_videos/
│
└── README.md
```

---

## 📊 Sample Metrics (Training Snapshot)

| Metric          | Value |
| --------------- | ----- |
| Mean IoU        | 0.74  |
| Pixel Accuracy  | 0.90  |
| Validation Loss | 0.27  |

---

## 👨‍💻 Authors

**Developed by:**

- **Yash Potdar** - Model architecture & deployment, CUDA optimizations
- **Sahil Pawar** - Data preprocessing, model training, testing& evaluation
- **Akanksha Singh** - Web interface design, documentation, and system integration

**Affiliation:**

🎓 Final-Year B.E. Project,
Department of Artiticial Intelligence & Data Science
AISSMS Institute of Information Technology (AISSMS)

---

## 📜 License

This project is intended for **academic and research purposes**.
The **FloodNet dataset** belongs to its original authors.
Please cite the dataset and this repository when used in derivative works.

---

## 🙏 Acknowledgments

- [FloodNet Dataset](https://github.com/BinaLab/FloodNet-Supervised_v1.0)
- [PyTorch](https://pytorch.org/)
- [Albumentations](https://albumentations.ai/)
- [OpenCV](https://opencv.org/)

---

> *"Mapping floods through vision - towards smarter, data-driven disaster response."*

---
