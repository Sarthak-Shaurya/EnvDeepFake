# 🔊 EnvDeepFake: Audio Deepfake Detection via Dual-Stream Fusion

**EnvDeepFake** is a deep learning–based framework for detecting **audio deepfakes** using a **dual-stream neural architecture**.  
The system jointly processes **raw audio waveforms** and **Mel-spectrograms** to detect speech forgeries, focusing on artifacts introduced by **voice cloning**, **speech synthesis**, or **frame interpolation** in compressed audio.



---

## 📘 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [System Description](#-system-description)
- [Installation](#-installation)
- [Usage](#-usage)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Training](#2-training)
  - [3. Evaluation / Testing](#3-evaluation--testing)
- [Configuration](#-configuration)

---

## 🧠 Overview

The **EnvDeepFake Audio** system detects manipulated or generated audio using a **dual-stream CNN + Transformer** architecture:

* **Raw-Waveform Stream** → Uses a pretrained **Wav2Vec2 encoder** followed by a **Conformer** for temporal context.
* **Mel-Spectrogram Stream** → Uses a **Swin-Tiny Transformer** backbone with **Graph Attention (GATv2)** pooling.
* **Fusion** → The two embeddings are fused into a joint representation and classified using an **OC-Softmax** head for binary detection.

This design captures both **fine-grained waveform irregularities** and **spectral texture inconsistencies** introduced during synthetic generation or frame interpolation.

---

## ⚙️ Key Features

* 🎧 **Dual-Stream Input:** Combines raw waveform and Mel-spectrogram features.
* 🧩 **Fusion Architecture:** Joint representation learning for enhanced detection.
* 🧠 **Pretrained Backbones:** Utilizes Wav2Vec2 and Swin-Tiny for efficient feature extraction.
* 🔁 **Audio Augmentation:** Noise addition, pitch shift, time-stretch, random gain.
* 📊 **Metrics:** Computes EER, AUC, Accuracy, Precision, Recall, F1.
* 💾 **Reproducible Pipeline:** Includes training, testing, and logging utilities.

---

## 🗂️ Repository Structure

```text
EnvDeepFake/
│
├── train.py           # Model training entry point
├── test.py            # Evaluation / inference script
├── preprocess_env.py  # Audio preprocessing & feature extraction
├── model.py           # Dual-stream CNN/Transformer model definition
├── utils.py           # Utility functions (metrics, loaders, EER computation)
├── logger.py          # Logging and experiment tracking
├── submission.txt     # Sample submission file
├── requirements.txt   # Dependencies (PyTorch, torchaudio, etc.)
└── README.md          # Project documentation
```

## 🧩 System Description
| **Component**      | **Description**                                                                 |
|---------------------|---------------------------------------------------------------------------------|
| **Sampling Rate**   | 16 kHz                                                                         |
| **Input Features**  | Raw audio waveform + log-Mel spectrogram                                       |
| **Model**           | Dual-Stream (Wav2Vec2 + Swin-Tiny + GAT)                                       |
| **Loss Function**   | OC-Softmax (margin-based cosine loss)                                          |
| **Augmentations**   | Gaussian noise, time-stretch, pitch-shift, gain variation                      |
| **Metrics**         | EER, AUC, Accuracy, Precision, Recall, F1-score                                |
| **GPU Used**        | NVIDIA RTX A4000 (16 GB VRAM)                                                  |
| **Parameters**      | ~128 Million                                                                   |

---


## ⚙️ InstallationStep 

Step 1: Clone the repository

```bash
git clone [https://github.com/Sarthak-Shaurya/EnvDeepFake.git](https://github.com/Sarthak-Shaurya/EnvDeepFake.git)
cd EnvDeepFake
```
Step 2: Create and activate a virtual environment

```bash
python -m venv venv
```

<b> On Windows </b>
venv\Scripts\activate

<b> On macOS/Linux </b>
source venv/bin/activate


Step 3: Install dependencies

```bash
pip install -r requirements.txt
```
💡 Note: Typical requirements include torch, torchaudio, timm, torch_geometric, numpy, librosa, scikit-learn, tqdm, and matplotlib.

🧪 Usage
1. Preprocessing \
Extract waveform and Mel-spectrogram features for all audio files.

```bash

python preprocess_env.py 
  --input_dir path/to/raw_audio 
  --output_dir path/to/processed_data 
  --sample_rate 16000
  ```
This script will normalize audio, extract Mel-spectrograms, and save tensors for model training.

2. Training\
Train the model using preprocessed features.

Bash
```
python train.py 
  --data_dir path/to/processed_data 
  --epochs 50 
  --batch_size 16 
  --lr 1e-4 
  --save_dir checkpoints
  ```
During training, the logger saves loss curves and metrics. The best checkpoint (based on validation EER) is saved automatically.

3. Evaluation / Testing\
Evaluate your model on a test set.

Bash
```
python test.py 
  --checkpoint checkpoints/model_best.pth 
  --test_data path/to/test_data 
  --output results.txt


```
Outputs include EER, AUC, accuracy, confusion matrix plots, and a submission file compatible with leaderboard formats.

## Configuration

| **Parameter**     | **Description**             | **Default**      |
|--------------------|-----------------------------|------------------|
| `epochs`         | Number of training epochs   | 50               |
| `batch_size`     | Batch size                  | 16               |
| `lr`             | Learning rate               | 1e-4             |
| `sample_rate`    | Audio sample rate           | 16000            |
| `save_dir`       | Checkpoint save path        | `./checkpoints`  |

## 👨‍💻 Authors

| **Name** | **Email** | **Affiliation** | **Role** |
|-----------|------------|-----------------|-----------|
| **Athar Ali** | [athar_a@ece.iitr.ac.in](mailto:athar_a@ece.iitr.ac.in) | Indian Institute of Technology Roorkee, Dept. of Electronics and Communication Engineering, Roorkee, Uttarakhand, India | 📨 Corresponding Author |
| **Jwalit Pandit** | [pandit_jt@ece.iitr.ac.in](mailto:pandit_jt@ece.iitr.ac.in) | Indian Institute of Technology Roorkee, Dept. of Electronics and Communication Engineering, Roorkee, Uttarakhand, India | — |
| **Sarthak Shaurya** | [sarthak_s@ece.iitr.ac.in](mailto:sarthak_s@ece.iitr.ac.in) | Indian Institute of Technology Roorkee, Dept. of Electronics and Communication Engineering, Roorkee, Uttarakhand, India | — |



