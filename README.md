# 🧠 EnvDeepFake

**EnvDeepFake** is a deep learning framework designed for detecting **Deep Video Frame Interpolation (VFI)** in environmental or “in-the-wild” video scenarios.  
It provides an end-to-end pipeline — from preprocessing to model training and testing — for identifying interpolated (fake) frames in videos.

---

## 📚 Table of Contents
- [Background](#background)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Training](#2-training)
  - [3. Testing / Evaluation](#3-testing--evaluation)
- [Configuration & Hyperparameters](#configuration--hyperparameters)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🧩 Background

**Video Frame Interpolation (VFI)** techniques generate intermediate frames between real frames to make videos smoother.  
While useful in video editing or frame rate conversion, these methods can also be misused for **video forgeries** — making fake content appear more realistic.

**EnvDeepFake** addresses this by detecting VFI-based manipulations using a deep neural network trained to identify interpolation artifacts even in environmental conditions (e.g., varying lighting, motion blur, compression).

---

## 🚀 Features

- 🔹 **Preprocessing tools** for interpolated and natural videos (`preprocess_env.py`)
- 🔹 **Custom CNN architecture** for deepfake/interpolation detection (`model.py`)
- 🔹 **Training and evaluation scripts** (`train.py`, `test.py`)
- 🔹 **Utility functions** for logging, metrics, and reproducibility (`utils.py`, `logger.py`)
- 🔹 **Submission-ready output** (`submission.txt`)
- 🔹 Modular and extensible — easy to adapt to new datasets or models

---

## 🗂️ Repository Structure

EnvDeepFake/
│
├── train.py # Model training entry point
├── test.py # Evaluation / inference script
├── preprocess_env.py # Preprocessing of environmental/interpolated videos
├── model.py # CNN architecture definition
├── utils.py # Utility functions (data loaders, metrics, etc.)
├── logger.py # Logging and experiment tracking
├── submission.txt # Example or template submission file
└── requirements.txt # Dependencies (PyTorch, OpenCV, NumPy, etc.)


---

## ⚙️ Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/Sarthak-Shaurya/EnvDeepFake.git
cd EnvDeepFake

Step 2: Create and activate a virtual environment (optional)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Step 3: Install dependencies
pip install -r requirements.txt


💡 If no requirements.txt exists, create one including packages like:
torch, torchvision, numpy, opencv-python, tqdm, matplotlib, scikit-learn, pandas, librosa.

🧪 Usage
1️⃣ Preprocessing

Prepare your dataset of interpolated and real videos:

python preprocess_env.py \
  --input_dir path/to/raw_videos \
  --output_dir path/to/processed_data \
  --frame_rate 30


This script extracts and formats data for training and testing.

2️⃣ Training

Train your model using the preprocessed dataset:

python train.py \
  --data_dir path/to/processed_data \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001 \
  --save_dir checkpoints/


Model checkpoints and training logs will be saved automatically.

3️⃣ Testing / Evaluation

Evaluate the trained model on test data:

python test.py \
  --checkpoint checkpoints/model_best.pth \
  --test_data path/to/test_data \
  --output submission.txt


This computes detection metrics (accuracy, F1, ROC-AUC, etc.) and outputs predictions in a standardized format.

⚙️ Configuration & Hyperparameters

You can modify hyperparameters through command-line arguments or a configuration file.

Parameter	Description	Default
--epochs	Number of training epochs	50
--batch_size	Mini-batch size	32
--lr	Learning rate	0.001
--data_dir	Directory for input data	./data
--save_dir	Directory for saving checkpoints	./checkpoints
🎥 Dataset

The dataset should contain real and interpolated (fake) video samples.
Example structure:

data/
├── train/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/


Each folder should contain frames or clips extracted via preprocess_env.py.

⚠️ Ensure that interpolated videos are properly labeled to enable supervised training.

📊 Results

(Add your experimental results here once available.)
For example:

Metric	Accuracy	Precision	Recall	F1-score
Baseline CNN	89.3%	88.7%	90.1%	89.4%
Proposed Model	94.1%	93.8%	94.5%	94.1%

Add visualizations such as ROC curves or confusion matrices for clarity.

🤝 Contributing

Contributions are welcome!

Fork the repository

Create your feature branch:

git checkout -b feature/YourFeature


Commit your changes:

git commit -m "Add new feature"


Push to the branch:

git push origin feature/YourFeature


Open a Pull Request

🪪 License

This project is licensed under the MIT License.
See the LICENSE
 file for details.

📬 Contact

Author: Sarthak Shaurya

📘 GitHub: github.com/Sarthak-Shaurya/EnvDeepFake

✉️ Email: (optional — add if you’d like to share contact info)
