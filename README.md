# Unsupervised Anomaly Detection in Industrial Textures

**Author:** Paweł Resiak 
**Institution:** Sapienza University of Rome | A.I. Lab: Computer Vision and NLP

This project implements an unsupervised anomaly detection framework for industrial surface inspection. By leveraging a Convolutional Autoencoder (CAE) trained exclusively on nominal (defect-free) textures, the network learns to reconstruct healthy spatial patterns. During inference, the model fails to reconstruct unfamiliar structural defects, allowing the system to isolate and localize anomalies using a dual-thresholding pipeline based on pixel-level reconstruction errors.


## 📁 Directory Structure & Dataset

This project utilizes a subset of the **MVTec AD** dataset. **You do not need to download the entire dataset to evaluate the model.** A `demo_data/` folder containing a small sample of images, along with pre-trained network weights (`autoencoder_weights.pth`), is included in this archive for quick testing.

If you wish to train the autoencoder from scratch or evaluate it on the full dataset, you can download the relevant texture categories (e.g., wood, leather, grid) from the [official MVTec AD website](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract them into a separate `data/` directory.

Ensure your project directory is organized as follows before executing the scripts:

```text
AILAB_Code/
├── demo_data/             # Included: Small sample for quick evaluation
│   └── wood/
│       └── test/          # Example 'good' and 'scratch' images
├── data/                  # Not included: Create this if using the full MVTec dataset
│   ├── grid/
│   ├── leather/
│   └── wood/
├── weights/               # Saved model checkpoints
│   └── autoencoder_weights.pth 
├── evaluate.py
├── heatmap.py
├── model.py
├── train.py
├── requirements.txt
└── README.md
```


---

## 🚀 Quick Start (Environment Setup)

To avoid conflicts with global system packages, it is highly recommended to run this project inside an isolated virtual environment (`venv`).

**Please open your terminal inside the main project directory (where the `.py` files are located)** and execute the following commands:

### 1. Create and Activate the Virtual Environment

**For macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

**For Windows:**

```cmd
python -m venv venv
venv\Scripts\activate

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

---



```markdown
## 🛠️ Usage & Execution

The scripts are hardware-agnostic and will automatically utilize the optimal available backend (CUDA for NVIDIA, MPS for Apple Silicon, or CPU).

---

### Pathway A: Quick Testing (Using Provided Demo Data)

If you want to quickly verify that the scripts run correctly without downloading the entire dataset, use the included `demo_data/` and the pre-trained weights.

**1. Generate an Anomaly Heatmap:**
Output a side-by-side comparison of the original image, CAE reconstruction, and the localized anomaly heatmap.

```bash
python heatmap.py --model_path ./weights/autoencoder_weights.pth --image_path ./demo_data/wood/test/scratch/007.png

```

**2. Run Evaluation Metrics:**
Verify the evaluation pipeline (Accuracy, Precision, Recall, Confusion Matrix) using the demo subset. *(Note: Metrics generated on this tiny subset are for code verification purposes only).*

```bash
python evaluate.py --test_dir ./demo_data/wood/test --model_path ./weights/autoencoder_weights.pth

```

---

### Pathway B: Full Pipeline (Requires MVTec AD Download)

If you have downloaded the full dataset into the `data/` directory, you can train the model from scratch and reproduce the full evaluation metrics.

**1. Training the Autoencoder:**
Train the model from scratch on healthy textures. This will generate a new weights file and plot the learning curve.

```bash
python train.py --data_dir ./data/wood/train --save_path ./weights/autoencoder_weights_new.pth --epochs 30

```

**2. Evaluating on the Full Test Set:**
Assess the model's performance over all test images.

```bash
python evaluate.py --test_dir ./data/wood/test --model_path ./weights/autoencoder_weights.pth

```

> 💡 **Note on Hyperparameters:** You can easily perform a Grid Search to adjust the sensitivity of the anomaly detection by overriding the default thresholds during evaluation:

```bash
python evaluate.py --test_dir ./data/wood/test --model_path ./weights/autoencoder_weights.pth --pixel_threshold 0.15 --min_area 20

```

**3. Generating Heatmaps for Custom Samples:**

```bash
python heatmap.py --model_path ./weights/autoencoder_weights.pth --image_path ./data/wood/test/combined/004.png



```

```

```


