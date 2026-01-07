# 📷 Real vs AI-Generated Image Detection System

## 📌 Overview

This project implements a **computer vision–based image authenticity detection system** that classifies whether an image is **real** or **AI-generated**.
The system is designed to study how different deep learning architectures perform on the task of **synthetic image detection**, an increasingly important problem in the context of generative AI.

Multiple modeling strategies are implemented and systematically compared, including:

* Custom CNN architectures trained from scratch
* Transfer learning using pre-trained **VGG16** and **ResNet50**

The emphasis of this project is on **model comparison, robustness, and deployment-aware trade-offs**, rather than relying on a single architecture.


## 🎯 Objectives

* Build baseline CNN models for real vs synthetic image classification
* Apply transfer learning using ImageNet-pretrained backbones
* Compare models using standard classification metrics
* Analyze trade-offs between **accuracy, robustness, and computational cost**
* Identify a practical model choice for real-world deployment scenarios

---

## 📂 Dataset

### 🔗 Source

* **CIFAKE – Real and AI-Generated Synthetic Images**
* Kaggle:
  [https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

### 📁 Dataset Structure

The dataset is automatically downloaded using `kagglehub` and organized as follows:

```text
cifake/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

* Binary classification task: **REAL vs FAKE**
* Images are low-resolution, emphasizing robustness over fine-grained details

---

## 🗂️ Project Structure

```text
real-and-ai-generated-synthetic-images/
│
├── dl_cifake/
├── runs/
├── trained_models/
│
├── CNN_implementation_design1.ipynb
├── CNN_implementation_design2.ipynb
├── TransferLearning_VGG16.ipynb
├── TransferLearning_ResNet50.ipynb
└── README.md
```

Each notebook is self-contained and focuses on a specific modeling strategy.

---

## 🧠 Models and Methods

### 🔹 Design 1 — Custom CNN (PyTorch)

A lightweight CNN designed as a strong baseline.

**Key characteristics**

* 3 convolutional blocks (Conv + BatchNorm + LeakyReLU + MaxPooling)
* Fully connected layers with Dropout
* Input size: `32 × 32`
* Experiments with and without data augmentation

This model prioritizes **efficiency and simplicity**, making it suitable for environments with limited computational resources.

---

### 🔹 Design 2 — CNN with ReLU (TensorFlow / Keras)

A CNN variant implemented using Keras.

**Key characteristics**

* Conv2D + ReLU + MaxPooling
* Global Average Pooling
* Built-in data augmentation layers
* Binary classification with Sigmoid output

This design serves as a comparative baseline to study the impact of architecture and activation choices.

---

### 🔹 Transfer Learning Models

#### ✅ VGG16

* Pre-trained on ImageNet
* Frozen backbone with fine-tuning of upper layers
* Strong performance but **high computational cost**

Notebook:

```text
TransferLearning_VGG16.ipynb
```

---

#### ✅ ResNet50

* ImageNet-pretrained ResNet50 backbone
* Fine-tuned using:

  * Data augmentation
  * Weighted loss
  * Learning rate scheduling
  * Mixed precision training
* Best overall performance among all evaluated models

Notebook:

```text
TransferLearning_ResNet50.ipynb
```

---

## ⚙️ Installation & Usage

### 1️⃣ Install Dependencies

```bash
pip install torch torchvision tensorflow kagglehub scikit-learn matplotlib seaborn tqdm
```

---

### 2️⃣ Run Experiments

Each notebook automatically:

* Downloads the dataset
* Performs preprocessing
* Trains the model
* Evaluates on the test set

Recommended execution order:

1. `CNN_implementation_design1.ipynb`
2. `CNN_implementation_design2.ipynb`
3. `TransferLearning_VGG16.ipynb`
4. `TransferLearning_ResNet50.ipynb`

---

## 📊 Experimental Results

### 🔬 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

---

### 📌 Overall Performance Comparison

| Model          | Accuracy   | Precision | Recall   | F1-score |
| -------------- | ---------- | --------- | -------- | -------- |
| Design 1 – CNN | 0.9399     | 0.9401    | 0.9400   | 0.9399   |
| Design 2 – CNN | 0.7993     | 0.9334    | 0.6445   | 0.7625   |
| VGG16          | 0.9351     | 0.9502    | 0.9182   | 0.9339   |
| **ResNet50**   | **0.9774** | **0.98**  | **0.98** | **0.98** |

---

## 📊 Result Analysis

* The **custom CNN (Design 1)** achieves strong performance (~94%) with relatively low computational cost, making it suitable for resource-constrained environments.
* **Design 2** performs significantly worse, particularly in recall, indicating limited robustness under augmentation and architectural constraints.
* **VGG16** delivers high accuracy but requires substantially more computation and training time.
* **ResNet50** provides the best balance between accuracy and robustness, achieving nearly **98% accuracy** with well-balanced precision and recall.

These results highlight the effectiveness of deeper residual architectures for detecting AI-generated images on the CIFAKE dataset.

---

## 🚀 Deployment Considerations

* **ResNet50-based model**
  Recommended when detection accuracy and robustness are the primary objectives.

* **Custom CNN (Design 1)**
  A viable alternative for scenarios with limited computational resources.

This comparison supports informed decision-making when selecting models for real-world deployment.

---

## 🔮 Future Work

* Evaluate generalization on newer AI-generated image datasets
* Explore frequency-domain and noise-based features
* Extend the system toward real-time or API-based deployment

---

## 👨‍💻 Author

**Ly Nguyen**
Purpose: Computer Vision System Development and Model Benchmarking


