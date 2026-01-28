# 🗑️ Waste Classification using Transfer Learning (VGG16)

This project implements an image classification model to distinguish **organic** and **recyclable** waste using **transfer learning with VGG16**, pretrained on ImageNet.

The focus is on building a **realistic end-to-end machine learning pipeline**, including data preparation, training, evaluation, visualization, and model export.

---

## 📌 Project Overview

- **Task:** Binary image classification (Organic vs Recyclable)
- **Model:** VGG16 (pretrained on ImageNet)
- **Framework:** TensorFlow / Keras
- **Dataset size:** ~30,000 images
- **Evaluation:** Confusion Matrix & Classification Report
- **Output:** Trained model + visualizations

---

## 🧠 Model Architecture

- Base model: **VGG16**
- Pretrained weights: ImageNet
- Custom classifier head:
  - Flatten
  - Dense (256, ReLU)
  - Dropout (0.5)
  - Dense (2, Softmax)
- Optimizer: Adam
- Loss function: Categorical Crossentropy

---

## 📂 Dataset Structure


