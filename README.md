# Simple-CNN-with-PyTorch

A convolutional neural network (CNN)-based classifier built using PyTorch to identify malware families from grayscale images derived from executable files. The project uses the Malimg dataset, which transforms malware binaries into 64x64 images, making it suitable for visual classification.

---

## Objective

The goal of this project is to train a CNN that can classify malware images into their corresponding families based on spatial and texture patterns in grayscale images.

---

## Dataset Overview

- **Dataset**: [Malimg](https://www.kaggle.com/datasets/manmandes/malimg)
- **Format**: Grayscale images (1 channel)
- **Size**: All images resized to `64x64`
---


## Training Details

| Hyperparameter       | Value         |
|----------------------|---------------|
| Epochs               | 15            |
| Batch size           | 32            |
| Learning rate        | 0.001         |
| Optimizer            | Adam          |
| Loss function        | CrossEntropy  |
| Train/Test split     | 80% / 20%     |

---

## Evaluation Metric

The final model performance is evaluated using:

- **Accuracy**
- (Other metrics like Precision/Recall can be added with `sklearn.metrics.classification_report` if needed)

---

## How to Run

### 1. Load Dataset (Colab recommended)

Download the Malimg dataset from Kaggle:

```python
!pip install kagglehub
import kagglehub
path = kagglehub.dataset_download("manmandes/malimg")
```

### 2. Train the Model

Run the training cells in the notebook. Adjust epochs and learning rate as needed.

---

## Key Learnings

* Image classification using CNNs
* Working with custom PyTorch `Dataset` and `DataLoader`
* CNN architecture design for binary image data
* Evaluating deep learning models in malware classification tasks
