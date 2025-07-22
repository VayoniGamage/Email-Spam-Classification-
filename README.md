# 📧 Email Spam Classification using Support Vector Machine (SVM)

This project is part of **CCS4340 – Machine Learning** Lab Sheet 06. The aim is to implement a **Support Vector Machine (SVM)** classifier using Python and scikit-learn to detect whether an email is spam based on its metadata.

---

## 🎯 Objective

- Understand the theory behind Support Vector Machines (SVMs).
- Implement a linear SVM classifier in Python.
- Visualize the decision boundary.
- Evaluate the classifier using real-world metrics.

---

## 📚 Key Concepts

- **Support Vectors**: Critical data points that define the optimal separating hyperplane.
- **Margin Maximization**: SVM seeks the hyperplane with the maximum margin between classes.
- **Hyperplane**: A decision boundary that separates different classes in the feature space.
- **Kernel Trick**: Enables non-linear decision boundaries using kernel functions (Linear, Polynomial, RBF).

---

## 🧠 Applications of SVM

- Face detection
- Text classification
- Image recognition
- Bioinformatics

---

## 🗂 Dataset

- **Filename**: `email_spam_dataset.csv`
- **Features**:
  - `email_length`
  - `num_links`
  - `num_attachments`
  - `num_exclamations`
- **Target**: `is_spam` (0: Not Spam, 1: Spam)

---

## 🧪 Lab Activities

### ✅ Step 1: Load Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
