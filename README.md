# Image Recognition – Gender Detection 👤🤖

This repository contains the **Gender Detection** project developed as part of my **Capstone Project for the Machine Learning Engineer Nanodegree (Udacity)**.
The goal of this project is to build and evaluate a **deep learning–based image classification model** capable of predicting **gender from facial images**.

Due to **technical constraints in Kaggle Notebooks**, the original Udacity submission code has been **modified** to ensure compatibility and efficient execution.

<img width="1902" height="566" alt="image" src="https://github.com/user-attachments/assets/bb465075-b972-4b0c-9e09-962fdadfeead" />
<img width="1644" height="276" alt="image" src="https://github.com/user-attachments/assets/250a97e4-c8e6-42b6-9135-776c5348bacf" />

## 📌 Project Overview

* **Task:** Binary image classification (Male / Female)
* **Approach:** Convolutional Neural Networks (CNN) using Keras & TensorFlow
* **Platform:** Kaggle Notebook
* **Frameworks:** TensorFlow, Keras, NumPy, Matplotlib
* **Dataset Source:** Kaggle (facial image dataset)


### Memory Constraints

Kaggle Notebooks have **limited RAM and GPU memory**, which restricts:

* Loading the full dataset
* Training large models
* Long training epochs


## 🧠 Model Architecture

The model is a **Convolutional Neural Network (CNN)** consisting of:

* Convolutional layers with ReLU activation
* MaxPooling layers for spatial downsampling
* Dropout layers to prevent overfitting
* Fully connected (Dense) layers
* Sigmoid activation in the output layer for binary classification



## 🏗️ Data Pipeline

1. Load images into NumPy arrays
2. Normalize pixel values
3. Encode gender labels
4. Split data into training and validation sets
5. Apply data augmentation using `ImageDataGenerator.flow()`



## 📊 Evaluation Metrics

The model performance is evaluated using:

* Training Accuracy
* Validation Accuracy
* Loss Curves
* Visual comparison of predictions


## 🔍 Key Modifications from Udacity Submission

* Replaced `flow_from_directory()` with `flow()`
* Manual label handling
* Reduced dataset size
* Optimized memory usage for Kaggle environment


## 🎓 Learning Outcomes

* Practical experience with CNN-based image classification
* Handling real-world platform constraints
* Memory-efficient deep learning workflows
* Understanding data generators in Keras


## 📌 Future Improvements

* Use full dataset on a local or cloud GPU environment
* Experiment with transfer learning (ResNet, MobileNet, VGG)
* Improve accuracy with hyperparameter tuning
* Add test set evaluation and confusion matrix

