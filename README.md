# Fashion Style Classification Using CNNs & Transfer Learning 👗🧥

This project focuses on classifying fashion outfit styles from images using Convolutional Neural Networks (CNNs) and transfer learning. The model is trained and evaluated on the [FashionStyle14 dataset](https://esslab.jp/~ess/en/data/fashionstyle14/), which consists of images labeled with different fashion styles.

---

##Motivation

Fashion recognition tasks are a growing area of computer vision. The goal of this project was to:
- Build a CNN model to classify images into one of the fashion style categories.
- Explore transfer learning with ResNet50 to improve model performance.
- Compare performance between a custom CNN and a fine-tuned pretrained model.

---

##Dataset

**FashionStyle14**: Contains over 14,000 labeled images across 14 distinct fashion style categories.

- For final model performance, a subset of **10 classes** was selected to improve generalization.
- Preprocessing includes resizing, normalization, and data augmentation (random crop, flip, rotation).

---

##Models

### 1. Custom CNN
- Basic architecture with a few convolutional and pooling layers.
- Achieved ~50% accuracy across 14 classes.

### 2. Transfer Learning with ResNet50
- Used a pretrained ResNet50 model from `torchvision.models`.
- Fine-tuned only the final layers.
- Achieved ~80% accuracy across 10 classes.

---

##Evaluation
- Accuracy, confusion matrix, and loss curves used to evaluate performance.
- Training and validation splits created from dataset using PyTorch’s `Dataset` and `DataLoader`.

---

##Technologies Used

- **Python**
- **PyTorch**
- **NumPy**, **Pandas**
- **Matplotlib**, **Seaborn**
- **Jupyter Notebooks**

---

##Future Improvements

Extend back to all 14 original style classes with more data balancing
Try alternative pretrained models like EfficientNet or MobileNet
Deploy a demo app using Streamlit or Flask

---

📈 Results

Model | Accuracy |
--- | --- |
Custom CNN | ~50% |
ResNet50 (FT)|	~80% |



