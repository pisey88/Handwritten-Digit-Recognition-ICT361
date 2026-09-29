# Assignment 1: Handwritten Digit Recognition

This repository contains a PyTorch implementation of a **Multiclass Logistic Regression** model trained on the **MNIST handwritten digit dataset** (digits 0–9).

---

## 1. Dataset and Preprocessing

* **How many training and test images are there?**

  * **Training Set:** 60,000 images
  * **Test Set:** 10,000 images

* **What are the image dimensions and class labels?**

  * **Dimensions:** `28 × 28` pixels (grayscale, 1 channel)
  * **Class Labels:** 10 classes representing the digits `0` through `9`

* **Why do we convert each image into 784 values?**

  * Multiclass Logistic Regression works with a 1D feature vector. Each `28 × 28` image is flattened into `784` pixel values, making it suitable for the linear transformation:

  `Y = XW + b`

* **Why do we divide pixel values by 255?**

  * Raw pixel values range from `0` to `255`. Dividing by `255.0` normalizes the values to the range `[0.0, 1.0]`. This improves numerical stability and helps the model train more efficiently.

* **Why must training and test data remain separate?**

  * Training and test data must remain separate to prevent **data leakage**. The test set is used only for final evaluation, providing a more reliable measurement of how well the model performs on unseen data.

### Dataset Shapes

```text
Training Features (X_train): (60000, 784)
Training Labels   (y_train): (60000,)
Test Features     (X_test):  (10000, 784)
Test Labels       (y_test):  (10000,)
```

### Sample Digits (0–9)

<img width="711" height="170" alt="Screenshot 2026-09-29 230646" src="https://github.com/user-attachments/assets/63b6a947-d403-4293-a8dd-2417747b1e52" />
<img width="990" height="476" alt="image" src="https://github.com/user-attachments/assets/dd3cf845-fb61-4353-ae26-876224c933c3" />


---

## 2. Model and Training

* **Why does the model use 784 inputs and 10 outputs?**

  * **784 Inputs:** Each `28 × 28` image contains 784 pixel values after flattening.
  * **10 Outputs:** Each output represents the logit score for one of the 10 digit classes (`0–9`).

* **Why do we use `CrossEntropyLoss`?**

  * `CrossEntropyLoss` is commonly used for multiclass classification. It combines `LogSoftmax` and `NLLLoss` in a numerically stable way and measures how well the predicted class scores match the correct labels.

* **What batch size, learning rate, and number of epochs did you use?**

  * **Batch Size:** `64`
  * **Learning Rate:** `0.0005`
  * **Number of Epochs:** `40`
  * **Optimizer:** Adam
  * **Weight Decay:** `0.0001`

* **How did the training loss change? What does this tell you?**

  * The training loss decreased quickly during the first few epochs, dropping from approximately `0.70` to `0.33`. It then continued to decrease gradually and stabilized around `0.25` by epoch 40.
  * This shows that the model was learning effectively and that the optimization process converged during training.

### Model Architecture

```text
MNISTLogisticRegression(
  (linear): Linear(in_features=784, out_features=10, bias=True)
)
```

**Total Trainable Parameters:**

```text
(784 × 10) + 10 = 7,850
```

### Training Loss Curve

<img width="711" height="85" alt="image" src="https://github.com/user-attachments/assets/3044d114-e17c-4ed9-a699-9a393516d3d7" />

<img width="552" height="460" alt="image" src="https://github.com/user-attachments/assets/b24dd212-1c51-473d-a0ba-c10cfd6c2a3a" />

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/8ba534be-9677-4ae9-8545-3a3b702e7b67" />

---

## 3. Evaluation and Predictions

* **What test accuracy did you achieve?**

  * **Test Accuracy:** **92.68%**

* **How does it compare with the most-frequent-class baseline?**

  * **Baseline Accuracy:** **11.35%**
  * The baseline is obtained by always predicting digit `1`, which is the most frequent class in the test set.
  * The Logistic Regression model achieved **92.61%**, which is **81.26 percentage points higher** than the baseline.

* **Which digit has the highest recall?**

  * **Digits `0` and `1`** tied for the highest recall at approximately **98%**.
  * Digit `0`: `962 / 980` correctly recalled
  * Digit `1`: `1111 / 1135` correctly recalled

* **Which digit has the lowest recall?**

  * **Digit `5`** had the lowest recall at approximately **86%**.
  * Correctly classified: `763 / 892`

### What Might Explain the Incorrect Predictions?

Several factors can cause incorrect predictions:

1. **Visual Similarity**

   * Some digits have similar shapes and stroke patterns.
   * For example, `5` and `3`, `4` and `9`, and `2` and `8` can look similar depending on the handwriting style.

2. **Linear Decision Boundaries**

   * Logistic Regression works directly with the flattened `784` pixel values.
   * It does not use spatial feature extraction like a Convolutional Neural Network (CNN).
   * As a result, it may have difficulty handling differences in stroke shape, slant, thickness, and position.

3. **Handwriting Variation**

   * Different people write the same digit in different ways.
   * Unusual or unclear handwriting can therefore lead to incorrect predictions.


### Does High Test Accuracy Guarantee Correct Predictions on Camera Images?

**No.** A high MNIST test accuracy does not guarantee that the model will perform equally well on handwritten digits captured using a camera.

Camera images can introduce **domain shift**, including:

* Different lighting conditions
* Shadows
* Background noise
* Camera noise
* Motion blur
* Perspective distortion
* Different image sizes
* Poor cropping or positioning
* Different handwriting styles

MNIST images are relatively clean and standardized, while real-world camera images can have much more variation.

Therefore, additional preprocessing such as **cropping, resizing, grayscale conversion, normalization, and background cleaning** may be necessary before using the model with camera images.

<img width="627" height="426" alt="image" src="https://github.com/user-attachments/assets/018d35d6-fe5d-45b1-89f1-4708ef65ea86" />

<img width="841" height="690" alt="image" src="https://github.com/user-attachments/assets/b2e33476-008c-4036-b64f-cc93d70357a6" />

<img width="766" height="790" alt="image" src="https://github.com/user-attachments/assets/da918e4d-814e-4c03-98fc-36a98fe217dd" />

---

## 5. Conclusion

This project demonstrates how **Multiclass Logistic Regression** can be used to recognize handwritten digits from the MNIST dataset.

The model achieved a **92.61% test accuracy**, showing that a simple linear classification model can perform well on a standardized image dataset. However, the model still struggles with visually similar digits, especially digit `5`.

The results also show that strong performance on MNIST does not necessarily translate directly to real-world camera images because of differences in image quality, background, lighting, and handwriting style.
