

## 🧠 Face Mask Detection using CNN

This project is a **Convolutional Neural Network (CNN)** model trained to detect whether a person is **wearing a face mask** or **not** using image classification.

---

### 📁 Dataset

The model uses the [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) by **Omkargurav** on Kaggle, which contains:

* `with_mask/`: Images of people wearing face masks
* `without_mask/`: Images of people without face masks

---

### 📊 Model Architecture

A custom CNN model built with TensorFlow/Keras:

* **Conv2D** layers with ReLU activation
* **MaxPooling2D** layers
* **Dense** fully connected layers
* **Dropout** layers for regularization
* Final layer with `softmax` for binary classification (`with_mask`, `without_mask`)

---

### 🔁 Training Details

* Input image size: `128 x 128 x 3`
* Train/Test split: `80% / 20%`
* Normalized pixel values to `[0, 1]`
* Loss: `Sparse Categorical Crossentropy`
* Optimizer: `Adam`
* Epochs: `5`

---

### 📈 Evaluation

* Training and validation accuracy plotted
* Tested on unseen data (20% test set)
* Prediction example included using a sample image

---

### 🧪 Prediction Example

You can upload an image of a face and the model will output:

* 😷 “The person is **wearing a mask**.”
* ❌ “The person is **not wearing a mask**.”

---

### 💾 How to Save the Model

In the Kaggle notebook:

```python
model.save('face_mask_model.h5')
```

Then download it from the sidebar or use:

```python
from IPython.display import FileLink
FileLink('face_mask_model.h5')
```

---

### 🧑‍💻 How to Deploy (Streamlit)

To deploy the model using Streamlit:

1. Create `app.py` (your Streamlit app file)
2. Place `face_mask_model.h5` in the same folder
3. Run locally:

```bash
streamlit run app.py
```

4. Or deploy via [Streamlit Cloud](https://share.streamlit.io/)

---

### 📁 Project Structure

```
face-mask-detection/
│
├── app.py                   # Streamlit app (optional)
├── face_mask_model.h5       # Saved CNN model
├── README.md                # Project readme
└── requirements.txt         # (Optional) for deployment
```

---

### ✅ Requirements

```txt
tensorflow
numpy
opencv-python
matplotlib
Pillow
streamlit
```

---

### 📌 Credits

---
