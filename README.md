# Dyslexia Detection System using ResNet50V2

## 🚀 Project Overview
This project is an early-stage screening tool designed to identify potential signs of dyslexia through image-based analysis. By leveraging Deep Learning, the system provides a high-accuracy, low-friction method for initial screening, making diagnostic resources more accessible.

## 📊 Technical Performance
* **Model Architecture:** ResNet50V2 (Transfer Learning).
* **Accuracy:** Achieved **97% accuracy** on test datasets.
* **Optimization:** Implemented oversampling techniques to address dataset imbalance, ensuring reliable detection across all classes.
* **Deployment:** Integrated with **Streamlit** for a real-time, user-friendly web interface.

## 🛠️ Tech Stack
* **Language:** Python.
* **Deep Learning:** TensorFlow, Keras.
* **Computer Vision:** OpenCV.
* **Web Framework:** Streamlit.

## 📦 Model Access
Due to GitHub's file size limitations, the trained model file (**resnet50_oversampled.keras**) is hosted in the **Releases** section of this repository.
1. Navigate to the **Releases** tab on the right sidebar.
2. Download the `.keras` model file.
3. Place it in the root directory of the project before running the application.

## 🧪 How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/SahithiK792/Dyslexia-Detection-ResNet50V2.git](https://github.com/SahithiK792/Dyslexia-Detection-ResNet50V2.git)# Dyslexia-De
   tection-ResNet50V2
Early-stage dyslexia screening tool using a ResNet50V2 deep learning model with 97% accuracy, deployed via Streamlit
Install dependencies: This command installs all necessary libraries like TensorFlow and Streamlit at once:
pip install -r requirements.txt
Launch the application: This starts the local web server and opens the screening tool in your browser:
streamlit run app.py
