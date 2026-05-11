# Plant Disease Detection & Recovery System

An AI-powered web application designed for Ethiopian farmers to identify plant diseases and get recovery recommendations in **English, Amharic, and Tigrinya**.

## 🚀 Live
https://plant-disease-detection-recovery-system.streamlit.app/

## ✨ Features
- **Multilingual Support:** English, Amharic, and Tigrinya.
- **Smart Image Analysis:** Detects 38 different plant diseases using a CNN model.
- **Side-by-Side Results:** View the uploaded leaf and the diagnosis simultaneously.
- **Confidence Filtering:** Prevents false predictions on non-leaf images.

## 🛠️ Tech Stack
- **AI/ML:** TensorFlow, Keras, NumPy
- **Frontend:** Streamlit, Custom CSS
- **Model Hosting:** Hugging Face Hub
- **Deployment:** Streamlit Cloud
## 📊 Dataset Information

This project is powered by the **New Plant Diseases Dataset**, an extensive collection of leaf images used to train the AI model. 

*   **Source:** [New Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
*   **Total Images:** ~87,000 images.
*   **Classes:** 38 distinct classes (including healthy and diseased leaves).
*   **Crops Covered:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

### Why this dataset?
This dataset is a reconstructed and augmented version of the original **PlantVillage** dataset. It provides a robust foundation for deep learning by providing a high variety of leaf backgrounds and orientations, which significantly improves the AI's accuracy when used in real-world Ethiopian farming conditions.

## 📂 Project Structure
- `main.py`: The core application script.
- `notebooks/`: Research and model training process.
- `src/`: UI design elements and images.
- `requirements.txt`: List of Python dependencies.

## 📝 How to Run Locally
1. Clone the repo: `git clone https://github.com/tedd12t/Plant-disease-detection-recovery-system.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run main.py`

---
**Developed by Tedros Nigus**


