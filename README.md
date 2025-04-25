# 🧠 FeelTheWaves: EEG-Based Emotion Recognition Web App

NeuroFeel is a Streamlit-powered web application that classifies emotional states from EEG signal features. Upload a CSV file (like the `emotions.csv` dataset from Kaggle), train different ML models, visualize performance, and even test real-time predictions from manual input.

## 🚀 Features

- 📥 Upload EEG feature CSV
- 🧪 Choose between multiple classifiers:
  - Random Forest
  - SVM
  - K-Nearest Neighbors
  - Logistic Regression
- 🔁 5-Fold Cross-Validation
- 📊 Label Distribution Pie Chart
- ✅ Model Evaluation (Accuracy + Classification Report)
- 🔢 Confusion Matrix Heatmap
- 🔬 Feature Importance for Tree Models
- 🔎 Predict a Single Row
- 🎛️ Real-Time Input for Manual Predictions
- 💾 Download trained model (.pkl) and predictions (.csv)

## 📁 Folder Structure

```bash
NeuroFeel/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # Project info
```

## ⚙️ Installation

### Locally
```bash
git clone https://github.com/yourusername/NeuroFeel.git
cd NeuroFeel
pip install -r requirements.txt
streamlit run app.py
```

### In Google Colab (Minimal version)
```python
!pip install streamlit pandas scikit-learn matplotlib seaborn joblib
```

## 📊 Dataset Format
Ensure your CSV file:
- Has **numerical EEG features**
- Includes a **label column** for emotion names

Example:
```csv
alpha,beta,theta,gamma,label
0.1,0.2,0.3,0.4,happy
...
```

## 📦 Dependencies
```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit
```

## 📌 To-Do
- [ ] Add deep learning model options (LSTM/MLP)
- [ ] Add multi-model comparison feature
- [ ] Add time-series EEG support

---

## 💡 Inspiration
This project is perfect for neuroscience, AI + mental health research, or as a student portfolio project exploring the intersection of brain signals and emotions.

## 🧠 Made with ❤️ by [Your Name]

