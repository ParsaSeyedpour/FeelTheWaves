import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🧠 EEG-based Emotion Recognition")
st.markdown("Upload the EEG features CSV (like `emotions.csv` from Kaggle)")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.write(df.head())

    st.subheader("🔍 Dataset Info")
    st.write(f"Shape: {df.shape}")

    st.subheader("📊 Label Distribution")
    label_counts = df['label'].value_counts()
    fig_label, ax_label = plt.subplots()
    ax_label.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    ax_label.set_title('Emotion Label Distribution')
    st.pyplot(fig_label)

    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    # Split features and target
    X = df.drop(columns=['label', 'label_encoded'])
    y = df['label_encoded']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model selection
    st.subheader("⚙️ Choose Model")
    model_name = st.selectbox("Select a classifier:", ["Random Forest", "SVM", "KNN", "Logistic Regression"])

    if model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "SVM":
        clf = SVC(kernel='rbf', probability=True, random_state=42)
    elif model_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)

    # Cross-validation
    st.subheader("🔁 Cross-Validation (5-fold)")
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
    st.write(f"Mean Accuracy: {cv_scores.mean():.2f}")
    st.write(f"Standard Deviation: {cv_scores.std():.2f}")

    # Train classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.subheader("✅ Model Evaluation")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    st.subheader("🔢 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # Feature importance visualization (for tree-based models only)
    if model_name == "Random Forest":
        st.subheader("🔬 Top 10 Feature Importances")
        importances = clf.feature_importances_
        indices = np.argsort(importances)[-10:]
        top_features = X.columns[indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots()
        ax.barh(top_features, top_importances, color='skyblue')
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Important Features')
        st.pyplot(fig)

    # Single row prediction
    st.subheader("🔎 Predict a Single Row")
    single_row = st.selectbox("Choose a row number to predict:", list(range(len(df))))
    input_features = X.iloc[[single_row]]
    input_scaled = scaler.transform(input_features)
    prediction = clf.predict(input_scaled)[0]
    pred_label = le.inverse_transform([prediction])[0]
    st.write(f"**Prediction for row {single_row}:** {pred_label}")

    # Manual input prediction
    st.subheader("🧪 Real-Time Prediction from Manual Input")
    st.markdown("Fill in the feature values below to simulate real-time input:")
    manual_input = []
    for col in X.columns:
        val = st.number_input(f"{col}", value=float(X[col].mean()))
        manual_input.append(val)

    manual_array = np.array(manual_input).reshape(1, -1)
    manual_scaled = scaler.transform(manual_array)
    manual_pred = clf.predict(manual_scaled)[0]
    manual_label = le.inverse_transform([manual_pred])[0]
    st.write(f"**Predicted Emotion:** {manual_label}")

    # Save model
    st.subheader("💾 Download Model and Predictions")
    model_buffer = BytesIO()
    joblib.dump(clf, model_buffer)
    model_buffer.seek(0)
    st.download_button("Download Trained Model (.pkl)", data=model_buffer, file_name="trained_model.pkl")

    # Save predictions
    prediction_df = pd.DataFrame({
        'Actual': le.inverse_transform(y_test),
        'Predicted': le.inverse_transform(y_pred)
    })
    prediction_csv = prediction_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions (.csv)", data=prediction_csv, file_name="predictions.csv", mime="text/csv")
else:
    st.info("Please upload a CSV file to begin.")



































# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score

# st.title("🧠 EEG-based Emotion Recognition")
# st.markdown("Upload the EEG features CSV (like `emotions.csv` from Kaggle)")

# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.subheader("📊 Data Preview")
#     st.write(df.head())

#     st.subheader("🔍 Dataset Info")
#     st.write(f"Shape: {df.shape}")
#     st.write("Label distribution:")
#     st.write(df['label'].value_counts())

#     # Encode labels
#     le = LabelEncoder()
#     df['label_encoded'] = le.fit_transform(df['label'])

#     # Split features and target
#     X = df.drop(columns=['label', 'label_encoded'])
#     y = df['label_encoded']

#     # Feature scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42, stratify=y
#     )

#     # Train classifier
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)

#     st.subheader("✅ Model Evaluation")
#     st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
#     st.text("Classification Report:")
#     st.text(classification_report(y_test, y_pred, target_names=le.classes_))
# else:
#     st.info("Please upload a CSV file to begin.")
