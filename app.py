import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Function to upload dataset
def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    try:
        # Load dataset
        data = pd.read_csv(file_path)
        messagebox.showinfo("File Upload", f"File '{file_path}' uploaded successfully!")

        # Preprocessing
        data = data.dropna()

        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']

        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Build the neural network
        model = Sequential()
        model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        messagebox.showinfo("Model Evaluation", f"Test Accuracy: {accuracy:.4f}")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_classes = (y_pred > 0.5).astype(int)

        # Display classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred_classes))

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        print('Confusion Matrix:')
        print(conf_matrix)

        # ROC AUC Score
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f'ROC AUC Score: {roc_auc:.4f}')

        # Plotting Training History
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Model Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Model Loss')

        plt.show()

        # Function to predict new patient data
        def predict_heart_disease(input_data):
            input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
            prediction = model.predict(input_data_scaled)
            return 'Heart Disease' if prediction > 0.5 else 'No Heart Disease'

        # Example Prediction
        new_patient_data = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
        prediction = predict_heart_disease(new_patient_data)
        print(f'Prediction for new patient data: {prediction}')

        # Pie chart for actual and predicted classes
        actual_counts = y_test.value_counts()
        predicted_counts = pd.Series(y_pred_classes.flatten()).value_counts()

        plt.figure(figsize=(16, 6))

        plt.subplot(1, 3, 1)
        plt.pie(actual_counts, labels=['No Heart Disease', 'Heart Disease'], autopct='%1.1f%%', startangle=140)
        plt.title('Actual Class Distribution')

        plt.subplot(1, 3, 2)
        plt.pie(predicted_counts, labels=['No Heart Disease', 'Heart Disease'], autopct='%1.1f%%', startangle=140)
        plt.title('Predicted Class Distribution')

        # Bar graph for actual vs predicted counts
        plt.subplot(1, 3, 3)
        bar_width = 0.35
        index = np.arange(2)

        plt.bar(index, actual_counts, bar_width, label='Actual')
        plt.bar(index + bar_width, predicted_counts, bar_width, label='Predicted')

        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(index + bar_width / 2, ('No Heart Disease', 'Heart Disease'))
        plt.legend()
        plt.title('Actual vs Predicted Class Counts')

        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process the file: {str(e)}")

# Create GUI window
root = tk.Tk()
root.title("Heart Disease Prediction")
root.geometry("600x400")

# Upload CSV Button
upload_btn = tk.Button(root, text="Upload CSV", command=upload_file)
upload_btn.pack(pady=20)

# Run the application
root.mainloop()