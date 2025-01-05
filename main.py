import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay,
)
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Path to the dataset
dataset_path = "patern_dataset"

# Load dataset


def load_dataset(dataset_path):
    images = []
    labels = []
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if image_name.lower().endswith('.jpg'):
                    # Read image in grayscale
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        # Resize image to 70x70
                        image = cv2.resize(image, (70, 70))
                        # Flatten the image to a 1D array
                        images.append(image.flatten())
                        # Use folder name as label (assuming folder names are numbers)
                        labels.append(int(folder_name))
    return np.array(images), np.array(labels)


# Load the dataset
images, labels = load_dataset(dataset_path)

# Binarize labels for ROC curve (one-vs-rest)
labels_binarized = label_binarize(labels, classes=np.unique(labels))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    images, labels, labels_binarized, test_size=0.2, random_state=42
)

# Train and save SVM model


def train_svm(X_train, y_train):
    svm_model = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    svm_model.fit(X_train, y_train)
    joblib.dump(svm_model, "svm_model.pkl")
    return svm_model

# Train and save KNN model


def train_knn(X_train, y_train):
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    joblib.dump(knn_model, "knn_model.pkl")
    return knn_model

# Train and save Decision Tree model


def train_decision_tree(X_train, y_train):
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    joblib.dump(dt_model, "decision_tree_model.pkl")
    return dt_model


# Train models
svm_model = train_svm(X_train, y_train)
knn_model = train_knn(X_train, y_train)
dt_model = train_decision_tree(X_train, y_train)

# Evaluate models


def evaluate_model(model, X_test, y_test, y_test_bin, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(
        model, "predict_proba") else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print metrics
    print(f"--- {model_name} Metrics ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot ROC curve (only for models with predict_proba)
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
        roc_auc = auc(fpr, tpr)
        plt.figure()
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
        plt.title(f'{model_name} ROC Curve')
        plt.show()


# Evaluate SVM
evaluate_model(svm_model, X_test, y_test, y_test_bin, "SVM")

# Evaluate KNN
evaluate_model(knn_model, X_test, y_test, y_test_bin, "KNN")

# Evaluate Decision Tree
evaluate_model(dt_model, X_test, y_test, y_test_bin, "Decision Tree")
