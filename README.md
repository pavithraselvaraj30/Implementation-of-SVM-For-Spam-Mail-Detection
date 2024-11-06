# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Load and Preprocess Data: Load the dataset, select relevant columns, and map labels to numeric values.

Split Data: Divide the dataset into training and test sets.

Vectorize Text: Convert text messages into a numerical format using TfidfVectorizer.

Train SVM Model: Fit an SVM model with a linear kernel on the training data.

Evaluate Model: Predict on the test set and print the accuracy score and classification report.

Visualize Results: Plot the confusion matrix, ROC curve, and precision-recall curve for detailed evaluation.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MOPURI ANKITHA
RegisterNumber: 212223040117 
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]  # Select relevant columns
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Map labels to 0 (ham) and 1 (spam)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)
y_proba = model.decision_function(X_test_vec)  # For ROC and Precision-Recall curves

# 1. Accuracy Score
print("Accuracy:", accuracy_score(y_test, y_pred))

# 2. Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 4. ROC Curve and AUC Score
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 5. Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/44cf86ee-1c2b-422d-9af2-9164d21a8ada)
![image](https://github.com/user-attachments/assets/7828d970-2fcf-42c2-a83b-098d35cfced2)
![image](https://github.com/user-attachments/assets/d5bf464f-0f4f-4563-bb11-1ba580bf278c)
![image](https://github.com/user-attachments/assets/0d2980fc-24df-4857-a999-fba03b4b0790)






## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
