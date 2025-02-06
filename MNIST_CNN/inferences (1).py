# inferences.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN

# Load test dataset
test_data = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Function to calculate the confusion matrix
def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# Function to calculate precision, recall, and F1-score for each class
def compute_metrics(cm):
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    return precision, recall, f1_score

# Store true and predicted labels
y_true, y_pred = [], []
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)

        y_true.extend(target.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

        correct_predictions += (pred == target).sum().item()
        total_samples += target.size(0)

# Calculate overall accuracy
accuracy = correct_predictions / total_samples
print(f"Test Accuracy: {accuracy:.6f}")

# Compute confusion matrix and metrics
cm = compute_confusion_matrix(y_true, y_pred)
precision, recall, f1 = compute_metrics(cm)

# Print class-wise precision, recall, and F1-score
for i in range(10):
    print(f"Class {i}: Precision={precision[i]:.6f}, Recall={recall[i]:.6f}, F1-score={f1[i]:.6f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
