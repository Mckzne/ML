# inference.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model import CNN

# Load MNIST test dataset
test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True
)

# Create a DataLoader for the test dataset
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

# Set device for model (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# Load the trained model (make sure to adjust the path if necessary)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

def make_inference(index):
    """Make a prediction on a single test image."""
    data, target = test_data[index]
    
    data = data.unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    output = model(data)
    
    prediction = output.argmax(dim=1, keepdim=True).item()
    print(f'Prediction: {prediction}')
    
    image = data.squeeze(0).squeeze(0).cpu().numpy()
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted: {prediction}')
    plt.axis('off')
    plt.show()

# Example usage: make inference on the 3rd test image
make_inference(3)

def compute_precision_recall_f1(model, loaders, device, num_classes=10):
    """Compute precision, recall, and F1-score for each class."""
    model.eval()
    true_positives = torch.zeros(num_classes)
    false_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = output.argmax(dim=1)

            for i in range(len(target)):
                pred_label = predictions[i].item()
                true_label = target[i].item()

                if pred_label == true_label:
                    true_positives[true_label] += 1
                else:
                    false_positives[pred_label] += 1
                    false_negatives[true_label] += 1

    precision = true_positives / (true_positives + false_positives + 1e-8)  # Avoid division by zero
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    for i in range(num_classes):
        print(f'Class {i}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-Score: {f1_score[i]:.4f}')

# Compute precision, recall, and F1-score on the test set
compute_precision_recall_f1(model, loaders={'test': test_loader}, device=device)

def compute_confusion_matrix(model, loaders, device, num_classes=10):
    """Compute and display the confusion matrix."""
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = output.argmax(dim=1)

            for i in range(len(target)):
                true_label = target[i].item()
                pred_label = predictions[i].item()
                confusion_matrix[true_label, pred_label] += 1

    print("Confusion Matrix:\n", confusion_matrix.int())
    
  
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix.numpy(), cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.title("Confusion Matrix")
    plt.show()

# Compute and display the confusion matrix on the test set
compute_confusion_matrix(model, loaders={'test': test_loader}, device=device)
