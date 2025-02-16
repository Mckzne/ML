# train.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import CNN
import torch.nn as nn
import numpy as np
from IPython.display import display, clear_output  # Import display functions

# Load MNIST dataset
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='data', train=False, transform=transform, download=True)

# Split training data into training and validation sets
train_size = int(0.9 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

# Data loaders
loaders = {
    'train': DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0),
    'val': DataLoader(val_dataset, batch_size=100, shuffle=True, num_workers=0),
    'test': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=0)
}

# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
train_accs = []

best_val_loss = float('inf')
patience = 5  # Number of epochs to wait for improvement
counter = 0

def train(epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\tLoss: {loss.item():.6f}')

    train_accuracy = 100 * correct / total
    train_accs.append(train_accuracy)
    print(f'Epoch {epoch}: Training Accuracy: {train_accuracy:.2f}%')

def validate():
    model.eval()
    val_loss = 0
    correct = 0
    incorrect_samples = []
    with torch.no_grad():
        for data, target in loaders['val']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_tensor = pred.eq(target.view_as(pred))
            correct += correct_tensor.sum().item()

            # Find incorrect predictions
            incorrect_indices = (~correct_tensor).nonzero().flatten()
            for i in incorrect_indices:
                index = i.item()
                img = data[index].cpu().numpy()
                true_label = target[index].item()
                predicted_label = pred[index].item()
                incorrect_samples.append((img, true_label, predicted_label))

    val_loss /= len(loaders['val'].dataset)
    val_losses.append(val_loss)
    accuracy = 100. * correct / len(loaders['val'].dataset)
    print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(loaders["val"].dataset)} ({accuracy:.6f}%)')

    # Display some incorrect samples using IPython display functions
    print("\nSome Incorrect Predictions:")
    num_to_display = min(10, len(incorrect_samples))  # Display up to 10 images

    fig, axes = plt.subplots(1, num_to_display, figsize=(15, 3))  # Adjust figsize as needed
    fig.tight_layout()

    for i in range(num_to_display):
        img, true_label, predicted_label = incorrect_samples[i]
        ax = axes[i]
        ax.imshow(img.squeeze(), cmap='gray')  # img is (1, 28, 28), so squeeze to (28, 28)
        ax.set_title(f"T: {true_label}, P: {predicted_label}")
        ax.axis('off')  # Hide the axes

    plt.show()  # Show the plot of incorrect predictions

    return val_loss


def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs Iterations')
    plt.legend()
    plt.show()

# Training loop
for epoch in range(1, 21):
    train(epoch)
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth') # Save the best model
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break

torch.save(model.state_dict(), 'model.pth')
plot_losses(train_losses, val_losses)
