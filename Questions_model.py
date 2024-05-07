# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns  

# This workaround might cause the environment to crash, not a good solution
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Environment setup and global variables
dataset_path = 'train_data_3'
test_data_path = 'test_data_3'
# dataset_path = 'ID_train'
# test_data_path = 'ID_test'
torch.manual_seed(0)  # Ensure reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check and set the device
print(f"Using {device} device")

# Data preprocessing
def get_transforms():
    return transforms.Compose([
        # resize images
        transforms.Resize((30, 150)),
        # Convert to grayscale image
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # Reducing differences in training data
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

# Load datasets
def load_datasets(dataset_path, transform):
    CUSTOM_CLASS_TO_IDX = {'A': 0, 'B': 1, 'C': 2, 'D':3, 'E':4, 'None':5}
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    full_dataset.class_to_idx = CUSTOM_CLASS_TO_IDX
    return full_dataset

# Data loaders
def get_dataloaders(train_dataset,batch_size=8):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# Define the neural network model for the problem
class Questions_model(nn.Module):
    def __init__(self):
        super(Questions_model, self).__init__()
        # Define convolutional layers - 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  
        self.bn4 = nn.BatchNorm2d(256)  
        # Reducing the number of parameters and computational complexity of calculations 
        # helps to retain strong features.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define fully connected layers
        # Mapping the features extracted from the convolutional layer to the final output class
        self.fc1 = nn.Linear(256 * 1 * 9, 512)
        self.bn5 = nn.BatchNorm1d(512)  
        self.fc2 = nn.Linear(512, 256)  
        self.bn6 = nn.BatchNorm1d(256)  
        self.fc3 = nn.Linear(256, 6)  
        # Used to reduce model overfitting， remove 50% features randomly
        self.dropout = nn.Dropout(0.5)
    
    # Forward propagation of trained parameters
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  
        x = x.view(-1, 256 * 1 * 9)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.fc2(x))) 
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Training and validation function
def train_model(model, num_epochs, train_loader, criterion, optimizer, scheduler, device, model_path):
    # Initialize lists to store training loss and accuracy for each epoch
    train_losses = []
    train_accuracies = []

    # Begin the training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Accumulate total loss for the current epoch
        correct = 0  # Accumulate number of correct predictions for the current epoch
        total = 0  # Accumulate total number of processed samples for the current epoch

        # Iterate over the training data for the current epoch
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass: compute the model output
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backward pass: compute gradient
            optimizer.step()  # Optimization step: update weights

            running_loss += loss.item() * images.size(0)  # Update total loss
            _, predicted = torch.max(outputs.data, 1)  # Get predictions
            total += labels.size(0)  # Update total number of samples
            correct += (predicted == labels).sum().item()  # Update number of correct predictions

        # Calculate the average loss and accuracy for the current epoch
        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Record the loss and accuracy for the current epoch
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')
        scheduler.step()  # Update the learning rate

    # Save model parameters to the specified file
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')


def load_test_data(test_data_path, transform):
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    return test_loader

def test_model(test_loader, model, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    misclassified = []  # List for recording information about misclassifications
    all_labels = []  # Collect all true labels
    all_preds = []  # Collect all predicted labels

    with torch.no_grad():  # No gradient calculation to speed up inference
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # Check which images were classified incorrectly
            mismatches = (predicted != labels)
            mis_idxs = np.where(mismatches.cpu().numpy())[0]
            for idx in mis_idxs:
                img_path = test_loader.dataset.samples[batch_idx * test_loader.batch_size + idx][0]
                img_name = os.path.basename(img_path)
                class_label = test_loader.dataset.classes[labels[idx].item()]
                predicted_label = test_loader.dataset.classes[predicted[idx].item()]
                misclassified.append((img_name, class_label, predicted_label))

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')
    # Print out information about misclassified images if there are any
    if misclassified:
        print("Misclassified images: [name: (true label, predicted label)]")
        for item in misclassified:
            print(f"Name {item[0]}: (True: {item[1]}, Pred: {item[2]})")

    # Return the test accuracy and lists of all true labels and predictions
    return test_accuracy, misclassified, all_labels, all_preds


def visualize_performance(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:\n", cr)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()


# Main execution logic
def main_train(model_filename):
    transforms = get_transforms() 
    train_dataset = load_datasets(dataset_path, transforms)  
    train_loader = get_dataloaders(train_dataset)  
    model = Questions_model().to(device)  
    criterion = nn.CrossEntropyLoss()  
    # Define the optimizer, here the Adam optimizer is used
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  
    # The learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  
    num_epochs = 45  # Number of training epochs
    
    # Train the model
    train_model(model, num_epochs, train_loader, criterion, optimizer, scheduler, device, model_path=model_filename)

    # Test the model
    test_loader = load_test_data(test_data_path, transforms)  
    model.load_state_dict(torch.load(model_filename))  
    model.to(device)  
    test_accuracy, misclassified, y_true, y_pred = test_model(test_loader, model, device)  

    # Calculate and visualize performance metrics
    class_names = test_loader.dataset.classes  
    visualize_performance(y_true, y_pred, class_names)  




if __name__ == "__main__":
    # model_filename = '4CN_bz8_lr0.0005_ep45_3'  
    model_filename = 'Final_model'  
    # model_filename = 'ID_lr0.0005_ep10'
    main_train(model_filename)
    # main_notrain(model_filename)




def main_notrain(model_filename):
    # Setting up transformations, loading test sets, etc
    transforms = get_transforms()
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize the model and load pre-trained weights
    model = Questions_model().to(device)
    model.load_state_dict(torch.load(model_filename))  # Make sure to replace with your model file path
    model.to(device)

    # test model
    test_model(test_loader, model, device)


# draw the picture
def plot_metrics(train_loss, train_acc, save_path):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'r-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path)
    plt.show()