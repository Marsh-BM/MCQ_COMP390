
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


dataset_path = 'ID_train'
test_data_path = 'ID_test'
torch.manual_seed(0)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(f"Using {device} device")

# Define the transformation for the image data
def get_transforms():
    return transforms.Compose([
        transforms.Resize((184, 30)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

# Load the dataset
def load_datasets(dataset_path, transform):
    CUSTOM_CLASS_TO_IDX = {'0_file':0,'1_file': 1, '2_file': 2, '3_file': 3, '4_file':4, '5_file':5, '6_file':6, '7_file':7, '8_file':8, '9_file':9}
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    full_dataset.class_to_idx = CUSTOM_CLASS_TO_IDX
    return full_dataset

# Split the dataset into training and validation sets
def get_dataloaders(train_dataset,batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# Define the CNN model
class ID_model(nn.Module):
    def __init__(self):
        super(ID_model, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.fc1 = nn.Linear(128 * 23 * 3, 512)  
        self.fc2 = nn.Linear(512, 10)  
        self.dropout = nn.Dropout(0.5) 

# Define the forward pass
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  
        x = x.view(-1, 128 * 23 * 3)  
        x = F.relu(self.fc1(x))  
        x = self.dropout(x) 
        x = self.fc2(x)  
        return x

# Train the model
def train_model(model, num_epochs, train_loader, criterion, optimizer, device, model_path):

    train_losses = []
    train_accuracies = []

    # Train the model
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0  
        correct = 0  
        total = 0  
        # Iterate over the train_loader
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()  
            outputs = model(images) 
            loss = criterion(outputs, labels) 
            loss.backward()  
            optimizer.step()  
            # Calculate the accuracy
            running_loss += loss.item() * images.size(0) 
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)  
            correct += (predicted == labels).sum().item() 

        # Calculate the epoch loss and accuracy
        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Append the loss and accuracy
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')

    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    plot_metrics(train_losses, train_accuracies, 'ID_Model_Picture.png')

# Load the test data
def load_test_data(test_data_path, transform):
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_loader

# Test the model
def test_model(test_loader, model, device):
    model.eval()  
    correct = 0
    total = 0
    misclassified = [] 

    with torch.no_grad(): 
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Find misclassified images
            mismatches = (predicted != labels)
            mis_idxs = np.where(mismatches.cpu().numpy())[0]  
            for idx in mis_idxs:
                img_path = test_loader.dataset.samples[batch_idx * test_loader.batch_size + idx][0]  # 获取图片路径
                img_name = os.path.basename(img_path)
                class_label = test_loader.dataset.classes[labels[idx].item()]
                predicted_label = test_loader.dataset.classes[predicted[idx].item()]
                misclassified.append((img_name, class_label, predicted_label))
    # Calculate the accuracy
    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')
    #  Print the misclassified images
    if misclassified:
        print("Misclassified images: [name: (true label, predicted label)]")
        for item in misclassified:
            print(f"Name {item[0]}: (True: {item[1]}, Pred: {item[2]})")

    return test_accuracy, misclassified



# Plot the training metrics
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


# Train the model
def main_train(model_filename):
    transforms = get_transforms()
    train_dataset= load_datasets(dataset_path, transforms)
    train_loader= get_dataloaders(train_dataset)
    model = ID_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    num_epochs = 30

    
    train_model(model, num_epochs, train_loader, criterion, optimizer, device, model_path=model_filename)

   
    test_loader = load_test_data(test_data_path, transforms)
    model.load_state_dict(torch.load(model_filename))  
    model.to(device)
    test_model(test_loader, model, device)

def main_notrain(model_filename):

    transforms = get_transforms()
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    model = ID_model().to(device)
    model.load_state_dict(torch.load(model_filename))  
    model.to(device)

    test_model(test_loader, model, device)

if __name__ == "__main__":
    # model_filename = 'lr0.0005_ep10'  
    model_filename = 'ID_lr0.00005_ep30'
    main_train(model_filename)
    # main_notrain(model_filename)

    