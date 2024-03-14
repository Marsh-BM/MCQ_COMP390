import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        # First convolution layer with specified input channels, output channels, kernel size, padding, and stride
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        # Second convolution layer with specified number of channels, kernel size, and padding
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        # Optional 1x1 convolution layer used for matching dimensions in residual connection
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        # Batch normalization layers to normalize the inputs
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # Apply first convolution and batch normalization, followed by ReLU activation
        Y = F.relu(self.bn1(self.conv1(X)))
        # Apply second convolution and batch normalization
        Y = self.bn2(self.conv2(Y))

        # Apply the optional 1x1 convolution to the input if it's defined
        if self.conv3:
            X = self.conv3(X)

        # Add the input (residual) to the output of the convolutional layers (residual connection)
        Y += X
        # Apply ReLU activation and return the output
        return F.relu(Y)

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        # Initialization of the global average pooling layer

    def forward(self, x):
        # Applies global average pooling
        return nn.functional.avg_pool2d(x, kernel_size=x.size()[2:])

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    # Create several residual blocks
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # First block of the sequence, with downsampling
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            # Regular residual block
            blk.append(Residual(num_channels, num_channels))
    return blk  # Return the sequence of blocks


def init_weights(m):
    # Initialize weights for linear and convolutional layers
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def create_resnet():
    # Block 1: Convolutional layer followed by BatchNorm, ReLU, and MaxPooling
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    # Block 2: First ResNet block with 64 channels
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # Unpacking the blocks

    # Block 3: Second ResNet block with 128 channels
    b3 = nn.Sequential(*resnet_block(64, 128, 2))

    # Block 4: Third ResNet block with 256 channels
    b4 = nn.Sequential(*resnet_block(128, 256, 2))

    # Block 5: Fourth ResNet block with 512 channels
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    # Adding Dropout for regularization
    dropout_rate = 0.25  # Dropout rate, typically between 0.2 and 0.5

    # Final network with Global Average Pooling, Dropout, Flatten, and a Linear layer
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        GlobalAvgPool2d(),
                        nn.Dropout(dropout_rate),
                        nn.Flatten(), nn.Linear(512, 10))

    return net


def load_data_fashion_mnist(batch_size, resize=None):
    """Download Fashion-MNIST dataset and load it into memory."""

    # Define transformations: convert to tensor, optionally resize
    transform = [transforms.ToTensor()]
    if resize:
        transform.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transform)

    # Load training and test datasets
    train_dataset = torchvision.datasets.FashionMNIST(root="./data/FashionMNIST", train=True, download=True,
                                                      transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root="./data/FashionMNIST", train=False, download=True,
                                                     transform=transform)

    # DataLoader for batch processing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Extract the first batch from the training set
    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break
    batch_x = b_x.squeeze().numpy()  # Convert tensor to numpy array
    batch_y = b_y.numpy()  # Convert tensor to numpy array
    class_label = train_dataset.classes  # Class labels of the training set
    class_label[0] = "T-shirt"  # Rename first class
    print("the size of batch in train data:", batch_x.shape)
    print("Size of a single image in the batch:", batch_x[0].shape)

    # Extract the first batch from the test set
    for step, (b_i, b_j) in enumerate(test_loader):
        if step > 0:
            break
    batch_i = b_i.squeeze().numpy()  # Convert tensor to numpy array
    batch_j = b_j.numpy()  # Convert tensor to numpy array
    print("The size of batch in test data:", batch_i.shape)

    # Visualize a batch of images
    plt.figure(figsize=(12, 5))
    for ii in np.arange(len(batch_y)):
        plt.subplot(4, 16, ii + 1)
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
        plt.title(class_label[batch_y[ii]], size=9)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()

    return train_loader, test_loader


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):
    # Lists to store training and testing metrics
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    epoch_times = []

    # Loop over the number of epochs
    for epoch in range(num_epochs):
        start_time = time.time()  # Start time of the epoch
        model.train()  # Set the model to training mode
        total_loss = 0  # Total loss for the epoch
        correct = 0    # Total number of correct predictions
        total = 0      # Total number of predictions

        # Loop over the training data
        for images, labels in train_loader:
            # Move images and labels to the specified device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Get model outputs
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Accumulate loss and accuracy
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average training loss and accuracy
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        # Evaluate the model on the test set
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Calculate and print epoch metrics
        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Time: {epoch_duration:.2f} sec')

        # Update the learning rate
        scheduler.step()

    return train_losses, train_accuracies, test_losses, test_accuracies, epoch_times

import torch

def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0  # Total loss for the test data
    correct = 0     # Total number of correct predictions
    total = 0       # Total number of predictions

    # No gradient updates needed during evaluation
    with torch.no_grad():
        # Loop over the test data
        for images, labels in test_loader:
            # Move images and labels to the specified device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Get model outputs
            loss = criterion(outputs, labels)  # Calculate the loss
            total_loss += loss.item()  # Accumulate the loss

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average test loss and accuracy
    avg_test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    return avg_test_loss, test_accuracy



def main():
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Batch size for loading data
    batch_size = 64
    # Load the Fashion MNIST dataset
    train_loader, test_loader = load_data_fashion_mnist(batch_size, resize=96)

    # Create the ResNet model
    net = create_resnet()
    # Apply weight initialization to the network
    net.apply(init_weights)
    # Move the network onto the device
    net.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (uncomment the desired optimizer)
    # optimizer = optim.Adam(net.parameters(), lr=0.0075)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Number of epochs to train for
    num_epochs = 25

    # Train the model
    train_losses, train_accuracies, test_losses, test_accuracies, epoch_times = train_model(
        net, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device
    )

    # Plot training and test losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and test accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()

# if __name__ == '__main__':
#     main()
