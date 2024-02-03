import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# EMNIST Constants
input_size = 784  # 28x28
output_classes = 62 # output size EMNIST byclass

total_epochs = 50
batch_size = 100
learning_rate = 0.1


# EMNIST train/test dataset and loader
train_dataset = torchvision.datasets.EMNIST(root='./data', split='byclass', train=True,transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.EMNIST(root='./data', split='byclass', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# CNN Model inspired by Lenet-5
class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        channels = 1
        self.input_size = input_size

        # Lenet 5 Modified
        self.CNN = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 6, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(16 * 4 * 4, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, num_classes),
        )

    def forward(self, x):
        out = self.CNN(x)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


# Setup model
model = Model(output_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Keeps track of training and testing accuracy
training_acc = []
testing_acc = []
n_total_steps = len(train_loader)

# Training and Testing Loop
for epoch in range(total_epochs):

    # Start of Training
    train_acc = 0
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy calculation
        prediction = torch.argmax(outputs, 1)
        train_acc += (prediction==labels).sum().item()

        # Prints progress
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{total_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

    train_acc = train_acc/len(train_dataset)
    model.eval()
    # End of Training

    # Start of Testing
    n_correct = 0
    n_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Accuracy Calculations
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    test_acc = n_correct / n_samples

    # Prints test Accuracy
    print(f'Accuracy of the network on the 10000 test images: {test_acc}')
    training_acc.append(train_acc)
    testing_acc.append(test_acc)

    # time.sleep(60)

# Plots Training and Testing Accuracy VS Epochs
plt.title("Train & Test acc VS Epochs")
plt.plot(range(total_epochs), training_acc, label="training acc")
plt.plot(range(total_epochs), testing_acc, label="testing acc")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.show()

