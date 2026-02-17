import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Step 1: Load CIFAR-100 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Step 2: Load the MobileNetV2 Model
# Load MobileNetV2 and adjust the classifier for CIFAR-100
model = mobilenet_v2(weights=None)  # Weights can be set to MobileNet_V2_Weights.IMAGENET1K_V1 for pretrained weights
model.classifier[1] = torch.nn.Linear(model.last_channel, 100)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Step 3: Train the Model
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
num_epochs = 5

print("\nTraining MobileNetV2 on CIFAR-100:")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.float()  # Ensure inputs are of type float
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # Optimize
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

# Step 4: Evaluate the Model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        images = images.float()  # Ensure images are of type float
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'MobileNetV2 Accuracy on Test Data: {accuracy:.2f}%')