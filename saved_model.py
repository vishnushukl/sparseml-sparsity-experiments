import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.optim import ScheduledOptimizer
from sparseml.pytorch.utils import tensor_sparsity, get_prunable_layers
import numpy as np
import matplotlib.pyplot as plt
import onnx

# Step 1: Load MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Step 2: Define the Model Architecture (2 Hidden Layers)
class SimpleSparseNet(nn.Module):
    def __init__(self):
        super(SimpleSparseNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 6)  # Input layer to Hidden Layer 1
        self.fc2 = nn.Linear(6, 10)       # Hidden Layer 1 to Hidden Layer 2
        self.fc3 = nn.Linear(10, 10)      # Hidden Layer 2 to Output Layer (10 classes for MNIST)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Step 3: Train the Base Model
model = SimpleSparseNet()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
num_epochs = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device} is using to train.')

train_losses = []
print("\nTraining Base Model:")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.float()  # Ensure inputs are of type float
        base_optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # Optimize
        base_optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # Record loss every 100 mini-batches
            train_losses.append(running_loss / 100)
            running_loss = 0.0

# Step 4: Evaluate the Base Model
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
base_accuracy = 100 * correct / total
print(f'Base Model Accuracy on Test Data: {base_accuracy:.2f}%')

# Step 5: Sparsify and Save Model in ONNX Format (80% Sparsity)
sparsity_target = 0.80
print(f"\nTesting Sparsity Level: {sparsity_target * 100:.0f}%")

# Reinitialize the Model for Sparsification
sparse_model = SimpleSparseNet()
sparse_model.load_state_dict(model.state_dict())  # Copy weights from the trained base model
sparse_model = sparse_model.to(device)

# Step 6: Apply Sparsification
optimizer = torch.optim.SGD(sparse_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
steps_per_epoch = len(trainloader)

# Create a new sparsification recipe with the target sparsity level
recipe_content = f'''
modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: 5.0

  - !GlobalMagnitudePruningModifier
    params: __ALL_PRUNABLE__
    start_epoch: 1.0
    end_epoch: 4.0
    update_frequency: 0.1
    init_sparsity: 0.05
    final_sparsity: {sparsity_target}
    mask_type: unstructured
'''

with open('temp_recipe.yaml', 'w') as f:
    f.write(recipe_content)

manager = ScheduledModifierManager.from_yaml('temp_recipe.yaml')
optimizer = ScheduledOptimizer(optimizer, sparse_model, manager, steps_per_epoch=steps_per_epoch)

print("\nTraining Sparse Model:")
for epoch in range(num_epochs):
    sparse_model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.float()  # Ensure inputs are of type float
        optimizer.zero_grad()

        # Forward pass
        outputs = sparse_model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # Optimize
        optimizer.step()

manager.finalize(sparse_model)

# Save the sparse model in ONNX format
onnx_path = "./saved_model/sparse_model.onnx"
dummy_input = torch.randn(1, 1, 28, 28, device=device)
torch.onnx.export(sparse_model, dummy_input, onnx_path, verbose=True, input_names=['input'], output_names=['output'])
print(f"Sparse model saved in ONNX format at: {onnx_path}")

# Step 7: Predict a Random Image with Ground Truth and Actual
sparse_model.eval()
with torch.no_grad():
    random_index = np.random.randint(len(testset))
    image, label = testset[random_index]
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    output = sparse_model(image)
    _, predicted_label = torch.max(output, 1)

    # Plot the image with ground truth and prediction
    plt.figure()
    plt.imshow(image.cpu().squeeze(), cmap='gray')
    plt.title(f'Ground Truth: {label}, Predicted: {predicted_label.item()}')
    plt.savefig('random_prediction.png')
    print(f"Random prediction saved as 'random_prediction.png' with Ground Truth: {label}, Predicted: {predicted_label.item()}")