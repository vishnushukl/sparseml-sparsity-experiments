import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.optim import ScheduledOptimizer
from sparseml.pytorch.utils import tensor_sparsity, get_prunable_layers
import matplotlib.pyplot as plt
import os

# Step 1: Load CIFAR-100 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Step 2: Define the Model Architecture
class SimpleSparseNet(nn.Module):
    def __init__(self):
        super(SimpleSparseNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 64)  # Input layer to Hidden Layer 1
        self.fc2 = nn.Linear(64, 32)          # Hidden Layer 1 to Hidden Layer 2
        self.fc3 = nn.Linear(32, 100)          # Hidden Layer 2 to Output Layer (100 classes for CIFAR-100)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
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
print(f'GPU found {device}')

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
        if i % 100 == 99:    # Print every 100 mini-batches
            # print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            train_losses.append(running_loss / 100)
            running_loss = 0.0

# Plot training loss and save the figure
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Batch Index (in hundreds)')
plt.ylabel('Loss')
plt.title('Training Loss Over Time for Base Model')
plt.legend()
plt.grid()
plt.savefig('./cifar-100_vizualizations/base_model_training_loss.png')
plt.close()

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

# Step 5: Test Various Sparsity Levels
sparsity_targets = [0.60, 0.70, 0.75, 0.80]
for sparsity_target in sparsity_targets:
    print(f"\nTesting Sparsity Level: {sparsity_target * 100:.0f}%")

    # Reinitialize the Model for Sparsification
    sparse_model = SimpleSparseNet()
    sparse_model.load_state_dict(model.state_dict())  # Copy weights from the trained base model
    sparse_model = sparse_model.to(device)

    # Step 6: Apply Sparsification and Train the Sparse Model
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

    sparsity_levels = []
    accuracies = []
    sparse_train_losses = []

    print("\nTraining Sparse Model:")
    for epoch in range(num_epochs):
        sparse_model.train()
        running_loss = 0.0
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

            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                # print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                sparse_train_losses.append(running_loss / 100)
                running_loss = 0.0

        # Step 7: Evaluate Sparsity and Accuracy after Each Epoch
        sparse_model.eval()
        prunable_layers = get_prunable_layers(sparse_model)
        sparsity = 0.0
        for (name, layer) in prunable_layers:
            layer_sparsity = tensor_sparsity(layer.weight).item()
            sparsity += layer_sparsity
            print(f"{name}.weight: Sparsity = {layer_sparsity:.4f}")
        sparsity = (sparsity / len(prunable_layers)) * 100 if len(prunable_layers) > 0 else 0

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                images = images.float()  # Ensure images are of type float
                outputs = sparse_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        sparsity_levels.append(sparsity)
        accuracies.append(accuracy)
        print(f'Epoch {epoch + 1} - Sparsity: {sparsity:.2f}% - Accuracy: {accuracy:.2f}%')

        # Print sparsity details for debugging
        print(f'Sparsity check after Epoch {epoch + 1}: {sparsity:.2f}% sparsity')

    # Save training loss plot for sparse model
    plt.figure(figsize=(10, 5))
    plt.plot(sparse_train_losses, label=f'Training Loss for Sparsity {sparsity_target * 100:.0f}%')
    plt.xlabel('Batch Index (in hundreds)')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Time for Sparsity Level {sparsity_target * 100:.0f}%')
    plt.legend()
    plt.grid()
    plt.savefig(f'./cifar-100_vizualizations/sparse_model_training_loss_{sparsity_target * 100:.0f}.png')
    plt.close()

    manager.finalize(sparse_model)

    # Save sparsity and accuracy plot over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), sparsity_levels, label='Sparsity Level (%)')
    plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'Sparsity and Accuracy Over Epochs for Sparsity Target {sparsity_target * 100:.0f}%')
    plt.legend()
    plt.grid()
    plt.savefig(f'./cifar-100_vizualizations/sparsity_accuracy_epochs_{sparsity_target * 100:.0f}.png')
    plt.close()

    # Print Final Sparsity and Accuracy Report
    print("\nFinal Sparsity and Accuracy Report:")
    print(f'Sparsity Target: {sparsity_target * 100:.0f}%')
    for i in range(num_epochs):
        print(f'Epoch {i + 1} - Sparsity: {sparsity_levels[i]:.2f}% - Accuracy: {accuracies[i]:.2f}%')