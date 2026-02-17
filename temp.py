import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
start_time = time.time()
from sparseml.pytorch.optim import ScheduledModifierManager
end_time = time.time()
import_time = end_time - start_time
print(f"Time taken to import ScheduledModifierManager: {import_time} seconds")
from sparseml.pytorch.optim import ScheduledOptimizer
from sparseml.pytorch.utils import tensor_sparsity, get_prunable_layers
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

newsgroups = fetch_20newsgroups(subset='all', categories=['rec.autos', 'sci.space', 'talk.politics.misc'])
X = newsgroups.data
y = newsgroups.target

def clean_text(text, min_word_frequency=2):
    # Initialize stemmer, lemmatizer, and stopwords
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    tokens = [word for word in tokens if not word.isdigit()]
    return " ".join(tokens)

cleaned_text = [clean_text(i) for i in X]
vectorizer = TfidfVectorizer(max_features=150)
cleaned_text = vectorizer.fit_transform(cleaned_text).toarray() 
scaler = StandardScaler()
cleaned_text = scaler.fit_transform(cleaned_text)
X_train, X_test, y_train, y_test = train_test_split(cleaned_text, y, test_size=0.2, random_state=42)
# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
# Create DataLoader for training and testing
trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device} is being used for computation')

# Step 2: Define the Model Architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(150, 8)
        self.bn1 = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Step 3: Train the Base Model
model = SimpleNN()
model = model.to(device)
    

# Step 3: Train the Base Model
# Optimizer (use Adam instead of the non-existent 'ADA' optimizer)
base_optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
num_epochs = 5

# Step 4: Training Loop
train_losses = []
print("\nTraining Base Model:")
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        # print(inputs.shape, labels.shape)
        base_optimizer.zero_grad()  # Zero gradients before backprop

        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()  # Backpropagate the loss

        # Update the weights
        base_optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # Record loss every 10 mini-batches
            train_losses.append(running_loss / 10)
            running_loss = 0.0

    # Step 5: Evaluate the Base Model
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No gradient tracking during evaluation
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch {epoch + 1} - Accuracy: {accuracy:.2f}%')



# # Step 5: Test Various Sparsity Levels
# Step 5: Test Various Sparsity Levels
sparsity_targets = [0.60, 0.70, 0.75, 0.80]
for sparsity_target in sparsity_targets:
    print(f"\nTesting Sparsity Level: {sparsity_target * 100:.0f}%")

    # Reinitialize the Model for Sparsification
    sparse_model = SimpleNN()
    sparse_model.load_state_dict(model.state_dict())  # Copy weights from the trained base model
    sparse_model = sparse_model.to(device)

    # Step 6: Apply Sparsification and Train the Sparse Model
    # optimizer = torch.optim.SGD(sparse_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(sparse_model.parameters(), lr=0.001, weight_decay=1e-4)
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
        end_epoch: 5.0
        update_frequency: 0.3
        init_sparsity: 0.2
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
            optimizer.zero_grad()

            # Forward pass
            outputs = sparse_model(inputs)
            # labels = labels.float().unsqueeze(1)  # Adjust labels shape for binary classification
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            # Optimize
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:    # Record loss every 10 mini-batches
                sparse_train_losses.append(running_loss / 10)
                running_loss = 0.0

        # Step 7: Evaluate Sparsity and Accuracy after Each Epoch
        sparse_model.eval()
        prunable_layers = get_prunable_layers(sparse_model)
        sparsity = 0.0
        total_weights = 0
        remaining_weights = 0
        for (name, layer) in prunable_layers:
            layer_sparsity = tensor_sparsity(layer.weight).item()
            sparsity += layer_sparsity
            total_weights += layer.weight.numel()
            remaining_weights += (layer.weight != 0).sum().item()
        sparsity = (remaining_weights / total_weights) * 100 if remaining_weights > 0 else 0

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = sparse_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        sparsity_levels.append(sparsity)
        accuracies.append(accuracy)
        print(f'Epoch {epoch + 1} - Sparsity: {sparsity:.2f}% - Accuracy: {accuracy:.2f}% - Remaining Weights: {remaining_weights}/{total_weights}')
        # print(f'Epoch {epoch + 1} - Accuracy: {accuracy:.2f}%')
        # print(f'Remaining Weights: {remaining_weights}/{total_weights}')

    manager.finalize(sparse_model)