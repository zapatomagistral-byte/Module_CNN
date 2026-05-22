import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time

COLAB = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = '/content/data' if COLAB else './data'

# Data Augmentation for training
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.3,
        value=0,
        scale=(0.05, 0.2)
    ),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

# Transforms for evaluation
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

train_data = datasets.CIFAR10(
    download=True,
    root=root,
    train=True,
    transform=train_transform
)

test_data = datasets.CIFAR10(
    download=True,
    root=root,
    train=False,
    transform=test_transform
)

in_channels = 3

# ── Configuration ─────────────────────────────────────────────────────────────
conv_blocks = [8, 8, 16, 16, 32, 32, 64, 64, 128, 128]
poolings = [1, 3, 5, 7]
dropouts = [0, 2, 4, 6]
kernel_size = 3
fc_neurons = 128

# Training Parameters
epochs = 30
batch_size = 128
learning_rate = 0.0015

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_in = 3
        self.layers = []
        
        # Dynamically build convolutional block pipeline
        for i, out_channels in enumerate(conv_blocks):
            self.layers.append(nn.Conv2d(self.current_in, out_channels, stride=1, kernel_size=kernel_size, padding=1))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU())
            
            # Add spatial downsampling via MaxPool
            if i in poolings:
                self.layers.append(nn.MaxPool2d(2))
            
            # Add Dropout for regularization
            if i in dropouts:
                self.layers.append(nn.Dropout2d(p=0.2))
                
            self.current_in = out_channels
            
        self.convs = nn.Sequential(*self.layers)
        
        # Global Average Pooling (GAP) collapses spatial dimensions from (H, W) to (1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection to class logits
        self.fc1 = nn.Linear(conv_blocks[-1], 10)

    def forward(self, x):
        x = self.convs(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x 

if __name__ == '__main__':

    USE_DATALOADER = True

    if USE_DATALOADER:
        train_dataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataLoader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        print("Loading datasets into memory... (this might take a few seconds)")
        # Use list comprehension so samples pass through the defined transforms
        train_x = torch.stack([train_data[i][0] for i in range(len(train_data))])
        train_y = torch.tensor([train_data[i][1] for i in range(len(train_data))]) 
        
        # Load test set directly to VRAM because it fits without issues
        test_x = torch.stack([test_data[i][0] for i in range(len(test_data))]).to(device)
        test_y = torch.tensor([test_data[i][1] for i in range(len(test_data))]).to(device)

    model = CNN().to(device)
    loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Training loop
    print_every = 5
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Select data source
        if USE_DATALOADER:
            batches = train_dataLoader
        else:
            perm = torch.randperm(len(train_x))
            train_x = train_x[perm]
            train_y = train_y[perm]
            batches = [(train_x[i:i+batch_size], train_y[i:i+batch_size]) 
                       for i in range(0, len(train_x), batch_size)]

        # ── TRAINING PHASE ────────────────────────────────────────────────────
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            cost = loss(logits, labels)
            cost.backward()
            optimizer.step()
            
            running_loss += cost.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # ── TESTING AND LOGS PHASE ────────────────────────────────────────────
        if epoch % print_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                if USE_DATALOADER:
                    test_correct = 0
                    test_total = 0
                    for images, labels in test_dataLoader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()
                    test_acc = 100 * test_correct / test_total
                else:
                    test_outputs = model(test_x)
                    test_preds = test_outputs.argmax(dim=1)
                    test_acc = (test_preds == test_y).float().mean() * 100
            
            epoch_loss = running_loss / len(batches)
            train_acc = 100 * train_correct / train_total
            elapsed_time = time.time() - start_time
            
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {elapsed_time:.1f}s")