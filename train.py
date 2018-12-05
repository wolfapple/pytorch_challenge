import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets, models
from collections import OrderedDict
import numpy as np

data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# Define transforms for the training and validation sets
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
valid_transforms = transforms.Compose([
    transforms.Resize(356),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)

# Use pretrained model
model = models.inception_v3(pretrained=True)
# Freeze parameters
for param in model.parameters():
    param.requires_grad = False
    
# Define a untrained feed-forward network as a classifier
clf = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(2048, 1024)),
    ('selu', nn.ELU()),
    ('dropout1', nn.Dropout(p=0.3)),
    ('fc2', nn.Linear(1024, 500)),
    ('selu', nn.ReLU()),
    ('dropout2', nn.Dropout(p=0.3)),
    ('fc3', nn.Linear(500, 102))
]))
model.fc = clf

# Specify loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    model.cuda()

n_epochs = 30
valid_loss_min = np.Inf

for epoch in range(1, n_epochs+1):
    train_loss = 0
    valid_loss = 0
    accuracy = 0
    # Train model
    model.train()
    for x, y in trainloader:
        if train_on_gpu:
            x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    
    # Validate model
    model.eval()
    for x, y in validloader:
        if train_on_gpu:
            x, y = x.cuda(), y.cuda()
        output = model(x)
        loss = criterion(output, y)
        valid_loss += loss.item() * x.size(0)
        # accuracy
        pred = torch.argmax(output, dim=1)
        correct = pred == y.view(*pred.shape)
        accuracy += torch.mean(correct)
    
    # Caculate loss and accuracy
    train_loss = train_loss / len(trainloader.dataset)
    valid_loss = valid_loss / len(validloader.dataset)
    accuracy = accuracy / len(validloader.dataset) * 100
    
    # Print loss
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:2d}%'.format(epoch, train_loss, valid_loss, accuracy))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        checkpoint = {'state_dict': model.state_dict(), 'class_to_idx': train_datasets.class_to_idx}
        torch.save(checkpoint, 'model.pt')
        valid_loss_min = valid_loss