import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

# Data parameters
hr_size = (50, 50)
lr_size = (25, 25)
batch_size = 256
shuffle = True
drop_last = True
download = True
# Training parameters
epochs = 10
learning_rate = 0.0002
# Network parameters
num_layers = 10
# Other parameters
log_period = 50

# Training setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

# Set the transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(hr_size)])

# Set the training data
data_train = datasets.CIFAR10('~/.pytorch/CIFAR_data/', download=download, train=True, transform=transform)
loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

# Set the test data
data_test = datasets.CIFAR10('~/.pytorch/CIFAR_data/', download=download, train=False, transform=transform)
loader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

class REDNet(nn.Module):
    def __init__(self, num_layers=10):
        super(REDNet, self).__init__()
        self.num_layers = num_layers

        # Encoders
        self.input_layer = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.encoders = [nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='same') for _ in range(self.num_layers - 1)]
    
        # Decoders
        self.decoders = [nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='same') for _ in range(self.num_layers - 1)]
        self.output_layer = nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding='same')

        # Residual
        self.residual = []

    def forward(self, x):
        # If the number of layers is even
        if self.num_layers % 2 == 0:
            # Input layer
            x = F.relu(self.input_layer(x))
            residual = x

            # Encoder
            for i in range(self.num_layers - 1):
                x = F.relu(self.encoders[i](x))
                if i % 2 == 1:
                    self.residual.append(x)

            # Decoder
            for i in range(self.num_layers - 1):
                if i % 2 == 0:
                    x = F.relu(self.decoders[i](x))
                else:
                    x = F.relu(self.decoders[i](x + self.residual.pop()))

            # Output layer
            x = F.relu(self.output_layer(x + residual))

        # If the number of layers is odd
        if self.num_layers % 2 == 1:
            # Input layer
            x = F.relu(self.input_layer(x))

            # Encoder
            for i in range(self.num_layers - 1):
                x = F.relu(self.encoders[i](x))
                if i % 2 == 0:
                    self.residual.append(x)

            # Decoder
            for i in range(self.num_layers - 1):
                if i % 2 == 0:
                    x = F.relu(self.decoders[i](x))
                else:
                    x = F.relu(self.decoders[i](x + self.residual.pop()))

            # Output layer
            x = F.relu(self.output_layer(x))

        return x
    
def split_lr_hr(img):
    hr = img
    trans1 = transforms.Compose([transforms.Resize(lr_size)])
    trans2 = transforms.Compose([transforms.Resize(hr_size)])
    lr = trans2(trans1(img))
    return lr, hr

model = REDNet(num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
print(model, device)

model.train()
plt.figure(figsize=(3, 4))
for epoch in range(epochs):
    for step, (img, label) in enumerate(loader_train):
        lr, hr = split_lr_hr(img)
        input = lr.to(device)
        output = model(input)

        target = hr.to(device)

        loss = loss_fn(output, target.detach())  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % log_period == 0:
            print("Train Epoch: {}\tStep: {}\tLoss: {:.6f}".format(epoch, step, loss.item()))

    plt.subplot(1, 3, 1)
    plt.imshow(hr[0].squeeze(0).permute(1, 2, 0))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(lr[0].squeeze(0).permute(1, 2, 0).to('cpu').detach().numpy())
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(output[0].squeeze(0).permute(1, 2, 0).to('cpu').detach().numpy())
    plt.axis('off')

    plt.show()