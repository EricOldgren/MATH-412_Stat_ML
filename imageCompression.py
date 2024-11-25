# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kzavKlHc_-iq8oWmsDU92s1NAUkxwqTF
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import sampler
from PIL import Image
import numpy as np
import matplotlib.colors as mcolors

# Hyperparameters
RANDOM_SEED = 456
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_CLASSES = 10

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def get_dataloaders_fashion_mnist(batch_size, num_workers=0, train_transforms=None, test_transforms=None):
  if train_transforms is None:
    train_transforms= transforms.Compose([
    transforms.Resize((256, 256)),
    #transforms.Grayscale(),
    transforms.ToTensor(),])
  if test_transforms is None:
    test_transforms= transforms.Compose([
    transforms.Resize((256, 256)),
    #transforms.Grayscale(),
    transforms.ToTensor(),])

  train_dataset = datasets.Imagenette(root='data', split='train', size='320px', transform=train_transforms, download=True)
  test_dataset = datasets.Imagenette(root='data/test', split='val', size='320px', transform=test_transforms, download=True)

  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  test_loader = DataLoader(dataset=test_dataset, batch_size=5, num_workers=num_workers, shuffle=False)
  return train_loader, test_loader

train_loader, test_loader = get_dataloaders_fashion_mnist(batch_size=BATCH_SIZE, num_workers=2)

class AutoEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                #nn.Flatten(),
                #nn.Linear(4096, 4096),
                nn.Conv2d(256, 64, stride=1, kernel_size=3, padding=1),
        )
    self.decoder = nn.Sequential(
                #nn.Linear(4096, 4096),
                #nn.Unflatten(1, (16, 16, 16)),
                nn.ConvTranspose2d(64, 256, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.Upsample(32),
                nn.ConvTranspose2d(256, 128, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.Upsample(64),
                nn.ConvTranspose2d(128, 64, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.Upsample(128),
                nn.ConvTranspose2d(64, 32, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.Upsample(256),
                nn.ConvTranspose2d(32, 3, stride=1, kernel_size=3, padding=1),
                nn.Sigmoid(),
                )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

  def encode(self, x):
    return self.encoder(x)

  def decode(self, x):
    return self.decode(x)

model = AutoEncoder().to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr = LEARNING_RATE,
                             weight_decay = 1e-8)

def train_autoencoder(num_epochs, model, optimizer, train_loader, save_model=None):
  losses = []
  test_losses = []
  outputs = []
  start_time = time.time()
  for epoch in range(num_epochs):
    model.train()
    running_losses = []
    for batch_idx, (features, _) in enumerate(train_loader):
      features = features.to(device)
      output = model(features)
      loss = loss_function(output, features)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_losses.append(loss.item())
    losses.append(np.mean(running_losses))

    model.eval()
    with torch.no_grad():
      running_test_loss = []
      for i, (test_feature, _) in enumerate(test_loader):
        test_feature = test_feature.to(device)
        test_output = model(test_feature)
        test_loss = loss_function(test_output, test_feature)
        running_test_loss.append(test_loss.item())
      test_losses.append(np.mean(running_test_loss))

    print('Epoch: %03d/%03d | Loss: %.4f | Test Loss: %.4f' % (epoch+1, num_epochs, np.mean(running_losses), np.mean(running_test_loss)))
    outputs.append([epoch, features, output])

  print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
  if save_model is not None:
    torch.save(model.state_dict(), save_model)
  return losses, test_losses, outputs

def plot_latent(model, test_loader):
  latents = []
  labels = []
  for batch_idx, (features, label) in enumerate(test_loader):
    model.eval()
    latent = model.encode(features).detach().numpy()
    label = label.detach().numpy()
    latents.extend(latent)
    labels.extend(label)

  colors = ['green', 'red', 'blue', 'grey', 'yellow', 'orange', 'cyan', 'magenta', 'black', 'purple']
  classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  for idx, point in enumerate(latents):
    plt.scatter(point[0], point[1], marker='.', s=20, color=colors[labels[idx]])
  plt.show()

losses, test_losses, outputs = train_autoencoder(num_epochs=NUM_EPOCHS, model=model, optimizer=optimizer, train_loader=train_loader, save_model="model1.pt")

#plot_latent(model=model, test_loader=test_loader)

plt.title('Loss curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(losses, label='Train loss')
plt.plot(test_losses, label='Test loss')
plt.legend()
plt.show()

f, ax = plt.subplots(2, 5)
feature, label = next(iter(test_loader))
model.eval()
feature = feature.to(device)
#compressed = model.encode(feature)
output = model(feature)
for i in range(5):
  original = Image.fromarray(np.transpose(feature[i].cpu().numpy(), (1,2,0))*255, mode='RGB')
  #comp = Image.fromarray(compressed[i].cpu().detach().numpy().reshape(64, 64), mode='L')
  out = Image.fromarray(np.transpose(output[i].cpu().detach().numpy(),(1,2,0))*255, mode='RGB')

  original.save(f'original{i}.png')
  #comp.save(f'compressed{i}.png')
  out.save(f'output{i}.png')

  ax[0, i].imshow(np.transpose(feature[i].cpu(), (1,2,0)))
  #ax[1, i].imshow(compressed[i].cpu().detach().numpy().reshape(64, 64))
  ax[1, i].imshow(np.transpose(output[i].cpu().detach().numpy(), (1,2,0)))
plt.show()