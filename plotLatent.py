import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import sampler
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import numpy as np
import matplotlib.colors as mcolors
from PIL import Image
import io

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_EPOCHS = 100
EARLY_STOPPING_EPOCHS = 20
JPEG_COMPRESSION = 64

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def get_dataloaders(batch_size, num_workers=0, train_transforms=None, test_transforms=None):
    if train_transforms is None:
        train_transforms= transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        #transforms.Grayscale(),
        transforms.ToTensor(),])
    if test_transforms is None:
        test_transforms= transforms.Compose([
        transforms.Resize((128, 128)),
        #transforms.Grayscale(),
        transforms.ToTensor(),])

    train_dataset = datasets.Imagenette(root='data160/test', split='train', size='160px', transform=train_transforms, download=False)
    val_dataset, test_dataset = random_split(datasets.Imagenette(root='data160/train', split='val', size='160px', transform=test_transforms, download=False), [0.5, 0.5], torch.Generator().manual_seed(555))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_loader

def sigmoid(z):
    return 1/(1 + np.exp(-z))

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
                #nn.Conv2d(64, 128, stride=1, kernel_size=3, padding=1),
                #nn.LeakyReLU(0.01),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                #nn.Conv2d(128, 256, stride=1, kernel_size=3, padding=1),
                #nn.LeakyReLU(0.01),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 32, stride=1, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 64, stride=1, kernel_size=3, padding=1),
                #nn.LeakyReLU(0.01),
                #nn.Upsample(16),
                #nn.ConvTranspose2d(256, 128, stride=1, kernel_size=3, padding=1),
                #nn.LeakyReLU(0.01),
                #nn.Upsample(32),
                #nn.ConvTranspose2d(128, 64, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.Upsample(64),
                nn.ConvTranspose2d(64, 32, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.Upsample(128),
                nn.ConvTranspose2d(32, 3, stride=1, kernel_size=3, padding=1),
                nn.Sigmoid(),
                )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
        
train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=2)
train_length = len(train_loader.dataset)
val_length = len(val_loader.dataset)
test_length = len(test_loader.dataset)
total_length = train_length + val_length + test_length
print(f'Length of train dataset: {train_length} ({train_length/total_length})')
print(f'Length of validation dataset: {val_length} ({val_length/total_length})')
print(f'Length of test dataset: {test_length} ({test_length/total_length})')
    
    
model = AutoEncoder().to(device)
model.load_state_dict(torch.load('32_32_32.pt', weights_only=True))
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
                                lr = LEARNING_RATE,
                                weight_decay = 0.01)
    
f, ax = plt.subplots(4, 8, figsize=(10, 5))
model.eval()
with torch.no_grad():
    for batch_idx, (test_features, _) in enumerate(test_loader):
        test_features = test_features.to(device)
        test_output = model(test_features)
        compr = model.encode(test_features)
            
        for image in test_features:
            if batch_idx == 0:
                input_image = np.transpose(image.cpu().numpy(), (1,2,0))
                compr_image = compr[idx].cpu().detach().numpy()
                
                #ax[0, 0].imshow(input_image)
                for i in range(32):
                    ax[int((i-i%8)/8), i%8].imshow(compr_image[i], cmap='binary')
                

for axs in ax:
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
plt.savefig('reconstructedImagesLatent.png')
plt.show()
