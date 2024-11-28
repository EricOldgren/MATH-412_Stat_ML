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

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_EPOCHS = 100
EARLY_STOPPING_EPOCHS = 20

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
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=2)
train_length = len(train_loader.dataset)
val_length = len(val_loader.dataset)
test_length = len(test_loader.dataset)
total_length = train_length + val_length + test_length
print(f'Length of train dataset: {train_length} ({train_length/total_length})')
print(f'Length of validation dataset: {val_length} ({val_length/total_length})')
print(f'Length of test dataset: {test_length} ({test_length/total_length})')

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                #nn.Conv2d(32, 64, stride=1, kernel_size=3, padding=1),
                #nn.LeakyReLU(0.01),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                #nn.Conv2d(64, 128, stride=1, kernel_size=3, padding=1),
                #nn.LeakyReLU(0.01),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                #nn.Conv2d(128, 256, stride=1, kernel_size=3, padding=1),
                #nn.LeakyReLU(0.01),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                #nn.Flatten(),
                #nn.Linear(4096, 4096),
                nn.Conv2d(32, 8, stride=1, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
                #nn.Linear(4096, 4096),
                #nn.Unflatten(1, (8, 8, 8)),
                nn.ConvTranspose2d(8, 32, stride=1, kernel_size=3, padding=1),
                #nn.LeakyReLU(0.01),
                #nn.Upsample(16),
                #nn.ConvTranspose2d(256, 128, stride=1, kernel_size=3, padding=1),
                #nn.LeakyReLU(0.01),
                #nn.Upsample(32),
                #nn.ConvTranspose2d(128, 64, stride=1, kernel_size=3, padding=1),
                #nn.LeakyReLU(0.01),
                #nn.Upsample(64),
                #nn.ConvTranspose2d(64, 32, stride=1, kernel_size=3, padding=1),
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

model = AutoEncoder().to(device)
#model.load_state_dict(torch.load('imageCompression50epochs.pt', weights_only=True))
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
                             lr = LEARNING_RATE,
                             weight_decay = 0.01)

def train_autoencoder(num_epochs, model, optimizer, train_loader, val_loader, early_stopping_epochs, save_model=None):
    losses = []
    val_losses = []
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
            running_val_loss = []
            for i, (val_feature, _) in enumerate(val_loader):
                val_feature = val_feature.to(device)
                val_output = model(val_feature)
                val_loss = loss_function(val_output, val_feature)
                running_val_loss.append(val_loss.item())
            val_losses.append(np.mean(running_val_loss))

        print('Epoch: %03d/%03d | Train Loss: %.4f | Val Loss: %.4f' % (epoch+1, num_epochs, np.mean(running_losses), np.mean(running_val_loss)))
        outputs.append([epoch, features, output])

        if len(val_losses) > early_stopping_epochs:
            if val_losses[-1] >= val_losses[-early_stopping_epochs-1]:
                print(f'Early Stopping... {val_losses[-1]} ; {val_losses[-early_stopping_epochs-1]}')
                model.load_state_dict(torch.load(f'epoch{epoch-1}.pt', weights_only=True))
                break

        torch.save(model.state_dict(), f'epoch{epoch}.pt')
            

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    return losses, val_losses, outputs

def plot_latent(model, test_loader):
  latents = []
  labels = []
  for batch_idx, (features, label) in enumerate(test_loader):
    model.eval()
    latent = model.encode(features).detach().numpy()
    label = label.detach().numpy()
    latents.extend(latent)
    labels.extend(label)

  for idx, point in enumerate(latents):
    plt.scatter(point[0], point[1], marker='.', s=20, color=colors[labels[idx]])
  plt.show()

def plot_loss_curve(losses, val_losses):
    plt.title('Loss curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.show()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

losses, val_losses, outputs = train_autoencoder(num_epochs=NUM_EPOCHS, model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, early_stopping_epochs=EARLY_STOPPING_EPOCHS, save_model="model1.pt")

#plot_latent(model=model, test_loader=test_loader)
plot_loss_curve(losses, val_losses)

f, ax = plt.subplots(2, BATCH_SIZE, figsize=(15, 5))
model.eval()
with torch.no_grad():
    test_losses = []
    ssim = []
    psnr = []
    for batch_idx, (test_features, _) in enumerate(test_loader):
        test_features = test_features.to(device)
        test_output = model(test_features)
        #compr = model.encode(test_features)
        test_loss = loss_function(test_output, test_features)
        test_losses.append(test_loss.item())
        for idx, image in enumerate(test_features):
            input_image = np.transpose(image.cpu().numpy(), (1,2,0))
            output_image = np.transpose(test_output[idx].cpu().detach().numpy(),(1,2,0))
            #compr_image = sigmoid(np.transpose(compr[idx].cpu().detach().numpy(),(1,2,0)))
            ssim.append(structural_similarity(input_image, output_image, data_range=1, channel_axis=2))
            psnr.append(peak_signal_noise_ratio(input_image, output_image, data_range=1))

            if batch_idx == 0:
                plt.imsave(f"original{idx}.png", input_image)
                plt.imsave(f"output{idx}.png", output_image)
            
                ax[0, idx].imshow(input_image)
                #ax[1, idx].imshow(compr_image)
                ax[1, idx].imshow(output_image)

plt.show()
print(f'Test Loss: {np.mean(test_losses)}')
print(f'Average SSIM: {np.mean(ssim)}')
print(f'Average PSNR: {np.mean(psnr)}')
