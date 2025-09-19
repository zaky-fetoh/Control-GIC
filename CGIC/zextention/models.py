import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torchvision as tv 
import torchvision.transforms as transforms
import torch.optim as optim

class ff121(nn.Module):
    def __init__(self, latent_dim=256, kernel_dim=256):
        super(ff121, self).__init__()
        self.fc1 = nn.Linear(latent_dim, kernel_dim)
        self.bn1 = nn.BatchNorm1d(kernel_dim)
        self.fc2 = nn.Linear(kernel_dim, kernel_dim)
        self.bn2 = nn.BatchNorm1d(kernel_dim)
        self.fc3 = nn.Linear(kernel_dim, latent_dim)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))  
        return self.bn3( self.fc3(x))
        
    def fit(self, dataloader, opt = optim.Adam, epochs = 10000, lr = 0.01):
        optimizer = opt(self.parameters(), lr=lr)
        for i in range(epochs):
            for batch_idx, (z, y) in enumerate(dataloader):
                optimizer.zero_grad()
                z = z.view(z.shape[0], -1)
                y = y.view(y.shape[0], -1)
                loss = F.mse_loss(self.forward(z), y)
                print("ff121 loss: ", loss.item())
                loss.backward()
                optimizer.step()
        return self

class VCAE(nn.Module):
    def __init__(self, encoder = VarConvEncoder(), decoder = ConvDecoder(),):
        super(VCAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.ff121 = ff121
    def forward(self, x):
        z, mu, sig = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z, mu, sig
    def get_acc_latent_x(self, x, opt = optim.Adam, optit = 1000):
        z, mu, sig = self.encoder(x)
        mu, sig = mu.detach(), sig.detach()
        latentv = torch.nn.Parameter( torch.randn_like(z))
        optimizer = opt([latentv], lr=0.1)
        for i in range(optit):
            optimizer.zero_grad()
            x_hat = self.decoder( mu + sig * latentv)
            loss = F.mse_loss(x_hat, x) + _ssim_loss(x_hat, x)
            print(loss.item())################################
            loss.backward()
            optimizer.step()
        return mu + sig * (latentv.detach())
    def set_memory_bank_x(self, data, optit = 10000):
        self.memory_bank_x = self.get_acc_latent_x(data, optit = optit)
        self.memory_bank_x = self.memory_bank_x.detach()

    def set_memory_bank_y(self, data, optit = 10000):
        self.memory_bank_y = self.get_acc_latent_x(data, optit = optit)
        self.memory_bank_y = self.memory_bank_y.detach()

    def get_memory_bank_loader(self, batch_size = 512):
        # Create dataset from memory banks
        memory_dataset = torch.utils.data.TensorDataset(
            self.memory_bank_x.view(self.memory_bank_x.shape[0], -1),
            self.memory_bank_y.view(self.memory_bank_y.shape[0], -1)
        )
        
        # Create and return dataloader
        return torch.utils.data.DataLoader(
            memory_dataset, shuffle=True,
            batch_size=batch_size,
        )

    def route(self, x, softargmaxit = 20, optit = 10):
        z = self.get_acc_latent_x(x, optit = optit)
        z = z.detach()
        z = z.view(z.shape[0], -1)
        print(z.shape) ###########################
        zz =torch.softmax(z @ self.memory_bank_x.T, dim=1)
        print(zz.shape) ###########################
        for _ in range(softargmaxit):
            z = torch.softmax(z @ self.memory_bank_x.T, dim=1) @ self.memory_bank_x
        z = torch.softmax(z @ self.memory_bank_x.T, dim=1) @ self.memory_bank_y
        z = z.view(z.shape[0], -1)
        dstimage = self.decoder(z)
        return dstimage

    def mapping121(self, z):
        z = z.view(z.shape[0], -1)
        z = self.ff121(z)
        return z
    def fit121(self, opt = optim.Adam, epochs = 1000, lr = 0.01):
        dataloader = self.get_memory_bank_loader()
        self.ff121.fit(dataloader, opt = opt, epochs = epochs, lr = lr)
        return self
    def route121(self, x, softargmaxit = 20, optit = 10):
        z = self.get_acc_latent_x(x, optit = optit)
        z = z.detach()
        z = z.view(z.shape[0], -1)
        z = self.ff121(z)
        dstimage = self.decoder(z)
        return dstimage
    def set_ff121(self, ff121):
        self.ff121 = ff121

