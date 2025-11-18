import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Более простая версия (меньше параметров)
class SimpleImageAutoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(SimpleImageAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 178x218 -> 89x109
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # 89x109 -> 45x55
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # 45x55 -> 23x28
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # 23x28 -> 12x14
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # 12x14 -> 23x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # 23x28 -> 45x55
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            
            # 45x55 -> 89x109
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            
            # 89x109 -> 178x218
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Функции для обучения
def train_model(model, train_loader, val_loader, epochs=50, model_type='autoencoder'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            if model_type == 'vae':
                recon_batch, mu, logvar = model(data)
                loss = vae_loss(recon_batch, data, mu, logvar)
            else:
                output = model(data)
                loss = nn.MSELoss()(output, data)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                if model_type == 'vae':
                    recon_batch, mu, logvar = model(data)
                    loss = vae_loss(recon_batch, data, mu, logvar)
                else:
                    output = model(data)
                    loss = nn.MSELoss()(output, data)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Сохранение модели каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'autoencoder_epoch_{epoch+1}.pth')
    
    return train_losses, val_losses

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Функции для визуализации
def visualize_results(model, test_loader, num_images=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:num_images].to(device)
        
        if isinstance(model, VAE_178x218):
            output, _, _ = model(data)
        else:
            output = model(data)
        
        # Визуализация
        fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
        
        for i in range(num_images):
            # Оригинальные изображения
            orig_img = data[i].cpu().permute(1, 2, 0).numpy()
            axes[0, i].imshow(np.clip(orig_img, 0, 1))
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Восстановленные изображения
            recon_img = output[i].cpu().permute(1, 2, 0).numpy()
            axes[1, i].imshow(np.clip(recon_img, 0, 1))
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Создание датасета
def create_data_loaders(data_path, batch_size=16, img_size=(218, 178)):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    dataset = ImageFolder(root=data_path, transform=transform)
    
    # Разделение на train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Пример использования
if __name__ == "__main__":
    # Параметры
    DATA_PATH = "path/to/your/images"  # Укажите путь к вашим изображениям
    BATCH_SIZE = 8
    EPOCHS = 30
    
    # Создание даталоадеров
    train_loader, val_loader = create_data_loaders(DATA_PATH, BATCH_SIZE)
    
    # Инициализация модели
    model = SimpleImageAutoencoder(input_channels=3)  # 3 канала для RGB
    
    # Обучение
    print("Начало обучения...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=EPOCHS, model_type='autoencoder'
    )
    
    # Визуализация результатов
    visualize_results(model, val_loader)
    
    # График обучения
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.show()
    
    # Сохранение финальной модели
    torch.save(model.state_dict(), 'final_autoencoder_178x218.pth')
    print("Модель сохранена!")