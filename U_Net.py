import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F # Added for sigmoid in Dice

# --- Dice Coefficient --- 
def dice_coeff(pred_logits, target_mask, smooth=1e-6):
    """Calculates the Dice coefficient for binary segmentation."""
    # Apply sigmoid to logits and threshold to get binary predictions
    pred_probs = torch.sigmoid(pred_logits)
    pred_binary = (pred_probs > 0.5).float()

    # Flatten prediction and target tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target_mask.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()

# --- U-Net Model Components ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- Full U-Net Model ---
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# --- Trainer Class ---
class UNetTrainer:
    def __init__(self, model, device, lr=1e-4, weight_decay=1e-8, 
                 early_stopping_patience=10, min_metric_delta=0.001):
        self.model = model.to(device)
        self.device = device

        # —— Loss 设置 —— #
        if model.n_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                             mode='min',
                                                             factor=0.5,
                                                             patience=3) # Scheduler patience

        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.min_metric_delta = min_metric_delta
        self.best_val_dice = -float('inf')
        self.epochs_no_improve = 0
        self.early_stop_triggered = False

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        for images, true_masks in train_loader:
            images = images.to(self.device, dtype=torch.float32)
            true_masks = true_masks.to(self.device, dtype=torch.float32)
            true_masks = torch.clamp(true_masks, 0.0, 1.0)

            self.optimizer.zero_grad()
            masks_pred = self.model(images)

            if self.model.n_classes == 1:
                loss = self.criterion(masks_pred, true_masks)
            else:
                target = true_masks.squeeze(1).long()
                loss = self.criterion(masks_pred, target)

            # 稳定性检查
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("Loss became NaN or Inf. Check inputs and lr.")
            if loss.item() < 0:
                print("!!! WARNING: loss < 0, debugging stats …")
                print(f" pred min/max = {masks_pred.min().item():.3f}/{masks_pred.max().item():.3f}")
                print(f" mask min/max = {true_masks.min().item():.3f}/{true_masks.max().item():.3f}")

            loss.backward()
            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            # Calculate Dice for this batch (optional to average over epoch)
            # For simplicity, we'll primarily focus on validation Dice for decisions

        avg_loss = epoch_loss / len(train_loader)
        # Note: train_dice calculation could be added here if needed for detailed logging
        # self.scheduler.step(avg_loss) # Will be moved to main train loop, based on val_loss
        return avg_loss


    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_dice_scores = []
        with torch.no_grad():
            for images, true_masks in val_loader:
                images = images.to(self.device, dtype=torch.float32)
                true_masks = true_masks.to(self.device, dtype=torch.float32)
                true_masks = torch.clamp(true_masks, 0.0, 1.0)

                masks_pred_logits = self.model(images)

                if self.model.n_classes == 1:
                    loss = self.criterion(masks_pred_logits, true_masks)
                else:
                    target = true_masks.squeeze(1).long()
                    loss = self.criterion(masks_pred_logits, target)
                
                val_loss += loss.item()
                dice = dice_coeff(masks_pred_logits, true_masks)
                val_dice_scores.append(dice)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = np.mean(val_dice_scores) if val_dice_scores else 0
        return avg_val_loss, avg_val_dice

    def train(self, train_loader, val_loader, epochs=5, model_save_path='./models/unet_best_isbi.pth'):
        print(f"Starting training for {epochs} epochs on device: {self.device}")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # Ensure directory exists

        for epoch in range(epochs):
            avg_train_loss = self.train_epoch(train_loader)
            avg_val_loss, avg_val_dice = self.evaluate(val_loader)

            print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}')

            self.scheduler.step(avg_val_loss) # Step scheduler based on validation loss

            # Early stopping and model saving logic
            if avg_val_dice > self.best_val_dice + self.min_metric_delta:
                self.best_val_dice = avg_val_dice
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), model_save_path)
                print(f'Validation Dice improved to {avg_val_dice:.4f}. Model saved to {model_save_path}')
            else:
                self.epochs_no_improve += 1
                print(f'Validation Dice did not improve for {self.epochs_no_improve} epoch(s). Best Dice: {self.best_val_dice:.4f}')

            if self.epochs_no_improve >= self.early_stopping_patience:
                self.early_stop_triggered = True
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break
        
        if not self.early_stop_triggered:
            print('Training completed without early stopping.')
        print(f'Best validation Dice score: {self.best_val_dice:.4f}')

