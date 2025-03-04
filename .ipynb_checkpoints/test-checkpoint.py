import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
import wandb
import numpy as np
import matplotlib.pyplot as plt
from evaluate import evaluate
from unet.unet_model import UNet
from unet.Dulbranch_res import DualBranchUNetCBAMResnet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from torch.utils.tensorboard import SummaryWriter
from utils.distance_transform import one_hot2dist, SurfaceLoss
from utils.dice_score import multiclass_dice_coeff, dice_coeff
class CombinedLoss(nn.Module):
    def __init__(self, idc, surface_loss_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.idc = idc
        self.surface_loss_weight = surface_loss_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.surface_loss = SurfaceLoss(idc=idc)

    def forward(self, logits, edge_logits, targets, dist_maps, writer, global_step):
        ce_loss = self.ce_loss(logits, targets)
        
        edge_logits = torch.sigmoid(edge_logits)
        surface_loss = self.surface_loss(edge_logits, dist_maps)
        print("surface_loss: ", surface_loss)
        writer.add_scalar('surface_loss/surface_loss', surface_loss, global_step)
        total_loss = ce_loss + surface_loss*self.surface_loss_weight
        writer.add_scalar('surface_loss/surface_loss_weight', self.surface_loss_weight, global_step)
        return total_loss

def overlay_two_masks(groundtruth_mask, pred_mask, alpha=0.5, pred_alpha=0.5):
    groundtruth_mask = groundtruth_mask.squeeze(0).cpu().numpy()
    pred_mask = pred_mask.squeeze(0).cpu().numpy()

    overlay_image = np.zeros((groundtruth_mask.shape[0], groundtruth_mask.shape[1], 3))

    overlay_image[groundtruth_mask == 1] = [0, 1, 0]
    overlay_image = overlay_image * (1 - alpha)

    pred_overlay = np.zeros_like(overlay_image)
    pred_overlay[pred_mask == 1] = [1, 0, 0]
    overlay_image = overlay_image + pred_overlay * pred_alpha

    return overlay_image

def overlay_mask_on_image(image, mask, alpha=0.5):
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze(0).cpu().numpy()

    mask_overlay = np.zeros_like(image)
    mask_overlay[mask == 1] = [1, 1, 1]

    overlay = image * (1 - alpha) + mask_overlay * alpha

    return overlay

def test_model(model, device, test_loader, criterion, model_name="DBUCR"):
    model.eval()
    total_dice_score = 0
    total_pixel_accuracy = 0
    total_iou = 0
    total_f1 = 0
    total_recall = 0
    total_precision = 0
    total_batches = 0

    writer = SummaryWriter(log_dir=f'/root/tf-logs/{time.strftime("%Y-%m-%d_%H-%M-%S")}/{model_name}')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            image, masks_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks_true = masks_true.to(device=device, dtype=torch.long)

            # predict the mask
            masks_pred, _ = model(image)
            
            mask_pred = F.softmax(masks_pred, dim=1)  # Get probability map
            mask_pred = mask_pred.argmax(dim=1)  # Get predicted class per pixel
            mask_pred = F.one_hot(mask_pred, model.n_classes).permute(0, 3, 1, 2).float()

            # Convert true mask to one-hot encoding
            mask_true = F.one_hot(masks_true, model.n_classes).permute(0, 3, 1, 2).float()

            # Compute Dice score
            dice=multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            total_dice_score += dice

            # Compute Pixel Accuracy
            pixel_accuracy = (mask_pred.argmax(dim=1) == mask_true.argmax(dim=1)).float().mean().item()
            total_pixel_accuracy += pixel_accuracy

            # Compute IoU for each class
            iou_classwise = []
            f1_classwise = []
            recall_classwise = []
            precision_classwise = []
            for c in range(1, model.n_classes):  # We skip background class (index 0)
                intersection = torch.sum(mask_pred[:, c] * mask_true[:, c])
                union = torch.sum(mask_pred[:, c]) + torch.sum(mask_true[:, c])
                iou = intersection / (union - intersection + 1e-6)
                iou_classwise.append(iou.item())

                # Compute F1, Recall, and Precision for each class
                true_positives = torch.sum(mask_pred[:, c] * mask_true[:, c])
                false_positives = torch.sum(mask_pred[:, c] * (1 - mask_true[:, c]))
                false_negatives = torch.sum((1 - mask_pred[:, c]) * mask_true[:, c])

                precision = true_positives / (true_positives + false_positives + 1e-6)
                recall = true_positives / (true_positives + false_negatives + 1e-6)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

                f1_classwise.append(f1.item())
                recall_classwise.append(recall.item())
                precision_classwise.append(precision.item())

            mean_iou = sum(iou_classwise) / len(iou_classwise)  # Mean IoU across all classes (ignoring background)
            mean_f1 = sum(f1_classwise) / len(f1_classwise)
            mean_recall = sum(recall_classwise) / len(recall_classwise)
            mean_precision = sum(precision_classwise) / len(precision_classwise)

            total_iou += mean_iou
            total_f1 += mean_f1
            total_recall += mean_recall
            total_precision += mean_precision
            
            
            overlay_image = overlay_two_masks(
                        masks_true[0],  # Ground truth mask
                        masks_pred.argmax(dim=1)[0],  # Prediction mask
                        alpha=0.5,  # Ground truth mask 的透明度
                        pred_alpha=0.7  # Prediction mask 的透明度
                    )
            writer.add_image('Test-dual-Res/MaskOverlay', torch.tensor(overlay_image).permute(2, 0, 1), batch_idx)
            overlay_image = overlay_mask_on_image(image[0], masks_pred.argmax(dim=1)[0])

            writer.add_image('Test-dual-Res/Overlay', torch.tensor(overlay_image).permute(2, 0, 1), batch_idx)
            writer.add_image('Test-dual-Res/Image', image[0], batch_idx)  
            writer.add_image('Test-dual-Res/Mask', masks_true[0].unsqueeze(0), batch_idx)  
            writer.add_image('Test-dual-Res/Prediction', masks_pred.argmax(dim=1)[0].unsqueeze(0), batch_idx)  
            
    # Calculate average metrics
    
                
           
            writer.add_scalar('Dice/Test', dice, batch_idx)
            writer.add_scalar('Accuracy/Test', pixel_accuracy, batch_idx)
            writer.add_scalar('IoU/Test', mean_iou, batch_idx)
            writer.add_scalar('f1/Test', mean_f1, batch_idx)
            writer.add_scalar('Recall/Test', mean_recall, batch_idx)
            writer.add_scalar('Precision/Test', mean_precision, batch_idx)
            total_batches+=1

    avg_dice_score = total_dice_score / total_batches
    avg_pixel_accuracy = total_pixel_accuracy / total_batches
    avg_iou = total_iou / total_batches
    avg_f1 = total_f1 / total_batches
    avg_recall = total_recall / total_batches
    avg_precision = total_precision / total_batches
    writer.add_scalar('Dice/avgTest', avg_dice_score, 0)
    writer.add_scalar('Accuracy/avgTest',avg_pixel_accuracy, 0)
    writer.add_scalar('IoU/avgTest', avg_iou, 0)
    writer.add_scalar('f1/avgTest', avg_f1, 0)
    writer.add_scalar('Recall/avgTest', avg_recall, 0)
    writer.add_scalar('Precision/avgTest', avg_precision, 0)

    writer.close()

    return avg_dice_score, avg_pixel_accuracy, avg_iou, avg_f1, avg_recall, avg_precision

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DualBranchUNetCBAMResnet(n_classes=2, n_channels=3).to(device)

# 加载模型状态字典
checkpoint = torch.load('Dual.pth', map_location=device)

# 删除状态字典中不需要的键
if 'mask_values' in checkpoint:
    del checkpoint['mask_values']

# 加载模型状态，忽略不匹配的键
model.load_state_dict(checkpoint, strict=False)
model.to(memory_format=torch.channels_last)

# 初始化损失函数
criterion = CombinedLoss(idc=[1], surface_loss_weight=1)

# 加载测试数据集
test_dir_img = Path('./data/imgs/test/')
test_dir_mask = Path('./data/masks/test/')
test_dataset = CarvanaDataset(test_dir_img, test_dir_mask,0.5)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
# 测试模型
test_dice, test_acc, test_iou, test_f1, test_recall, test_precision = test_model(
    model=model,
    device=device,
    test_loader=test_loader,
    criterion=criterion,
    model_name="DBUCR"
)