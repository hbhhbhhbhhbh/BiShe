import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
#我是让科研变的更简单的叫叫兽！国奖，多篇SCI，深耕目标检测领域，多项竞赛经历，拥有软件著作权，核心期刊等成果。实战派up主，只做干货！让你不走弯路，直冲成果输出！！

# 大家关注我的B站：Ai学术叫叫兽
# 链接在这：https://space.bilibili.com/3546623938398505
# 科研不痛苦，跟着叫兽走！！！
# 更多捷径——B站干货见！！！

# 本环境和资料纯粉丝福利！！！
# 必须让我叫叫兽的粉丝有牌面！！！冲吧，青年们，遥遥领先！！！
@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    total_dice_score = 0
    total_pixel_accuracy = 0
    total_iou = 0
    total_batches = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                # For binary classification
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

                # Compute Dice score
                total_dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                
                # Compute Pixel Accuracy
                pixel_accuracy = (mask_pred == mask_true).float().mean().item()
                total_pixel_accuracy += pixel_accuracy

                # Compute IoU
                intersection = torch.sum(mask_pred * mask_true)
                union = torch.sum(mask_pred) + torch.sum(mask_true)
                iou = intersection / (union - intersection + 1e-6)
                total_iou += iou.item()

            else:
                # For multi-class classification
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
                mask_pred = F.softmax(mask_pred, dim=1)  # Get probability map
                mask_pred = mask_pred.argmax(dim=1)  # Get predicted class per pixel
                mask_pred = F.one_hot(mask_pred, net.n_classes).permute(0, 3, 1, 2).float()

                # Convert true mask to one-hot encoding
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

                # Compute Dice score
                total_dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

                # Compute Pixel Accuracy
                pixel_accuracy = (mask_pred.argmax(dim=1) == mask_true.argmax(dim=1)).float().mean().item()
                total_pixel_accuracy += pixel_accuracy

                # Compute IoU for each class
                iou_classwise = []
                for c in range(1, net.n_classes):  # We skip background class (index 0)
                    intersection = torch.sum(mask_pred[:, c] * mask_true[:, c])
                    union = torch.sum(mask_pred[:, c]) + torch.sum(mask_true[:, c])
                    iou = intersection / (union - intersection + 1e-6)
                    iou_classwise.append(iou.item())
                mean_iou = sum(iou_classwise) / len(iou_classwise)  # Mean IoU across all classes (ignoring background)
                total_iou += mean_iou

            total_batches += 1

    # Calculate average metrics
    avg_dice_score = total_dice_score / total_batches
    avg_pixel_accuracy = total_pixel_accuracy / total_batches
    avg_iou = total_iou / total_batches

    net.train()
    return avg_dice_score, avg_pixel_accuracy, avg_iou
