import numpy as np
import torch
import torch.nn.functional as F

try:
    from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False


def _safe_div(a, b, eps=1e-8):
    return float(a) / float(b + eps)


def _binary_counts(output_bool, target_bool):
    tp = float((output_bool & target_bool).sum())
    tn = float((~output_bool & ~target_bool).sum())
    fp = float((output_bool & ~target_bool).sum())
    fn = float((~output_bool & target_bool).sum())
    return tp, tn, fp, fn



def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)

    if HAS_MEDPY:
        try:
            hd95_ = hd95(output_, target_)
        except Exception:
            hd95_ = 0
    else:
        hd95_ = 0
    
    return iou, dice, hd95_


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    if HAS_MEDPY:
        iou_ = jc(output_, target_)
        dice_ = dc(output_, target_)
        try:
            hd_ = hd(output_, target_)
        except Exception:
            hd_ = np.nan
        try:
            hd95_ = hd95(output_, target_)
        except Exception:
            hd95_ = np.nan
        recall_ = recall(output_, target_)
        specificity_ = specificity(output_, target_)
        precision_ = precision(output_, target_)
    else:
        tp, tn, fp, fn = _binary_counts(output_, target_)
        iou_ = _safe_div(tp, tp + fp + fn)
        dice_ = _safe_div(2 * tp, 2 * tp + fp + fn)
        recall_ = _safe_div(tp, tp + fn)
        specificity_ = _safe_div(tn, tn + fp)
        precision_ = _safe_div(tp, tp + fp)
        hd_ = np.nan
        hd95_ = np.nan

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
