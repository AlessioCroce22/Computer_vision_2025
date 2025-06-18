import re
from torchvision.ops import generalized_box_iou_loss
import torch
import torch.nn as nn



#aggiungere funzioni che fanno i grafici




def extract_epoch(filename):
    match = re.search(r"checkpoint_epoch(\d+)\.pt", filename)
    return int(match.group(1)) if match else -1

def loss_bbox(pred_boxes, target_boxes):
   
    def to_xyxy(boxes):
        x_c, y_c, w, h = boxes.unbind(-1)
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    pred = to_xyxy(pred_boxes)
    target = to_xyxy(target_boxes)

    loss = generalized_box_iou_loss(pred, target, reduction='mean')
    return loss

loss_fn_ce = nn.CrossEntropyLoss(label_smoothing=0.05)

def loss_ocr(pred_logits, target_labels):
    B, T, V = pred_logits.size()
    pred_logits = pred_logits.reshape(B * T, V)
    targets = target_labels.view(B * T)
    return loss_fn_ce(pred_logits, targets)

def iou(box1, box2):
    
    def to_xyxy(b):
        x_c, y_c, w, h = b
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return x1, y1, x2, y2

    b1_x1, b1_y1, b1_x2, b1_y2 = to_xyxy(box1)
    b2_x1, b2_y1, b2_x2, b2_y2 = to_xyxy(box2)

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

def get_loss_weights_from_metrics(iou_mean, base_bbox=1.0, min_bbox=0.5):
    
    if iou_mean < 0.7:
        bbox_weight = base_bbox
        ocr_weight = 0.0
    elif 0.7 <= iou_mean < 0.8:
        t = (iou_mean - 0.7) / 0.1  
        bbox_weight = base_bbox * (1.0 - 0.5 * t) 
        ocr_weight = t  
    else:
        bbox_weight = base_bbox * 0.5
        ocr_weight = 1.0

    return float(bbox_weight), float(ocr_weight)