import torch
from tqdm import tqdm
from globals import device, batch, idx2char
from network import ViTDETRLPR
from utils import loss_bbox,loss_ocr,iou
from data import get_dataloader

test_loader = get_dataloader(
    image_dir='./CCPD2020/images/test',
    label_dir='./CCPD2020/labels/test',
    batch_size=batch
)

def evaluate_on_test_set(model, test_loader, checkpoint_path, device='cuda'):
    print("🔍 Caricamento modello migliore da checkpoint:", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    total_iou = 0.0
    total_loss, total_bbox_loss, total_ocr_loss = 0, 0, 0
    correct_bbox, correct_ocr_seq, correct_ocr_char = 0, 0, 0
    total_samples = 0
    total_chars = 0

    with torch.no_grad():
        loop = tqdm(test_loader, desc="Test")
        for batch_idx, (images, target_boxes, target_labels) in enumerate(loop):
            images = images.to(device)
            target_boxes = target_boxes.to(device)
            target_labels = target_labels.to(device)

            pred_boxes, pred_ocr = model(images)

            lbbox = loss_bbox(pred_boxes, target_boxes)
            locr = loss_ocr(pred_ocr, target_labels)

            B = images.size(0)
            total_loss += (lbbox + locr).item() * B
            total_bbox_loss += lbbox.item() * B
            total_ocr_loss += locr.item() * B
            total_samples += B
            total_chars += B * 8

            pred_text = pred_ocr.argmax(dim=2)  # [B, 8]
            for i in range(B):
                iou_val = iou(pred_boxes[i].detach().cpu(), target_boxes[i].detach().cpu())
                total_iou += iou_val
                if iou_val > 0.7:
                    correct_bbox += 1
                if iou_val > 0.6 and torch.equal(pred_text[i].cpu(), target_labels[i].cpu()):
                    correct_ocr_seq += 1
                correct_ocr_char += (pred_text[i] == target_labels[i]).sum().item()

    metrics = {
        'loss_total': total_loss / total_samples,
        'loss_bbox': total_bbox_loss / total_samples,
        'loss_ocr': total_ocr_loss / total_samples,
        'acc_bbox': correct_bbox / total_samples,
        'iou_mean': total_iou / total_samples,
        'acc_ocr_seq': correct_ocr_seq / total_samples,
        'acc_ocr_char': correct_ocr_char / total_chars
    }

    print("\n📌 Risultati Test Set:")
    print(f"🔢 Total Loss:    {metrics['loss_total']:.4f}")
    print(f"📦 BBox Loss:     {metrics['loss_bbox']:.4f}")
    print(f"🔤 OCR Loss:      {metrics['loss_ocr']:.4f}")
    print(f"📈 Mean IoU:      {metrics['iou_mean']:.4f}")
    print(f"✅ BBox Accuracy: {metrics['acc_bbox']:.4f}")
    print(f"✅ OCR Seq Acc:   {metrics['acc_ocr_seq']:.4f}")
    print(f"✅ OCR Char Acc:  {metrics['acc_ocr_char']:.4f}")

    return metrics

model = ViTDETRLPR(vocab_size=len(idx2char)).to(device)
checkpoint_path = "./CCPD2020/baseline_checkpoints_size_final/best_model.pt"
evaluate_on_test_set(model, test_loader, checkpoint_path, device=device)    