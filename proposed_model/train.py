import torch
import glob
from tqdm import tqdm
from pathlib import Path
from data import get_dataloader
from utils import loss_bbox,loss_ocr,iou,extract_epoch,get_loss_weights_from_metrics
from globals import batch,lr,weight_decay,save_every,epochs,device,idx2char
from network import ViTDETRLPR


train_loader = get_dataloader(
    image_dir='./CCPD2020/images/train',
    label_dir='./CCPD2020/labels/train',
    batch_size=batch,
    is_train=True
)
val_loader = get_dataloader(
    image_dir='./CCPD2020/images/val',
    label_dir='./CCPD2020/labels/val',
    batch_size=batch
)




def run_epoch(model, dataloader, optimizer=None, device='cuda', is_train=True, weight_bbox=1.0, weight_ocr=1.0,epoch=0):
    model.train() if is_train else model.eval()
    if epoch<100:
        scheduled_sampling_prob = 0.30
    elif epoch>=100 and epoch<250:
        scheduled_sampling_prob = 0.50
    else:
        scheduled_sampling_prob = epoch/500     
    total_iou = 0.0
    total_loss, total_bbox_loss, total_ocr_loss = 0, 0, 0
    correct_bbox, correct_ocr_seq, correct_ocr_char = 0, 0, 0
    total_samples = 0
    total_chars = 0

    loop = tqdm(dataloader, desc="Train" if is_train else "Val")
    for batch_idx, (images, target_boxes, target_labels) in enumerate(loop):
        images = images.to(device)
        target_boxes = target_boxes.to(device)
        target_labels = target_labels.to(device)

        with torch.set_grad_enabled(is_train):
            if is_train:
                pred_boxes, pred_ocr = model(images, target_labels, teacher_forcing=True,scheduled_sampling_prob=scheduled_sampling_prob)
            else:
                pred_boxes, pred_ocr = model(images)

            lbbox = loss_bbox(pred_boxes, target_boxes)
            locr = loss_ocr(pred_ocr, target_labels)

            loss = weight_bbox  * lbbox + weight_ocr * locr
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        B = images.size(0)
        total_loss += loss.item() * B
        total_bbox_loss += lbbox.item() * B
        total_ocr_loss += locr.item() * B
        total_samples += B
        total_chars += B * 8

        pred_text = pred_ocr.argmax(dim=2)  
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
    return metrics

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, checkpoint_dir="checkpoints", device='cuda', save_every=50):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50, min_lr=5e-5)

    ckpt_files = glob.glob(f"{checkpoint_dir}/checkpoint_epoch*.pt")
    ckpt_files.sort(key=extract_epoch)

    if ckpt_files:
        latest_ckpt = ckpt_files[-1]
        print(f"🔄 Caricamento da checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        best_acc_seq = checkpoint.get('best_acc_seq', 0.0)
        best_loss_ocr=checkpoint.get('best_loss_ocr', 10.0)
        start_epoch = checkpoint['epoch'] + 1
        iou_mean = checkpoint['iou_mean']
    else:
        print("🆕 Nessun checkpoint trovato, training da zero.")
        best_acc_seq = 0.0
        start_epoch = 1
        iou_mean = 0.1
        best_loss_ocr=10.0

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n🌟 Epoch {epoch}/{epochs}")

        weight_bbox, weight_ocr = get_loss_weights_from_metrics(iou_mean)

        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, device=device, is_train=True,
                                  weight_bbox=weight_bbox, weight_ocr=weight_ocr,epoch=epoch)
        val_metrics = run_epoch(model, val_loader, optimizer=None, device=device, is_train=False,
                                weight_bbox=weight_bbox, weight_ocr=weight_ocr,epoch=epoch)

        iou_mean = val_metrics['iou_mean']
        print(f"\n📊 Train | Loss: {train_metrics['loss_total']:.4f}, LossBBox: {train_metrics['loss_bbox']:.4f}, LossOcr: {train_metrics['loss_ocr']:.4f}, BBox Acc: {train_metrics['acc_bbox']:.4f}, Mean IoU: {train_metrics['iou_mean']:.4f}, OCR Acc: {train_metrics['acc_ocr_seq']:.4f}, Char Acc: {train_metrics['acc_ocr_char']:.4f}")
        print(f"📊  Val  | Loss: {val_metrics['loss_total']:.4f}, LossBBox: {val_metrics['loss_bbox']:.4f}, LossOcr: {val_metrics['loss_ocr']:.4f}, BBox Acc: {val_metrics['acc_bbox']:.4f}, Mean IoU: {val_metrics['iou_mean']:.4f}, OCR Acc: {val_metrics['acc_ocr_seq']:.4f}, Char Acc: {val_metrics['acc_ocr_char']:.4f}")

        scheduler.step(val_metrics['loss_total'])

        if epoch % save_every == 0:
            ckpt_path = f"{checkpoint_dir}/checkpoint_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'iou_mean': iou_mean,
                'best_acc_seq': best_acc_seq,
                'best_loss_ocr': best_loss_ocr
            }, ckpt_path)

        if val_metrics['acc_ocr_seq'] > best_acc_seq:
            best_acc_seq = val_metrics['acc_ocr_seq']
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pt")
            print("💾 Salvato miglior modello (OCR Acc Seq migliorata)!")
        if val_metrics['loss_ocr'] < best_loss_ocr:
            best_loss_ocr = val_metrics['loss_ocr']
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_loss_model.pt")
            print("💾 Salvato miglior modello (OCR loss migliorata)!")    

model = ViTDETRLPR(vocab_size=len(idx2char)).to(device)
train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, checkpoint_dir="./CCPD2020/baseline_checkpoints_prova", device=device, save_every=save_every)
