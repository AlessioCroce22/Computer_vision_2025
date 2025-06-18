from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import os
import torch
from globals import img_resolution,provinces,ads,char2idx,alphabets

class CustomImageLabelDataset(Dataset):
    def __init__(self, image_dir, label_dir, is_train=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                ], p=0.5),

                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3),
                    transforms.RandomAdjustSharpness(sharpness_factor=2)
                ], p=0.2),

                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=5,
                        translate=(0.02, 0.02),
                        scale=(0.95, 1.05),
                        shear=5,
                        interpolation=transforms.InterpolationMode.BILINEAR
                    )
                ], p=0.7),

                transforms.RandomApply([
                    transforms.RandomPerspective(distortion_scale=0.2, p=1.0)
                ], p=0.3),

                transforms.Resize((img_resolution, img_resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_resolution, img_resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        with open(label_path, 'r') as f:
            line = f.readline()
            numbers = list(map(float, line.strip().split()))
            bbox = torch.tensor(numbers[1:5], dtype=torch.float32) 

        parts = img_name.split('-')
        label_enc = [int(x) for x in parts[4].split('_')]
        label_str = []
        for i, code in enumerate(label_enc):
            if i == 0:
                label_str.append(provinces[code])
            elif i == 1:
                label_str.append(alphabets[code])
            else:
                label_str.append(ads[code])

        label_encoded = [char2idx[c] for c in label_str]
        label_tensor = torch.tensor(label_encoded, dtype=torch.long)

        return image, bbox, label_tensor

def get_dataloader(image_dir, label_dir, batch_size=8, shuffle=True, is_train=False):
    dataset = CustomImageLabelDataset(image_dir=image_dir, label_dir=label_dir, is_train=is_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def collate_fn(batch):
    images, bboxes, labels = zip(*batch)

    images = torch.stack(images)
    bboxes = torch.stack(bboxes)
    labels = torch.stack(labels) 

    return images, bboxes, labels