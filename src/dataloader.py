from PIL import Image
import random
import torch
import torchvision.transforms as T

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, data: list, is_train: bool):
        super().__init__()
        self.data = data
        self.img_paths = [x[0] for x in data]
        self.labels = torch.tensor([x[1] for x in data]) 

        if is_train:
            self.transforms = T.Compose([
                #T.Resize((224, 224)),
                T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomApply([
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ], p=0.5),
                T.RandomApply([T.RandomRotation(180)], p=0.5),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        img = self.transforms(img)
        return img, label, img_path
    

def patch_dataloaders(data: list, is_train: bool, batch_size: int, num_workers: int = 4):

    dataset = PatchDataset(data=data, is_train=is_train)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers
    )
