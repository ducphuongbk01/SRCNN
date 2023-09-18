import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from PIL import Image
from glob import glob

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class ImageTransform():
    def __init__(self, img_size, scale, mean, std):
        self.img_size = img_size
        self.lr_transform = transforms.Compose([
                                                transforms.Resize(int(img_size/scale)),
                                                transforms.Resize(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)
                                                ])
        self.hr_transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)
                                                ])
    
    def __call__(self, img):
        assert img.shape[0]==img.shape[1] and img.shape[0]==self.img_size, f"Image size should be {self.img_size}x{self.img_size}"
        lr_img = self.lr_transform(img)
        hr_img = self.hr_transform(img)
        return lr_img, hr_img


class T91Dataset(data.Dataset):
    def __init__(self, data_folder, transform = None):
        self.file_list = glob(data_folder+"/*.*")
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            train_img, label_img = self.transform(img)
        else:
            train_img = label_img = img
        return train_img, label_img


def load_loader(train_folder,
                val_folder,
                img_size: int = 33, 
                batch_size: int = 128, 
                num_workers: int = 1, 
                scale: float = 2.0):
    transform = ImageTransform(img_size=img_size, scale=scale, mean=MEAN, std=STD)

    train_dataset = T91Dataset(train_folder, transform=transform)
    val_dataset = T91Dataset(val_folder, transform=transform)


    # Create the dataloader object using the transforms (Not shuffled since we will be checking progress on the same examples)
    train_dataloader = torch.utils.data.DataLoader( train_dataset, 
                                                    batch_size = batch_size, 
                                                    num_workers = num_workers, 
                                                    shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(   val_dataset, 
                                                    batch_size = batch_size, 
                                                    num_workers = num_workers, 
                                                    shuffle = True)
    return train_dataloader, val_dataloader
