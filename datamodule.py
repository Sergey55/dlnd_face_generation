import torch
import pytorch_lightning as pl

from torchvision import datasets
from torchvision import transforms

class FaceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=128, image_size=32, num_workers=0):
        """Constructor

        Args:
            data_dir        Directory where images data is located
            batch_size      The size of each batch; the number of images in a batch
            image_size      The squaer size of the image data
            num_workers     
        """
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

    def train_dataloader(self):
        transformation = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        dataset = datasets.ImageFolder(self.data_dir, transform=transformation)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        return data_loader