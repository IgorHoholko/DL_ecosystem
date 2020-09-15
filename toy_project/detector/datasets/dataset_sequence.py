#Implementation of general PyTorch Dataset class to combine all data we have

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DatasetSequence(Dataset):
    def __init__(self, x, y,
                 transforms = None,
                 augments = None
                 ):
        self.x = x
        self.y = y
        self.transforms = transforms
        self.augments = augments

        self.n = x.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]
        if self.augments:
            image = self.augments(image=image)['image']
        if self.transforms:
            image = self.transforms(image)

        sample = {'image':image, 'label': label}
        return sample