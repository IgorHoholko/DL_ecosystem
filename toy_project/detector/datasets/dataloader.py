from torch.utils.data import DataLoader
from torchvision import transforms
from detector.datasets import SVHNDataset
from detector.datasets import get_augmentations
from detector.datasets import DatasetSequence
import numpy as np



def get_loaders(config):
    """Builds and returns Dataloaders for SVHN* dataset in format:
        (train_loader, val_loader, test_loader)"""

    augmentations = None
    if config['dataset_args'].get('augmentations', None):
        augmentations = get_augmentations()

    random_seed = config.get('random_seed', None)
    if random_seed:
        np.random.seed(random_seed)

    svhn = SVHNDataset(subsample_fraction=config['dataset_args'].get('subsample_fraction', None))
    svhn.load_or_generate_data()
    #TODO: if we had more datasets, we would combine them here..

    x_train = svhn.x_train
    y_train = svhn.y_train

    validation_ration = config['dataset_args'].get('validation_ration', None)
    if validation_ration:
        n_samples = x_train.shape[0]
        val_idxs = np.random.choice(np.arange(n_samples),
                                    size=int(n_samples*validation_ration),
                                    replace=False)
        train_idxs = np.array( list( set(np.arange(n_samples)) - set(val_idxs)) )
        assert len(val_idxs) + len(train_idxs) == n_samples, "Wrong Split to test/val"
        x_val, y_val = x_train[val_idxs], y_train[val_idxs]
        x_train, y_train = x_train[train_idxs], y_train[train_idxs]
    x_test = svhn.x_test
    y_test = svhn.y_test


    train_dataset = DatasetSequence(x_train, y_train, augmentations)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['train_args'].get('batch_size'),
                              shuffle=True,)
    test_dataset = DatasetSequence(x_test, y_test)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=config['train_args'].get('batch_size'),
                              shuffle=False,)
    if validation_ration:
        val_dataset = DatasetSequence(x_val, y_val)
        val_loader = DataLoader(dataset=val_dataset,
                                 batch_size=config['train_args'].get('batch_size'),
                                 shuffle=False,)

        return (train_loader, val_loader, test_loader)
    else:
        return (train_loader, None, test_loader)




