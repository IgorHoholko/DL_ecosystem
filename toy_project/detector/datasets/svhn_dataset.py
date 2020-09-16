from detector.datasets.dataset import Dataset, _parse_args
import os
import numpy as np
import h5py
import shutil
from detector.util import download_urls

SAMPLE_TO_BALANCE = True # If true, take at most the mean number of instances per class.

URL = 'http://ufldl.stanford.edu/housenumbers'

TRAIN_FILE_NAMES = ['train_32x32.mat', 'extra_32x32.mat']
TEST_FILE_NAMES = ['test_32x32.mat']
FILE_NAMES = (*TRAIN_FILE_NAMES, *TEST_FILE_NAMES)

RAW_DATA_DIRNAME = Dataset.data_dirname() / "raw" / "svhn_dataset"
PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / "processed" / "svhn_dataset"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'data.h5'

ESSENTIAL_FILE = Dataset.data_dirname() / 'svhn_loaded.flag'


class SVHNDataset(Dataset):
    """SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal
     requirement on data preprocessing and formatting. SVHN is obtained from house numbers in Google Street View images.

     10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
     73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples
     Comes in two formats:
        1. Original images with character level bounding boxes.
        2. MNIST-like 32-by-32 images centered around a single character (many of the images do contain some distractors
            at the sides).

    From https://www.nist.gov/itl/iad/image-group/emnist-dataset"""

    def __init__(self, subsample_fraction: float = None):
        if not os.path.exists(ESSENTIAL_FILE):
            _download_and_process_cvhn()

        self.subsample_fraction = subsample_fraction
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_or_generate_data(self):
        if not os.path.exists(ESSENTIAL_FILE):
            _download_and_process_cvhn()
        with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
            self.x_train = f["x_train"][:]
            self.y_train = f["y_train"][:]
            self.x_test = f["x_test"][:]
            self.y_test = f["y_test"][:]
        self._subsample()

    def _subsample(self):
        """Only this fraction of data will be loaded."""
        if self.subsample_fraction is None:
            return
        num_train = int(self.x_train.shape[0] * self.subsample_fraction)
        num_test = int(self.x_test.shape[0] * self.subsample_fraction)
        sub_train_ids = np.random.choice(np.arange(self.x_train.shape[0]),
                                         size=num_train, replace=False)
        sub_test_ids = np.random.choice(np.arange(self.x_test.shape[0]),
                                         size=num_test, replace=False)
        self.x_train = self.x_train[sub_train_ids]
        self.y_train = self.y_train[sub_train_ids]
        self.x_test = self.x_test[sub_test_ids]
        self.y_test = self.y_test[sub_test_ids]




def _download_and_process_cvhn():
    if not os.path.exists(RAW_DATA_DIRNAME):
        os.makedirs(RAW_DATA_DIRNAME)
    curdir = os.getcwd()
    os.chdir(RAW_DATA_DIRNAME)
    urls = [URL + '/' + filename for filename in FILE_NAMES]
    download_urls(urls, FILE_NAMES)
    _process_raw_dataset()
    os.chdir(curdir)


def _process_raw_dataset():
    from scipy.io import loadmat
    def load_mat(path):
        data = loadmat(path)
        return data['X'], data['y']

    print("Loading training data from .mat file...")
    x_train, y_train = np.array([]), np.array([])

    for filename in TRAIN_FILE_NAMES:
        x, y = load_mat(filename)
        x = x.transpose((3, 0, 1, 2))
        y = y[:, 0]
        x_train = np.vstack((x_train, x)) if len(x_train) else x
        y_train = np.hstack((y_train, y)) if len(y_train) else y

    x, y = load_mat(TEST_FILE_NAMES[0])
    x_test = x.transpose((3, 0, 1, 2))
    y_test = y[:, 0]

    # replace label "10" which represents 0 to label "0"
    y_train[np.where(y_train == 10)] = 0
    y_test[np.where(y_test == 10)] = 0

    if SAMPLE_TO_BALANCE:
        print("Balancing classes to reduce amount of data...")
        x_train, y_train = _sample_to_balance(x_train, y_train)
        x_test, y_test = _sample_to_balance(x_test, y_test)

    print("Saving to HDF5 in a compressed format...")
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
        f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
        f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
        f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
        f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

    # make mark that dataset is uploaded
    with open(ESSENTIAL_FILE, 'w') as f:
        f.write('Ok')

    print("Cleaning up...")
    shutil.rmtree(".")


def _sample_to_balance(x, y):
    """Because the dataset is not balanced, we take at most the mean number of instances per class."""
    np.random.seed(42)
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_inds = []
    for label in np.unique(y.flatten()):
        inds = np.where(y == label)[0]
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[ind]
    y_sampled = y[ind]
    return x_sampled, y_sampled


def main():
    """Load SVHN dataset and print info."""
    args = _parse_args()
    dataset = SVHNDataset(args.subsample_fraction)
    dataset.load_or_generate_data()

    print(dataset.x_train.shape, dataset.y_train.shape)  # pylint: disable=E1101
    print(dataset.x_test.shape, dataset.y_test.shape)  # pylint: disable=E1101


if __name__ == "__main__":
    main()