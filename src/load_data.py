import gzip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Lambda
import matplotlib.pyplot as plt
import struct

def visualize_sample(images: torch.Tensor, labels: torch.Tensor, rows: int, cols: int):
    # Create pyplot figure
    figure = plt.figure(figsize=(rows * 2, cols * 2))
    for i in range(1, rows * cols + 1):
        # Prepare data to display
        sample_idx = torch.randint(len(images), size=(1,)).item()
        img = images[sample_idx].squeeze()
        label = labels[sample_idx].argmax().item()
        img_title = f'Written digit : {label}'
        # Draw image and its label
        figure.add_subplot(rows, cols, i)
        plt.axis('off')
        plt.imshow(img, cmap=plt.cm.gray)
        plt.title(img_title)
    # Display window
    plt.show()

class HandwrittenDigitMNIST(Dataset):
    def __init__(self, images_filepath: str, labels_filepath: str, transform = None, target_transform = None):
        # Initialize fields from parameter values
        self.images_filepath = images_filepath
        self.labels_filepath = labels_filepath
        self.transform = transform
        self.target_transform = target_transform

        # Read dataset size, image's number of rows and cols
        with gzip.open(self.images_filepath, 'rb') as file:
            expected_magic = 2051
            images_file_metadata_size = 16
            magic, size, rows, cols = struct.unpack('>IIII', file.read(images_file_metadata_size))
            print(magic, size, rows, cols)
            if magic != expected_magic:
                raise ValueError(f'Magic number mismatch, expected {expected_magic}, got {magic}')
        # Store these values
        self.size = size
        self.rows = rows
        self.cols = cols

    def __len__(self):
        # Size of dataset
        return self.size
    
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        # Raise IndexError if index is out of range
        if index < 0 or index >= self.size:
            raise IndexError(f'index ({index}) out of range')

        # Read bytes of image and label at index
        image = self.read_image(index)
        label = self.read_label(index)

        # Create tensor from image's bytes and transform it with specified transform
        image = torch.frombuffer(image, dtype=torch.uint8)
        if self.transform:
            image = self.transform(image)

        # Create tensor from label's byte and transform it with specified target_transform
        label = torch.frombuffer(label, dtype=torch.uint8)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def read_image(self, index: int):
        # Uncompress file
        with gzip.open(self.images_filepath, 'rb') as file:
            expected_magic = 2051 # Magic number in images file
            metadata_size = 16 # Size of metadata in images file
            magic, _, _, _ = struct.unpack('>IIII', file.read(metadata_size))
            # Check magic number
            if magic != expected_magic:
                raise ValueError(f'Magic number mismatch, expected {expected_magic}, got {magic}')
            # Set file's cursor to where image's bytes start
            file.seek(index * self.rows * self.cols, whence=1)
            # Read image's bytes
            data = file.read(self.rows * self.cols)
        return data

    def read_label(self, index: int):
        # Uncompress file
        with gzip.open(self.labels_filepath, 'rb') as file:
            expected_magic = 2049 # Magic number in labels file
            metadata_size = 8 # Size of metadata in images file
            magic, _ = struct.unpack('>II', file.read(metadata_size))
            # Check magic number
            if magic != expected_magic:
                raise ValueError(f'Magic number mismatch, expected {expected_magic}, got {magic}')
            # Set file's cursor to label's byte position
            file.seek(index, whence=1)
            # Read label value
            data = file.read(1)
        return data

if __name__ == "__main__":
    # Initialize transforms
    images_transform = Lambda(lambda image: image.reshape(28, 28).divide(255))
    labels_transform = Lambda(lambda label: torch.zeros(10).put(label.type(torch.int64), torch.tensor(1.0)))

    # Set training and test files' paths
    training_images_filepath = "data/train-images-idx3-ubyte.gz"
    training_labels_filepath = "data/train-labels-idx1-ubyte.gz"
    test_images_filepath = "data/t10k-images-idx3-ubyte.gz"
    test_labels_filepath = "data/t10k-labels-idx1-ubyte.gz"

    # Create training dataset
    training_data = HandwrittenDigitMNIST(
        training_images_filepath,
        training_labels_filepath,
        images_transform,
        labels_transform)
    # Create test dataset
    test_data = HandwrittenDigitMNIST(
        test_images_filepath,
        test_labels_filepath,
        images_transform,
        labels_transform)

    # Create a DataLoader wrapping training dataset
    training_data_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    # Get a minibatch for data wrangling
    (images, labels) = next(iter(training_data_loader))
    # Visualize data in 3*3 grid
    visualize_sample(images, labels, 3, 3)
