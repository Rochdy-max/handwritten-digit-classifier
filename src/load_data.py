import gzip
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import struct

images_transform = ToTensor
labels_transform = one_hot

def visualize_sample(images: torch.Tensor, labels: torch.Tensor, rows: int, cols: int):
    figure = plt.figure(figsize=(rows * 2, cols * 2))
    for i in range(1, rows * cols + 1):
        sample_idx = torch.randint(len(images), size=(1,)).item()
        img = images[sample_idx].squeeze()
        label = labels[sample_idx].item()
        img_title = f'Written digit : {label}'
        figure.add_subplot(rows, cols, i)
        plt.axis('off')
        plt.imshow(img, cmap=plt.cm.gray)
        plt.title(img_title)
    plt.show()

def read_images(path: str, num_items: int | None = None):
    expected_magic = 2051
    metadata_size = 16
    with gzip.open(path, 'rb') as file:
        magic, size, rows, cols = struct.unpack('>IIII', file.read(metadata_size))
        print(magic, size, rows, cols)
        if magic != expected_magic:
            raise ValueError(f'Magic number mismatch, expected {expected_magic}, got {magic}')
        data = torch.frombuffer(file.read(), dtype=torch.uint8)

    if num_items and size > num_items:
        size = num_items

    images = torch.zeros((size, rows, cols))
    for i in range(size):
        if i > size:
            break
        image = data[i * rows * cols:(i + 1) * rows * cols] / 255.0
        images[i][:] = image.reshape(rows, cols)
    return images

def read_labels(path: str, num_items: int | None = None):
    expected_magic = 2049
    metadata_size = 8
    with gzip.open(path, 'rb') as file:
        magic, size = struct.unpack('>II', file.read(metadata_size))
        print(magic, size)
        if magic != expected_magic:
            raise ValueError(f'Magic number mismatch, expected {expected_magic}, got {magic}')

        if num_items and size > num_items:
            size = num_items

        data = torch.frombuffer(file.read(size), dtype=torch.uint8)
    return data

def load_handwritten_mnist_data():
    num_train = 60_000
    num_test = 10_000
    training_images = read_images("data/train-images-idx3-ubyte.gz", num_train)
    training_labels = read_labels("data/train-labels-idx1-ubyte.gz", num_train)
    test_images = read_images("data/t10k-images-idx3-ubyte.gz", num_test)
    test_labels = read_labels("data/t10k-labels-idx1-ubyte.gz", num_test)
    
    return (training_images, training_labels), (test_images, test_labels)

if __name__ == "__main__":
    (t1, l1), (t2, l2) = load_handwritten_mnist_data()

    print(f"{len(t1) = }, {len(l1) = }")
    print(f"{len(t2) = }, {len(l2) = }")

    visualize_sample(t1, l1, 3, 3)

