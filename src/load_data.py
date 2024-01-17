import gzip
import torch
from torch.nn.functional import one_hot
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import struct

images_transform = ToTensor
labels_transform = one_hot

def visualize_sample(images, labels, rows, cols):
    figure = plt.figure(figsize=(rows * 3, cols * 3))
    for i in range(1, rows * cols + 1):
        sample_idx = torch.randint(len(images), size=(1,)).item()
        img = images[sample_idx].squeeze()
        label = labels[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.axis('off')
        plt.imshow(img, cmap=plt.cm.gray)
        plt.title(label)
    plt.show()

def read_images(path, image_size, num_items):
    expected_magic = 2051
    metadata_size = 16
    with gzip.open(path, 'rb') as file:
        magic, size, rows, cols = struct.unpack('>IIII', file.read(metadata_size))
        print(magic, size, rows, cols)
        if magic != expected_magic:
            raise ValueError(f'Magic number mismatch, expected {expected_magic}, got {magic}')
        data = torch.frombuffer(file.read(), dtype=torch.uint8)

    if size > num_items:
        size = num_items

    images = torch.zeros((size, rows, cols))
    for i in range(size):
        if i > size:
            break
        img = data[i * rows * cols:(i + 1) * rows * cols] / 255.0
        images[i] = images[i].put(
            index=torch.tensor(range(rows * cols)),
            source=img)
        images[i] = images[i].reshape((rows, cols))
    return images

def load_handwritten_mnist_data():
    image_size = 28
    num_train = 60_000
    num_test = 10_000
    training_images = read_images("data/train-images-idx3-ubyte.gz", image_size, num_train)
    test_images = read_images("data/t10k-images-idx3-ubyte.gz", image_size, num_test)
    
    return training_images, test_images

if __name__ == "__main__":
    t1, t2 = load_handwritten_mnist_data()
    print(f"{len(t1) = }, {len(t1) = }")
