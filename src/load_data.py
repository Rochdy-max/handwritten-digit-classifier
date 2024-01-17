import torch
from torch.nn.functional import one_hot
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

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
