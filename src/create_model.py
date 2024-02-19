import torch
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from load_data import HandwrittenDigitMNIST
from classifier import HandwrittenDigitClassifier

def plot_training_hist(hist):
    accuracy_keys = ['Train accuracy', 'Validation accuracy']
    loss_keys = ['Train loss', 'Validation loss']
    panels = ['Accuracy results', 'Loss results']
    _, ax = plt.subplots(1, 2, figsize=(10, 6))

    for key in accuracy_keys:
        ax[0].plot(hist[key], label=key)
        ax[0].set_title(panels[0])
        ax[0].legend()
    for key in loss_keys:
        ax[1].plot(hist[key], label=key)
        ax[1].set_title(panels[1])
        ax[1].legend()
    plt.show()

if __name__ == "__main__":
    model_path = 'model/hwd_classifier.pth'
    # Initialize transforms
    images_transform = Lambda(lambda image: image.reshape(1, 28, 28).divide(255))
    # labels_transform = Lambda(lambda label: torch.zeros(10).put(label.type(torch.int64), torch.tensor(1.0)))

    # Set training and test files' paths
    training_images_filepath = "data/train-images-idx3-ubyte.gz"
    training_labels_filepath = "data/train-labels-idx1-ubyte.gz"
    test_images_filepath = "data/t10k-images-idx3-ubyte.gz"
    test_labels_filepath = "data/t10k-labels-idx1-ubyte.gz"

    # Create training dataset
    training_data = HandwrittenDigitMNIST(
        training_images_filepath,
        training_labels_filepath,
        images_transform)
    # Create test dataset
    test_data = HandwrittenDigitMNIST(
        test_images_filepath,
        test_labels_filepath,
        images_transform)

    # Create train and test dataloader
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size)

    # Create model
    model = HandwrittenDigitClassifier()

    # Model summary
    summary(model, input_size=(1, 1, 28, 28))
    
    # Set training parameters
    learning_rate = 0.01
    epochs = 5
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    # Fit model to training data
    # res = {'Train accuracy': torch.randint(0, 10, (5,)), 'Validation accuracy': torch.randint(0, 10, (5,)), 'Train loss': torch.randint(0, 10, (5,)), 'Validation loss': torch.randint(0, 10, (5,))}
    res = model.fit(train_dataloader, test_dataloader, epochs, optimizer, loss_fn)

    # Plot training results
    plot_training_hist(res)

    # Save model
    torch.save(model.state_dict(), model_path)
