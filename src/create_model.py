import torch
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from load_data import HandwrittenDigitMNIST
from classifier import HandwrittenDigitClassifier

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
    train_dataloader = DataLoader(training_data, batch_size)
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
    res = model.fit(train_dataloader, test_dataloader, epochs, optimizer, loss_fn)

    # Plot training results

    # Save model
    # torch.save(model.parameters(), model_path)

