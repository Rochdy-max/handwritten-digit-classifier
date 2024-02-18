import torch
from torchinfo import summary
from torchvision.transforms import Lambda
from load_data import HandwrittenDigitMNIST
from classifier import HandwrittenDigitClassifier

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

    # Create model
    model = HandwrittenDigitClassifier()

    # Model summary
    summary(model, input_size=(1, 1, 28, 28))
    
    # Compile model

    # Fit model to training data

    # Evaluate model

    # Save model
