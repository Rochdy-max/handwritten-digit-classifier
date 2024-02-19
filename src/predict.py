import torch
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from classifier import HandwrittenDigitClassifier
from load_data import HandwrittenDigitMNIST

if __name__ == '__main__':
    model_path = 'model/hwd_classifier.pth'
    # Initialize transforms
    images_transform = Lambda(lambda image: image.divide(255).view(1, 28, 28))

    # Set test data files' paths
    test_images_filepath = "data/t10k-images-idx3-ubyte.gz"
    test_labels_filepath = "data/t10k-labels-idx1-ubyte.gz"

    # Create test dataset
    test_data = HandwrittenDigitMNIST(
        test_images_filepath,
        test_labels_filepath,
        images_transform)

    # Load model
    model = HandwrittenDigitClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('Model loaded')

    # Model summary
    print('\nModel summary')
    summary(model, input_size=(1, 1, 28, 28))

    # Make single prediction
    print('\nMaking single prediction')
    input_index = torch.randint(0, len(test_data), (1,)).item()
    single_image = test_data[input_index][0].unsqueeze(dim=0)
    single_image_label = test_data[input_index][1]
    with torch.no_grad():
        logits = model(single_image)
        single_pred = torch.nn.functional.softmax(logits, dim=1).argmax(dim=1)
    print(f'Predicted={single_pred}\nExpected={single_image_label}')
        
    # Create test dataloader
    batch_size = 8
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

    # Make multiple prediction
    print('\nMaking multiple prediction at once')
    test_features, test_labels = next(iter(test_dataloader))
    test_labels = test_labels.flatten()
    with torch.no_grad():
        logits = model(test_features)
        multi_pred = torch.nn.functional.softmax(logits, dim=1).argmax(dim=1)
    print(f'Predicted={multi_pred}\nExpected={test_labels}')

    # Graphical displaying of all predictions
    response = input('\nDo you want a graphical presentation of predictions [(y,yes,1)|*]:')
    if response not in ['y', 'yes', '1']:
        exit()
    all_inputs = torch.cat((test_data[input_index][0].unsqueeze(dim=0), test_features))
    all_preds = torch.cat((single_pred, multi_pred))
    all_expected = torch.cat((single_image_label, test_labels))
    nrows, ncols = 3, 3
    fig = plt.figure(figsize=(10, 6))
    for i in range(nrows * ncols):
        image = all_inputs[i]
        pred = all_preds[i]
        expected = all_expected[i]
        message = f'Predicted={pred}, Expected={expected}'
        fig.add_subplot(nrows, ncols, i + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(message)
    plt.show()
