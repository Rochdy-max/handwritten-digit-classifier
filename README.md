# handwritten-digit-classifier
PyTorch implementation for handwritten digits classifier on MNIST handwritten digit dataset.

## Requirements
Some python packages are required to run this project:
- torch
- torchinfo
- torchvision
- matplotlib

Those packages are already listed in [requirements.txt](https://github.com/Rochdy-max/handwritten-digit-classifier/blob/main/requirements.txt).
So you just have to run `pip install -r requirements.txt` from applicatio's root directory to ensure they are correctly installed.

## Usage
Two modes of usage are available with this project.

### Model Creation
To create a new model, you can simply run [classifier.py](https://github.com/Rochdy-max/handwritten-digit-classifier/blob/main/src/classifier.py).

`python src/classifier.py`

### Model Prediction
The script [predict.py](https://github.com/Rochdy-max/handwritten-digit-classifier/blob/main/src/predict.py) provides a demonstration of model prediction over test data loaded from *data* folder.

`python src/predict.py`
