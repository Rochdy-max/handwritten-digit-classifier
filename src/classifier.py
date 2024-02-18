from torch import nn, Tensor

class HandwrittenDigitClassifier(nn.Module):
    def __init__(self):
        # Call super constructor
        super(HandwrittenDigitClassifier, self).__init__()        

        # Constants for input and ouput layers's size
        image_width = 28
        image_height = 28
        nb_classes = 10

        # Create layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x: Tensor):
        # Pass X through layers
        x = nn.functional.relu(self.pool(self.conv1(x)))
        x = nn.functional.relu(self.pool(self.conv2(x)))
        x = nn.functional.relu(self.conv3(x))
        x = self.flat(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.log_softmax(self.fc2(x), dim=1)
        return x
