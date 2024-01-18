from torch import nn, Tensor

class HandwrittenDigitClassifier(nn.Module):
    def __init__(self, *hidden_layers_sizes):
        # Constants for input and ouput layers's size
        image_width = 28
        image_height = 28
        nb_categories = 10

        # Create and group layers
        layers_sizes = [image_width * image_height] + hidden_layers_sizes + [nb_categories]
        layers = [nn.Flatten()]
        for i in range(layers_sizes):
            linear_layer = nn.Linear(layers_sizes[i], layers_sizes[i + 1])
            layers.append(linear_layer)
            layers.append(nn.ReLU())

        # Create Sequential module
        self.sequential = nn.Sequential(*layers)
        
    def forward(self, X: Tensor):
        # Pass input to Sequential
        logits = self.sequential(X)
        return logits
