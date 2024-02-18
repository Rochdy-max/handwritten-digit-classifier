from torch import nn, Tensor
import torch

class HandwrittenDigitClassifier(nn.Module):
    def __init__(self):
        # Call super constructor
        super(HandwrittenDigitClassifier, self).__init__()        
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
        x = self.fc2(x)
        return x
    
    def fit(self, train_dataloader, test_dataloader, epochs=10, optimizer=None, loss_fn=None):
        # Ensure training parameters initialization
        lr = 0.01
        optimizer = optimizer or torch.optim.SGD(self.parameters(), lr)
        loss_fn = loss_fn or nn.NLLLoss()
        # Create training results dictionary
        res = {'Train accuracy': [], 'Validation accuracy': [], 'Train loss': [], 'Validation loss': []}

        # Learning loop
        for i in range(1, epochs + 1):
            ta, tl = self.train_epoch(train_dataloader, optimizer, loss_fn)
            va, vl = self.validate_epoch(test_dataloader, loss_fn)
            print(f'Epoch {i:2}: Train acc={ta:.2f}%, Val acc={va:.2f}%, Train loss={tl:.4f}, Val loss={vl:.4f}')
            res['Train accuracy'].append(ta)
            res['Train loss'].append(tl)
            res['Validation accuracy'].append(va)
            res['Validation loss'].append(vl)
        # Return training results            
        return res
    
    def train_epoch(self, dataloader, optimizer, loss_fn):
        acc, loss = 0, 0
        data_size = len(dataloader.dataset)

        for features, labels in dataloader:
            labels = labels.view(len(labels))
            # Prediction phase
            out = self(features)
            # Learning phase
            optimizer.zero_grad()
            batch_loss = loss_fn(out, labels)
            batch_loss.backward()
            optimizer.step()
            # Update accuracy and loss
            loss += batch_loss
            pred = out.max(dim=1)[1]
            acc += (pred == labels).type(torch.float).sum()
        # Return epoch results
        return (acc.item() / data_size) * 100, loss.item() / data_size
    
    def validate_epoch(self, dataloader, loss_fn):
        acc, loss = 0, 0
        data_size = len(dataloader.dataset)

        with torch.no_grad():
            for features, labels in dataloader:
                labels = labels.view(len(labels))
                # Prediction phase
                out = self(features)
                # Update loss and accuracy
                loss += loss_fn(out, labels)
                pred = out.max(dim=1)[1]
                acc += (pred == labels).type(torch.float).sum()
        # Return epoch results
        return (acc.item() / data_size) * 100, loss.item() / data_size
