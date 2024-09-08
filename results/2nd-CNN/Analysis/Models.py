import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model superclass with common methods
class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        images = images.cuda()
        labels = labels.cuda()
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        images = images.cuda()
        labels = labels.cuda()
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
# Define the model where there is the layers description
class ImageSentimentClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.next_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.network = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(175232, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, xb):
        x = self.first_conv(xb)
        x = self.second_conv(x)
        x = self.next_conv(x)
        return self.network(x)

    def extract_features(self, xb):
        x = self.first_conv(xb)
        x = self.second_conv(x)
        x = self.next_conv(x)
        return x
    def classifier(self, xb):
        return self.network(xb)
    
# compute accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# compute the loss on the validation set
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = []
    count = 0
    for batch in val_loader:
        count += 1     
        print("Validation Batch : ",count)
        print("Validation Batch Size : ",batch[0].shape)
        outputs.append(model.validation_step(batch))
    return model.validation_epoch_end(outputs)

# perform the training phase
def fit(epochs, lr, model, train_loader, val_dl, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):

        print("Epoch : ",epoch)

        model.train()
        train_losses = []
        
        count = 0
        for batch in train_loader:
            count += 1        
            print("Batch : ",count)
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_dl)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history