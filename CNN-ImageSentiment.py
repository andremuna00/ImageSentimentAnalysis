import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from Models import *
from Auxmethods import *

#------------PREPARING THE DATASET------------------
data_dir = "c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/training_set"
test_data_dir = "c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/test_set"

# load the train and test data
dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))
test_dataset = ImageFolder(test_data_dir,transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

# verify data integrity
check_dataset(dataset)
check_dataset(test_dataset)

# print some infos to understand if all went correnct
img, label = dataset[0]
print(img.shape,label) # torch.Size([3, 150, 150]) 0
print("Follwing classes are there : \n",dataset.classes)
# display the first image in the dataset
# display_img(*dataset[0])

#------------SPLITTING DATA AND PREPARE BATCHES------------------
batch_size = 173 # choose batch size here
val_size = 346 # choose validation size here
train_size = len(dataset) - val_size 

train_data,val_data = random_split(dataset,[train_size,val_size])

print(f"Length of Train Data : {len(train_data)}")#Length of Train Data : 12034
print(f"Length of Validation Data : {len(val_data)}")#Length of Validation Data : 2000

# load the train and validation into batches.
train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 0, pin_memory = True)
val_dl = DataLoader(val_data, batch_size*2, num_workers = 0, pin_memory = True)

# show_batch(train_dl)

#------------TRAINING THE MODEL------------------
# setup the model
num_epochs = 1
opt_func = torch.optim.Adam
lr = 0.001
model = ImageSentimentClassification()

# fitting the model on training data and record the result after each epoch
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

#------------EVALUATING THE MODEL------------------
visualize_filters(model)

plot_accuracies(history)
plot_losses(history)

# Visualize the output of each layer by testing the model on the first 40 images of the test dataset
imgs = []
outputs = []
outputs2 = []
outputs3 = []
outputs4 = []
outputs5 = []
outputs6 = []
for i in range(0,40):
    img, label = dataset[i]
    imgs.append(img.permute(1,2,0))
    output = model.first_conv(img)
    outputs.append(output[0].detach().numpy())
    output = model.second_conv(output)
    outputs2.append(output[0].detach().numpy())
    output = model.third_conv(output)
    outputs3.append(output[0].detach().numpy())
    output = model.next_conv(output)
    outputs4.append(output[0].detach().numpy())
    output = model.last_conv(output)
    outputs5.append(output[0].detach().numpy())


visualize_images(imgs)
visualize_images(outputs)
visualize_images(outputs2)
visualize_images(outputs3)
visualize_images(outputs4)
visualize_images(outputs5)