from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import tqdm
from PIL import ImageFile

# show a batch of images
def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        plt.show()
        break

# visualize a set of images in a grid
def visualize_images(imgs):
    fig, axes = plt.subplots(5,8, figsize=(20,20))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# display an image and its label
def display_img(img,label):
    print(f"Label : {dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))
    plt.show()

# check if the dataset is valid
def check_dataset(dataset):
    for fn,label in tqdm.tqdm(dataset.imgs):
        try:
            im = ImageFile.Image.open(fn)
            im2 = im.convert('RGB')
        except OSError:
            print("Cannot load : {}".format(fn))

# Visualize the filters of the first convolutional layer
def visualize_filters(model):
    # Get the weights of the first convolutional layer
    first_layer = model.first_conv[0].weight.data.clone()
    # Normalize and scale the weights
    first_layer -= first_layer.min()
    first_layer /= first_layer.max()
    # Create a grid of images from the weights
    grid = make_grid(first_layer, nrow=8, padding=2)
    grid = grid.cpu()
    # Display the grid using matplotlib
    print(first_layer.shape)
    plt.figure(figsize=(10,10))
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    #save
    plt.savefig('filters.png')
    plt.show()

# Plot the history of accuracies
def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    #save
    plt.savefig('accuracy.png')

# Plot the history of losses
def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    #save
    plt.savefig('loss.png')