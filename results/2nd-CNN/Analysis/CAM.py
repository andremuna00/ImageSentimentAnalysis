import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Models import ImageSentimentClassification
import glob

# Load the trained model
model = ImageSentimentClassification()
model.load_state_dict(torch.load("D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/results/2nd-CNN/BestImageSentimentModel.pt"))

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the class labels
class_labels = ['negative', 'positive']

# Function to generate CAM for an input image
def generate_cam(image_path):
    # Load and preprocess the input image
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Perform forward pass and obtain the class activation map
    model.eval()
    features = model.extract_features(input_tensor)
    output = model.classifier(features)
    predicted_class = torch.argmax(output)

    # Retrieve the feature map from the final convolutional layer
    feature_map = features.detach().cpu().numpy()

    # Retrieve the weight of the output class from the fully connected layer
    weight = model.network[-1].weight[predicted_class].detach().cpu().numpy()
    cam = np.einsum('ijkl,m->mkl', feature_map, weight)  # Element-wise multiplication and summation
    cam = np.maximum(cam, 0)  # ReLU activation
    cam = cam / np.max(cam)  # Normalize the CAM values

    # Resize the CAM to the input image size
    cam = cv2.resize(cam[0, 0], (image.size[0], image.size[1]))

    # Apply the CAM as heatmap overlay on the input image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image), 0.5, heatmap, 0.5, 0)

    # Plot the input image, CAM heatmap, and the overlaid result with the predicted class label
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image)
    ax[0].axis('off')
    ax[0].set_title('Input Image')
    

    ax[1].imshow(cam, cmap='jet')
    ax[1].axis('off')
    ax[1].set_title('Class Activation Map')

    ax[2].imshow(overlay)
    ax[2].axis('off')
    ax[2].set_title('CAM Overlay')
    ax[0].text(0.5, -0.1, 'Predicted: ' + class_labels[predicted_class-1], size=12, ha="center", transform=ax[0].transAxes)

    plt.tight_layout()
    plt.show()


# Generate CAM for images in the specified directory
image_dir = "D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/dataset/test_set/Negative"
image_paths = glob.glob(image_dir + "/*.jpg")  # Adjust the file extension if necessary

for image_path in image_paths:
    #load the csv file
    
    generate_cam(image_path)