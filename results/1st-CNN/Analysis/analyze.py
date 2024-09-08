import torch
import torch.nn as nn
from torchsummary import summary
from Models import *

# Load the trained model
model = ImageSentimentClassification()
model.load_state_dict(torch.load("D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/results/1st-CNN/BestImageSentimentModel.pt"))

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a random input tensor and move it to the same device as the model
dummy_input = torch.randn(1, 3, 150, 150).to(device)  # Adjust input shape as per your model requirements

# Print the summary of the model
summary(model, input_size=(3, 150, 150))  # Adjust input shape as per your model requirements