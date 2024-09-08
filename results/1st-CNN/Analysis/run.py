import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from Models import *
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
# #------------TESTING THE MODEL------------------
test_data_dir = "D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/dataset/test_set"
# load the model
model = ImageSentimentClassification()
model.load_state_dict(torch.load('D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/results/1st-CNN/BestImageSentimentModel.pt'))
device = torch.cuda.current_device()
model.to(device)

# Set the path to the test dataset folders
test_folder = "D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/dataset/test_set"
negative_folder = os.path.join(test_folder, "Negative")
neutral_folder = os.path.join(test_folder, "Neutral")
positive_folder = os.path.join(test_folder, "Positive")

# Get the image paths from each folder
negative_images = [os.path.join(negative_folder, filename) for filename in os.listdir(negative_folder)]
neutral_images = [os.path.join(neutral_folder, filename) for filename in os.listdir(neutral_folder)]
positive_images = [os.path.join(positive_folder, filename) for filename in os.listdir(positive_folder)]

# Predict captions for each class and perform sentiment analysis
predicted_captions = []
sentiments = []
# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])
start_time = time.time()

index = 0
# Predict captions and analyze sentiment for negative images
for image in negative_images:
    if(index%100 == 0):
        print("Negative images processed {0}/{1}".format(index, len(negative_images)))
    # Preprocess the image
    image_load = Image.open(image).convert('RGB')
    input_tensor = preprocess(image_load).unsqueeze(0).to(device)

    # Compute the prediction
    model.eval()
    features = model.extract_features(input_tensor)
    output = model.classifier(features)
    print(output)
    predicted_class = torch.argmax(output)
    if predicted_class == 0:
        print("Negative image predicted as negative")
    sentiment = predicted_class - 1
    sentiments.append(sentiment)
    index += 1

index = 0
for image in neutral_images:
    if(index%100 == 0):
        print("Neutral images processed {0}/{1}".format(index, len(neutral_images)))
    # Preprocess the image
    image_load = Image.open(image).convert('RGB')
    input_tensor = preprocess(image_load).unsqueeze(0).to(device)

    # Compute the prediction
    model.eval()
    features = model.extract_features(input_tensor)
    output = model.classifier(features)
    print(output)
    predicted_class = torch.argmax(output)
    if predicted_class == 0:
        print("Negative image predicted as negative")
    sentiment = predicted_class - 1
    sentiments.append(sentiment)
    index += 1

index = 0
for image in positive_images:
    if(index%100 == 0):
        print("Positive images processed {0}/{1}".format(index, len(positive_images)))
    # Preprocess the image
    image_load = Image.open(image).convert('RGB')
    input_tensor = preprocess(image_load).unsqueeze(0).to(device)

    # Compute the prediction
    model.eval()
    features = model.extract_features(input_tensor)
    output = model.classifier(features)
    predicted_class = torch.argmax(output)
    print(output)
    if predicted_class == 0:
        print("Negative image predicted as negative")
    sentiment = predicted_class - 1
    sentiments.append(sentiment)
    index += 1

#end time
end_time = time.time()

ground_truth_sentiments = [-1.0] * len(negative_images) + [0.0] * len(neutral_images) + [1.0] * len(positive_images)

correct_predictions = 0
total_predictions = len(sentiments)

for predicted_sentiment, ground_truth_sentiment in zip(sentiments, ground_truth_sentiments):
    if predicted_sentiment > 0:
        predicted_label = 1 
    elif predicted_sentiment < 0:
        predicted_label = -1
    else:
        predicted_label = 0

    if predicted_label == ground_truth_sentiment:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions

print("Accuracy:", accuracy)
print("Time taken: {0}".format(end_time - start_time))
#save value
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))
    #time taken
    f.write("\nTime taken: {0}".format(end_time - start_time))

#draw graphs showing the distribution of sentiments for each class
import matplotlib.pyplot as plt
import numpy as np

negative_sentiments = [sentiment.cpu().item() for sentiment in sentiments if sentiment < 0]
neutral_sentiments = [sentiment.cpu().item() for sentiment in sentiments if sentiment == 0]
positive_sentiments = [sentiment.cpu().item() for sentiment in sentiments if sentiment > 0]

plt.hist(negative_sentiments, bins=20, color="red", alpha=0.5, label="Negative")
plt.hist(neutral_sentiments, bins=20, color="blue", alpha=0.5, label="Neutral")
plt.hist(positive_sentiments, bins=20, color="green", alpha=0.5, label="Positive")
plt.legend(loc="upper left")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Frequency")
#save
plt.savefig("sentiment_distribution.png")
plt.show()

# show the distribution of sentiments for each class in comparison to the ground truth
plt.hist(negative_sentiments, bins=20, color="red", alpha=0.5, label="Negative")
plt.hist(neutral_sentiments, bins=20, color="blue", alpha=0.5, label="Neutral")
plt.hist(positive_sentiments, bins=20, color="green", alpha=0.5, label="Positive")
plt.hist(ground_truth_sentiments, bins=20, color="black", alpha=0.5, label="Ground Truth")
plt.legend(loc="upper left")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Frequency")
#save
plt.savefig("sentiment_distribution.png")
plt.show()
