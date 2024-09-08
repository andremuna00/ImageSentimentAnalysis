# perform image captioning and then perform sentiment analysis on the caption. Check the data using the test dataset.

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
import torch
from PIL import Image
from textblob import TextBlob
import os

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 200
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

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

#start time
import time
start_time = time.time()

index = 0
# Predict captions and analyze sentiment for negative images
for image in negative_images:
    if(index%100 == 0):
        print("Negative images processed {0}/{1}".format(index, len(negative_images)))
    predicted_negative_captions = predict_step([image])
    predicted_captions.extend(predicted_negative_captions)
    for caption in predicted_negative_captions:
        blob = TextBlob(caption)
        sentiment = blob.sentiment.polarity
        sentiments.append(sentiment)
    index += 1

index = 0
for image in neutral_images:
    if(index%100 == 0):
        print("Neutral images processed {0}/{1}".format(index, len(neutral_images)))
    predicted_neutral_captions = predict_step([image])
    predicted_captions.extend(predicted_neutral_captions)
    for caption in predicted_neutral_captions:
        blob = TextBlob(caption)
        sentiment = blob.sentiment.polarity
        sentiments.append(sentiment)
    index += 1

index = 0
for image in positive_images:
    if(index%100 == 0):
        print("Positive images processed {0}/{1}".format(index, len(positive_images)))
    predicted_positive_captions = predict_step([image])
    predicted_captions.extend(predicted_positive_captions)
    for caption in predicted_positive_captions:
        blob = TextBlob(caption)
        sentiment = blob.sentiment.polarity
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
#save value
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))
    #time taken
    f.write("\nTime taken: {0}".format(end_time - start_time))

#draw graphs showing the distribution of sentiments for each class
import matplotlib.pyplot as plt
import numpy as np

negative_sentiments = [sentiment for sentiment in sentiments if sentiment < 0]
neutral_sentiments = [sentiment for sentiment in sentiments if sentiment == 0]
positive_sentiments = [sentiment for sentiment in sentiments if sentiment > 0]

plt.hist(negative_sentiments, bins=20, color="red", alpha=0.5, label="Negative")
plt.hist(neutral_sentiments, bins=20, color="blue", alpha=0.5, label="Neutral")
plt.hist(positive_sentiments, bins=20, color="green", alpha=0.5, label="Positive")
plt.legend(loc="upper right")
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
plt.legend(loc="upper right")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Frequency")
#save
plt.savefig("sentiment_distribution.png")
plt.show()
