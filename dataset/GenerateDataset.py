# import os
# import shutil
# import pandas as pd

# # Load the annotations file
# annotations = pd.read_csv('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/annotations.csv', sep=r'\s*;\s*', header=0, encoding='ascii', engine='python')

# # Iterate over each row in the annotations file
# for index, row in annotations.iterrows():
#     # Get the sentiment from the annotations file
#     sentiment = ''
#     if row['A1.Q3.1'] == 1:
#         sentiment = 'Joy'
#     if row['A1.Q3.2'] == 1:
#         sentiment = 'Sadness'
#     if row['A1.Q3.3'] == 1:
#         sentiment = 'Fear'
#     if row['A1.Q3.4'] == 1:
#         sentiment = 'Disgust'
#     if row['A1.Q3.5'] == 1:
#         sentiment = 'Anger'
#     if row['A1.Q3.6'] == 1:
#         sentiment = 'Surprise'
#     if row['A1.Q3.7'] == 1:
#         sentiment = 'Neutral'

#     # Get the filename from the annotations file
#     filename = row['filename']

#     # Set the source and destination paths
#     src_path = os.path.join('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/images', filename)
#     #select randomly is the image is used for training or testing
#     if (index % 10) == 0:
#         dst_path = os.path.join('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/test_set/'+sentiment, filename)
#     else:
#         dst_path = os.path.join('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/training_set/'+sentiment, filename)

#     #create the destination folder if it doesn't exist
#     if not os.path.exists(os.path.dirname(dst_path)):
#         os.makedirs(os.path.dirname(dst_path))

#     # Move the file to the destination folder
#     shutil.move(src_path, dst_path)


import os
import shutil
import pandas as pd
import requests

api_key = "cdb126f4295bf1fa84cc33c7543963e2"
secret = "d5a8a6038fbb76b9"
# Load the annotations file
dataset = pd.read_csv('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/flickr1_icassp2016_dataset.csv', sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

# Create a folder to save the images
if not os.path.exists('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/flickr_images'):
    os.makedirs('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/flickr_images')

for index, row in dataset.iterrows():
    image_id = row["ImageID"]
    print(index, image_id)
    src_path = os.path.join('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/flickr_images', (str(image_id)+".jpg"))
    if os.path.exists(src_path):
        continue
    image_id = row["ImageID"]
    url = f"https://api.flickr.com/services/rest/?method=flickr.photos.getSizes&api_key={api_key}&photo_id={image_id}&format=json&nojsoncallback=1"
    # Send a GET request to the URL
    response = requests.get(url)
    if response.status_code != 200:
        print(response)
        continue
    # Parse the JSON response to get the URL of the original image
    data = response.json()
    if data['stat'] != 'ok':
        continue

    original_url = data['sizes']['size'][-1]['source']
    
    # Download the image and save it to the folder
    response = requests.get(original_url)
    with open(f"c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/flickr_images/{image_id}.jpg", "wb") as f:
        f.write(response.content)
#foreach row get the flickr image from the id and copy it in the training_set folder with the sentiment as folder name
for index, row in dataset.iterrows():
    # Get the sentiment from the annotations file
    Positive = row['Num_of_Positive']
    Negative = row['Num_of_Negative']
    Neutral = row['Num_of_Neutral']

    sentiment = ''
    if Positive > Negative and Positive > Neutral:
        sentiment = 'Positive'
    elif Negative > Positive and Negative > Neutral:
        sentiment = 'Negative'
    elif Neutral > Positive and Neutral > Negative:
        sentiment = 'Neutral'
    else:
        sentiment = 'Neutral'
    
    # Get the filename from the annotations file
    filename = row['ImageID']+'.jpg'

    # Set the source and destination paths
    src_path = os.path.join('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/flickr_images', filename)
    #select randomly is the image is used for training or testing
    if (index % 10) == 0:
        dst_path = os.path.join('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/test_set/'+sentiment, filename)
    else:
        dst_path = os.path.join('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/training_set/'+sentiment, filename)

    #create the destination folder if it doesn't exist
    if not os.path.exists(os.path.dirname(dst_path)):
        os.makedirs(os.path.dirname(dst_path))

    # Move the file to the destination folder
    shutil.copy(src_path, dst_path)