import os
import shutil
import pandas as pd
import requests

path = 'D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/dataset/flickr_images'
dataset_path = 'D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/dataset/flickr3_icassp2016_dataset.csv'


# dataset = pd.read_csv(dataset_path, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

# if not os.path.exists(path):
#     os.makedirs(path)

# for index, row in dataset.iterrows():
#     image_id = row["ImageID"]
#     src_path = os.path.join(path, (str(image_id)+".jpg"))
#     if os.path.exists(src_path):
#         if index%1000 == 0:
#             print(index, image_id)
#         continue

#     print(index, image_id)
#     image_id = row["ImageID"]
#     url = f"https://api.flickr.com/services/rest/?method=flickr.photos.getSizes&api_key={api_key}&photo_id={image_id}&format=json&nojsoncallback=1"
#     # Send a GET request to the URL
#     response = requests.get(url)
#     if response.status_code != 200:
#         print(response)
#         continue
#     # Parse the JSON response to get the URL of the original image
#     data = response.json()

#     # some images are not available anymore
#     if data['stat'] != 'ok':
#         continue

#     original_url = data['sizes']['size'][-1]['source']
    
#     # Download the image and save it to the folder
#     response = requests.get(original_url)
#     with open(f"D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/dataset/flickr_images/{image_id}.jpg", "wb") as f:
#         f.write(response.content)

dataset = pd.read_csv(dataset_path, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

# foreach row get the flickr image from the id and copy it in the training_set folder with the sentiment as folder name
for index, row in dataset.iterrows():
    # Get the sentiment from the annotations file
    image_id = row['ImageID']

    if str(image_id) == '117159196' or str(image_id) == '117161695':
        continue

    print(index, image_id)

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
    filename = str(image_id)+'.jpg'

    # Set the source and destination paths
    src_path = os.path.join(path, filename)
    #select randomly is the image is used for training or testing
    if (index % 10) == 0:
        dst_path = os.path.join('D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/dataset/test_set/'+sentiment, filename)
    else:
        dst_path = os.path.join('D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/dataset/training_set/'+sentiment, filename)

    if not os.path.exists(src_path):
        print(index, image_id, "Not found")
        continue
    if os.path.exists(os.path.join('D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/dataset/test_set/'+sentiment, filename)) or os.path.exists(os.path.join('D:/UniveSources/MachineLearning-IVU/ImageSentimentAnalysis/dataset/training_set/'+sentiment, filename)):
            continue
    #create the destination folder if it doesn't exist
    if not os.path.exists(os.path.dirname(dst_path)):
        os.makedirs(os.path.dirname(dst_path))

    # Move the file to the destination folder
    shutil.copy(src_path, dst_path)