import os
import shutil
import pandas as pd

# Load the annotations file
annotations = pd.read_csv('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/annotations.csv', sep=r'\s*;\s*', header=0, encoding='ascii', engine='python')

# Iterate over each row in the annotations file
for index, row in annotations.iterrows():
    # Get the sentiment from the annotations file
    sentiment = ''
    if row['A1.Q3.1'] == 1:
        sentiment = 'Joy'
    if row['A1.Q3.2'] == 1:
        sentiment = 'Sadness'
    if row['A1.Q3.3'] == 1:
        sentiment = 'Fear'
    if row['A1.Q3.4'] == 1:
        sentiment = 'Disgust'
    if row['A1.Q3.5'] == 1:
        sentiment = 'Anger'
    if row['A1.Q3.6'] == 1:
        sentiment = 'Surprise'
    if row['A1.Q3.7'] == 1:
        sentiment = 'Neutral'

    # Get the filename from the annotations file
    filename = row['filename']

    # Set the source and destination paths
    src_path = os.path.join('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/images', filename)
    #select randomly is the image is used for training or testing
    if (index % 10) == 0:
        dst_path = os.path.join('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/test_set/'+sentiment, filename)
    else:
        dst_path = os.path.join('c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/training_set/'+sentiment, filename)

    #create the destination folder if it doesn't exist
    if not os.path.exists(os.path.dirname(dst_path)):
        os.makedirs(os.path.dirname(dst_path))

    # Move the file to the destination folder
    shutil.move(src_path, dst_path)