# perform image captioning and then perform sentiment analysis on the caption. Check the data using the test dataset.

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
import torch
from PIL import Image
from textblob import TextBlob

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
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


x = predict_step(['c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/images_backup/fa470418-5c2f-44de-b722-98e2a1dc2fbe.jpg'])[0] # ['a woman in a hospital bed with a woman in a hospital bed']
print(x)
blob = TextBlob(x)

print(blob.sentiment)

#image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
#print(image_to_text("c:/Sources/Unive/MachineLearning-IVU/ImageSentimentAnalysis/dataset/images_backup/fe4bd6a0-5176-4927-b774-67c253998098.jpg").generated_text)#