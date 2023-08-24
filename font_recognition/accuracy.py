import os
import sys
ws_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.extend([ws_dir])

import torch
from PIL import Image
from torchvision import transforms
from prettytable import PrettyTable
import pandas as pd

# Load CSV of Label
labels_csv = pd.read_csv(os.path.join(ws_dir, r'dataset/generated_images/labels.csv'))
labels = labels_csv['font'] # then we got all labels from 0 to end of file

# Define the paths
prediction_folder_path = os.path.join(ws_dir, r'dataset/generated_images')
model_path = os.path.join(ws_dir, r'dataset/models/CNN_en_5.pth')
fonts_path = os.path.join(ws_dir, r'dataset/fonts')

# Load fonts, ignore .DS_Store
fonts_ls = [font for font in os.listdir(fonts_path) if font != '.DS_Store']

print(fonts_ls)

# Load the model
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
])

# Define the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tb = PrettyTable(["File Name"] + fonts_ls)

# Move the model to the device
model = model.to(device)

# Iterate over the images in the prediction folder
total_image_num = 0
correct_image_num = 0
for image_name in os.listdir(prediction_folder_path):

    # skip image without _
    if '_' not in image_name:
        continue

    # get true label
    label_index = int(image_name.split('_')[0])
    true_label = labels[label_index].split('\\')[-1]

    # Load the image)
    image_path = os.path.join(prediction_folder_path, image_name)
    image = Image.open(image_path).convert('L')

    # Apply the transformation and add an extra batch dimension
    image = transform(image).unsqueeze(0).to(device)

    # Make the prediction
    output = model(image)
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()

    # Convert the index to a label
    predicted_label = fonts_ls[predicted_index]

    # find # of correct image
    if true_label == predicted_label:
        correct_image_num+=1

    total_image_num+=1
    tb.add_row([image_name] + list(output.detach().numpy()[0]))
    print(f'The predicted font for {image_name} is {predicted_label}. {correct_image_num}')

accuracy = correct_image_num/total_image_num
print(accuracy)
print(tb)
