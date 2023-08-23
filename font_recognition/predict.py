import os
import sys
ws_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.extend([ws_dir])

import torch
from PIL import Image
from torchvision import transforms
from prettytable import PrettyTable


# Define the paths
prediction_folder_path = os.path.join(ws_dir, r'dataset/predict')
model_path = os.path.join(ws_dir, r'dataset/models/CNN_en_5.pth')
fonts_path = os.path.join(ws_dir, r'dataset/fonts')

# Load fonts, ignore .DS_Store
fonts_ls = [font for font in os.listdir(fonts_path) if font != '.DS_Store']

print(fonts_ls)

# Load the model
model = torch.load(model_path)
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
for image_name in os.listdir(prediction_folder_path):
    # Load the image
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

    tb.add_row([image_name] + list(output.detach().cpu().numpy()[0]))
    print(f'The predicted font for {image_name} is {predicted_label}.')

print(tb)
