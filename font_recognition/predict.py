import os
import torch
from PIL import Image
from torchvision import transforms

# Define the paths
prediction_folder_path = '../dataset/predict'
model_path = '../dataset/models/CNN_en_3.pth'
fonts_path = '../dataset/fonts'

# Load fonts, ignore .DS_Store
fonts_ls = [font for font in os.listdir(fonts_path) if font != '.DS_Store']

# Load the model
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
])

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    print(f'The predicted font for {image_name} is {predicted_label}.')
