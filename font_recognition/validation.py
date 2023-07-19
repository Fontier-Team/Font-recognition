import os
import torch
from PIL import Image
import numpy as np

num_fonts = len(os.listdir(r'../dataset/fonts')) - 1
# model_path = r'../dataset/models/CNN_en_{}.pth'.format(num_fonts)
model_path = r'../dataset/models/CNN_en_2.pth'.format(num_fonts)
# font labels are ttf file names in ../dataset/fonts, except .DS_Store
font_labels = [font for font in os.listdir(r'../dataset/fonts') if font != '.DS_Store']
font_recognition_model = torch.load(model_path)
font_recognition_model.eval()

# load jpg images
eval_image_path = r'../dataset/validation'
eval_image_list = os.listdir(eval_image_path)
eval_image_list = [os.path.join(eval_image_path, img) for img in eval_image_list]
eval_image_list = [img for img in eval_image_list if img.endswith('.jpg')]

# load evaluation images to model
for img_path in eval_image_list:
    # weight of size [64, 1, 58, 58]
    img = Image.open(img_path).convert('L')
    img = img.resize((105, 105))
    img = np.array(img)
    img = img.reshape(1, 1, 105, 105)
    img = torch.from_numpy(img).float()
    output = font_recognition_model(img)
    print(output)
    font = torch.argmax(output, dim=1)
    label = font_labels[font]
    print(label)

