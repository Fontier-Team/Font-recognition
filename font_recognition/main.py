import os
import sys
sys.path.extend([os.getcwd()])

import json
import torch
from general_code import utils
from general_code import image_generation
import SCAE
import CNN


def main():
    utils.init_seed(42)

    current_dir = os.path.dirname(__file__)
    fonts_path = os.path.join(current_dir, r'../dataset/fonts')
    generated_image_path = os.path.join(current_dir, r'../dataset/generated_images')
    generated_label_path = os.path.join(current_dir, r'../dataset/generated_images/labels.csv')
    real_image_path = os.path.join(current_dir, r'../dataset/real_images')
    models_path = os.path.join(current_dir, r'../dataset/models')

    # make sure the path exists, if not, create it
    if not os.path.exists(generated_image_path):
        os.makedirs(generated_image_path)
    if not os.path.exists(real_image_path):
        os.makedirs(real_image_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    language = 'en'
    total_num = 1000  # total number for image across all fonts, should be much greater than batch size
    gen_batch_size = 10
    sample_batch_size = 10
    sample_num = 5
    sample_width = 105
    sample_height = 105

    train_batch_size = 50
    num_epochs = 50

    device_to_use = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fonts_ls = image_generation.get_fonts_list(fonts_path)
    num_fonts = len(fonts_ls)
    SCAE_model_path = f'{models_path}/SCAE_{language}_{num_fonts}.pth'
    CNN_model_path = f'{models_path}/CNN_{language}_{num_fonts}.pth'
    font_dict_path = f'{models_path}/font_dict_{language}_{num_fonts}.json'

    image_generation.generate_images(
        total_num=total_num,
        language=language,
        fonts_path=fonts_path,
        gen_batch_size=gen_batch_size,
        gen_image_path=generated_image_path,
        label_df_path=generated_label_path,
        need_save=True,
        need_return=False
    )
    image_generation.saved_images_sampling(
        total_num=total_num,
        img_path=generated_image_path,
        sample_path=generated_image_path,
        sample_batch_size=sample_batch_size,
        sample_num=sample_num,
        width=sample_width,
        height=sample_height,
        need_save=True,
        need_return=False,
    )

    # train SCAE (encoder trained with upervised learning)
    SCAE_train_iter, _ = SCAE.get_SCAE_dataloader_dataset(
        batch_size=train_batch_size,
        total_num=total_num,
        sample_num=sample_num,
        generated_img_path=generated_image_path,
        generated_label_path=generated_label_path,
        real_img_path=real_image_path
    )
    SCAE_net = SCAE.SCAE()
    SCAE.train_SCAE(SCAE_net, SCAE_train_iter, num_epochs, device=device_to_use)
    torch.save(SCAE_net, SCAE_model_path)

    # load SCAE model
    SCAE_net = torch.load(SCAE_model_path)

    # train CNN (classifier trained with supervised learning)
    CNN_train_iter, CNN_train_dataset = CNN.get_CNN_dataloader_dataset(
        batch_size=train_batch_size,
        total_num=total_num,
        sample_num=sample_num,
        generated_img_path=generated_image_path,
        generated_label_path=generated_label_path,
    )
    CNN_net = CNN.CNN(SCAE_net.encoder, num_fonts)
    CNN.train_CNN(CNN_net, CNN_train_iter, num_epochs=num_epochs, device=device_to_use)
    torch.save(CNN_net, CNN_model_path)

    font_dict = CNN_train_dataset.font_dict

    with open(font_dict_path, 'w') as f:
        json.dump(font_dict, f)


if __name__ == '__main__':
    main()
