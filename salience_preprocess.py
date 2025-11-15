from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from trans import *
from salience_trans import *
import main_salience

# Step 1: Load raw datasets
    # assumes data is stored in data/faces_cleaned/faces/{num_identities}_identities/{split}/{identity}/img#.jpg
# Step 2: Initialize Pipeline
# Step 3: Transform images using pipeline
# Step 4: Save transformed images
    #create save dir 
    #create subdir for each person
    #save each transformed image in subdir w/ img#_proc#

num_identities = 32
# for faces_data in ['updated', 'cnn']: # create LP dataset and CNN dataset
for split in ['test', 'train', 'valid']: 
    split_dir = Path(f'data/faces_cleaned/faces_cleaned/{num_identities}_identities/{split}') # directory w/ subdirectories (AdamRippon,Alicia,...) with images num.jpg 
    updated_save_dir = Path(f'processed_data/salience1/updated_faces/{num_identities}_identities/{split}')
    cnn_save_dir = Path(f'processed_data/salience1/cnn_faces/{num_identities}_identities/{split}')
    
    updated_save_dir.mkdir(parents=True, exist_ok=True)
    cnn_save_dir.mkdir(parents=True, exist_ok=True)

    # Create pipeline to transform images
    pipeline = SaliencePipeline(split, num_salient_points=64) #LP and CNN

    # Get directory for the current split
    for label_dir in split_dir.iterdir(): # folder of images for each person
        #Make subdir for each person in save_dir 
        Path(f'processed_data/salience1/updated_faces/{num_identities}_identities/{split}/{label_dir.name}').mkdir(parents=True, exist_ok=True)
        Path(f'processed_data/salience1/cnn_faces/{num_identities}_identities/{split}/{label_dir.name}').mkdir(parents=True, exist_ok=True)
        print('label_dir: ', label_dir)
        for img_path in label_dir.iterdir(): # image paths for each person
            img_pil = Image.open(img_path).convert("RGB") # load image as PIL object
            img_tensor = TF.to_tensor(img_pil).unsqueeze(0) # convert to torch.tensor of shape (C,H,W) -> unsqueeze to (1,C,H,W), since pipeline expects batched imgs
            transformed_imgs_lp, transformed_imgs_cnn = pipeline(img_tensor) # torch.tensor(B,N,C,H,W)
            # if faces_data == 'cnn':
            #     transformed_imgs = transformed_imgs.permute(1,0,2,3,4)
            
            for n, transformed_img_tensor in enumerate(transformed_imgs_lp[0]): #torch.tensor (C,H,W). note: batch_size=1 from unsqueeze above.
                transformed_img_pil = TF.to_pil_image(transformed_img_tensor.clamp(0, 1)) #convert tensor to PIL Image
                file_path = f'processed_data/salience1/updated_faces/{num_identities}_identities/{split}/{label_dir.name}/{img_path.stem}_proc{n}.png'
                transformed_img_pil.save(file_path) #save PIL image
            for n, transformed_img_tensor in enumerate(transformed_imgs_cnn[0]): #torch.tensor (C,H,W). note: batch_size=1 from unsqueeze above.
                transformed_img_pil = TF.to_pil_image(transformed_img_tensor.clamp(0, 1)) #convert tensor to PIL Image
                file_path = f'processed_data/salience1/cnn_faces/{num_identities}_identities/{split}/{label_dir.name}/{img_path.stem}_proc{n}.png'
                transformed_img_pil.save(file_path) #save PIL image

# start LP test when done
# for i in range(5):
#     print(f"starting LP {i}...")
#     main_salience.main(lp=True, dataset_name="salience", faces_data="updated") 
# for i in range(5):
#     print(f"starting CNN {i}...")
#     main_salience.main(lp=True, dataset_name="salience", faces_data="cnn") 