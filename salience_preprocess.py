from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from trans import *
from salience_trans import *
import main_salience
from Datasets import *

# Step 1: Load raw datasets
    # assumes data is stored in data/faces_cleaned/faces/{num_identities}_identities/{split}/{identity}/img#.jpg
# Step 2: Initialize Pipeline
# Step 3: Transform images using pipeline
# Step 4: Save transformed images
    #create save dir 
    #create subdir for each person
    #save each transformed image in subdir w/ img#_proc#

num_identities = 128
num_fixations = 16
root = 'dogs1k' # source
dataset_name = 'dogs1k'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for split in ['test', 'train', 'valid']: 
    split_dir = Path(f'data/{dataset_name}/{dataset_name}/{num_identities}_identities/{split}') # directory w/ subdirectories (AdamRippon,Alicia,...) with images num.jpg 
    if split == 'test':
        split_dir = Path(f'data/{dataset_name}/{dataset_name}/{num_identities}_identities/valid')
    updated_save_dir = Path(f'processed_data/{root}/updated_faces/{num_identities}_identities/{split}')
    cnn_save_dir = Path(f'processed_data/{root}/cnn_faces/{num_identities}_identities/{split}')
    
    updated_save_dir.mkdir(parents=True, exist_ok=True)
    cnn_save_dir.mkdir(parents=True, exist_ok=True)

    # Create pipeline to transform images
    pipeline = SaliencePipeline(split, num_salient_points=num_fixations, device=device).to(device) #LP and CNN

    # Get directory for the current split
    for label_dir in split_dir.iterdir(): # folder of images for each person
        #Make subdir for each person in save_dir 
        Path(f'processed_data/{root}/updated_faces/{num_identities}_identities/{split}/{label_dir.name}').mkdir(parents=True, exist_ok=True)
        Path(f'processed_data/{root}/cnn_faces/{num_identities}_identities/{split}/{label_dir.name}').mkdir(parents=True, exist_ok=True)
        print('label_dir: ', label_dir)
        for img_path in label_dir.iterdir(): # image paths for each person
            # t0 = time.perf_counter()
            img_pil = Image.open(img_path).convert("RGB") # load image as PIL object
            img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device) # convert to torch.tensor of shape (C,H,W) -> unsqueeze to (1,C,H,W), since pipeline expects batched imgs
            
            # t1 = time.perf_counter()

            transformed_imgs_lp, transformed_imgs_cnn = pipeline(img_tensor) # torch.tensor(B,N,C,H,W)
            # t2 = time.perf_counter()
            for n, transformed_img_tensor in enumerate(transformed_imgs_lp[0]): #torch.tensor (C,H,W). note: batch_size=1 from unsqueeze above.
                transformed_img_pil = TF.to_pil_image(transformed_img_tensor.clamp(0, 1))#.cpu()) #convert tensor to PIL Image
                file_path = f'processed_data/{root}/updated_faces/{num_identities}_identities/{split}/{label_dir.name}/{img_path.stem}_proc{n}.png'
                transformed_img_pil.save(file_path) #save PIL image
            for n, transformed_img_tensor in enumerate(transformed_imgs_cnn[0]): #torch.tensor (C,H,W). note: batch_size=1 from unsqueeze above.
                transformed_img_pil = TF.to_pil_image(transformed_img_tensor.clamp(0, 1))#.cpu()) #convert tensor to PIL Image
                file_path = f'processed_data/{root}/cnn_faces/{num_identities}_identities/{split}/{label_dir.name}/{img_path.stem}_proc{n}.png'
                transformed_img_pil.save(file_path) #save PIL image

            # t3 = time.perf_counter()
            # print(
            #     f"load→gpu: {t1 - t0:.3f}s | "
            #     f"infer: {t2 - t1:.3f}s | "
            #     f"save: {t3 - t2:.3f}s"
            # )

# start LP test when done
# for i in range(5):
#     print(f"starting LP {i}...")
#     main_salience.main(lp=True, dataset_name=root) 
# for i in range(5):
#     print(f"starting CNN {i}...")
#     main_salience.main(lp=False, dataset_name=root) 


# dataset_name = 'faces'

# all_datasets = {
#     ident: { split: load_dataset(dataset_name, ident, split)
#             for split in splits }
#     for ident in identity_counts
# }