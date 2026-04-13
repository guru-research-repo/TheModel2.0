# from pathlib import Path
# from PIL import Image
# import torchvision.transforms.functional as TF
# import torch
# from trans import *
# from salience_trans import *
# import main_salience
# from Datasets import *

# # Step 1: Load raw datasets
#     # assumes data is stored in data/faces_cleaned/faces/{num_identities}_identities/{split}/{identity}/img#.jpg
# # Step 2: Initialize Pipeline
# # Step 3: Transform images using pipeline
# # Step 4: Save transformed images
#     #create save dir 
#     #create subdir for each person
#     #save each transformed image in subdir w/ img#_proc#

# num_identities = 128
# num_fixations = 32
# root = 'salience128-48lp-mag'
# # for split in ['test', 'train', 'valid']: 
# for split in ['test']: 
#     split_dir = Path(f'cleaned_faces_dataset/faces/faces/{num_identities}_identities/{split}')
#     # split_dir = Path(f'data/faces_cleaned/faces_cleaned/{num_identities}_identities/{split}') # directory w/ subdirectories (AdamRippon,Alicia,...) with images num.jpg 
#     updated_save_dir = Path(f'processed_data/{root}/updated_faces/{num_identities}_identities/{split}')
#     cnn_save_dir = Path(f'processed_data/{root}/cnn_faces/{num_identities}_identities/{split}')
    
#     updated_save_dir.mkdir(parents=True, exist_ok=True)
#     cnn_save_dir.mkdir(parents=True, exist_ok=True)

#     # Create pipeline to transform images
#     pipeline = SaliencePipeline(split, num_salient_points=num_fixations) #LP and CNN

#     # Get directory for the current split
#     for label_dir in split_dir.iterdir(): # folder of images for each person
#         #Make subdir for each person in save_dir 
#         Path(f'processed_data/{root}/updated_faces/{num_identities}_identities/{split}/{label_dir.name}').mkdir(parents=True, exist_ok=True)
#         Path(f'processed_data/{root}/cnn_faces/{num_identities}_identities/{split}/{label_dir.name}').mkdir(parents=True, exist_ok=True)
#         print('label_dir: ', label_dir)
#         for img_path in label_dir.iterdir(): # image paths for each person
#             img_pil = Image.open(img_path).convert("RGB") # load image as PIL object
#             img_tensor = TF.to_tensor(img_pil).unsqueeze(0) # convert to torch.tensor of shape (C,H,W) -> unsqueeze to (1,C,H,W), since pipeline expects batched imgs
#             transformed_imgs_lp, transformed_imgs_cnn = pipeline(img_tensor) # torch.tensor(B,N,C,H,W)
            
#             for n, transformed_img_tensor in enumerate(transformed_imgs_lp[0]): #torch.tensor (C,H,W). note: batch_size=1 from unsqueeze above.
#                 transformed_img_pil = TF.to_pil_image(transformed_img_tensor.clamp(0, 1)) #convert tensor to PIL Image
#                 file_path = f'processed_data/{root}/updated_faces/{num_identities}_identities/{split}/{label_dir.name}/{img_path.stem}_proc{n}.png'
#                 transformed_img_pil.save(file_path) #save PIL image
#             for n, transformed_img_tensor in enumerate(transformed_imgs_cnn[0]): #torch.tensor (C,H,W). note: batch_size=1 from unsqueeze above.
#                 transformed_img_pil = TF.to_pil_image(transformed_img_tensor.clamp(0, 1)) #convert tensor to PIL Image
#                 file_path = f'processed_data/{root}/cnn_faces/{num_identities}_identities/{split}/{label_dir.name}/{img_path.stem}_proc{n}.png'
#                 transformed_img_pil.save(file_path) #save PIL image

# # start LP test when done
# for i in range(5):
#     print(f"starting LP {i}...")
#     main_salience.main(lp=True, dataset_name=root) 
# # for i in range(5):
# #     print(f"starting CNN {i}...")
# #     main_salience.main(lp=False, dataset_name=root) 


# # dataset_name = 'faces'

# # all_datasets = {
# #     ident: { split: load_dataset(dataset_name, ident, split)
# #             for split in splits }
# #     for ident in identity_counts
# # }


from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from trans import *
from salience_trans import *
import main_salience
from Datasets import *
import shutil
# =========================
# Parameters
# =========================
num_identities = 4
num_fixations = 32
# root = 'salience128-48lp-mag'
root = 'salience10-48lp-mag'

# Only process the test split
# for split in ['test','valid','train']: 
for split in ['test']: 
    # split_dir = Path(f'dogs_dataset/faces/faces/{num_identities}_identities/{split}')
    split_dir = Path(f'data/dogs1K/dogs1k/{num_identities}_identities/{split}')
    
    updated_save_dir = Path(f'dogs_processed_data/{root}/updated_dogs/{num_identities}_identities/{split}')
    cnn_save_dir = Path(f'dogs_processed_data/{root}/cnn_dogs/{num_identities}_identities/{split}')
    
    # Create base directories
    updated_save_dir.mkdir(parents=True, exist_ok=True)
    cnn_save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    pipeline = SaliencePipeline(split, num_salient_points=num_fixations) # LP and CNN

    # Process each person's folder
    for label_dir in split_dir.iterdir():
        if not label_dir.is_dir() or label_dir.name.startswith('.'):
            continue  # skip files or hidden directories
        
        # Paths for saving processed images
        updated_person_dir = updated_save_dir / label_dir.name
        cnn_person_dir = cnn_save_dir / label_dir.name
        updated_person_dir.mkdir(parents=True, exist_ok=True)
        cnn_person_dir.mkdir(parents=True, exist_ok=True)

        # Remove existing test images before saving new ones
        for folder in [updated_person_dir, cnn_person_dir]:
            for f in folder.iterdir():
                if f.is_file():
                    f.unlink()

        print(f"Processing {label_dir.name}...")

        # Process all images
        for img_path in label_dir.iterdir():
            if not img_path.is_file() or img_path.name.startswith('.'):
                continue  # skip directories and hidden files

            # Load image
            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # shape (1,C,H,W)

            # Apply pipeline
            transformed_imgs_lp, transformed_imgs_cnn = pipeline(img_tensor)

            # Save LP transformed images
            for n, transformed_img_tensor in enumerate(transformed_imgs_lp[0]):
                transformed_img_pil = TF.to_pil_image(transformed_img_tensor.clamp(0,1))
                file_path = updated_person_dir / f"{img_path.stem}_proc{n}.png"
                transformed_img_pil.save(file_path)

            # Save CNN transformed images
            for n, transformed_img_tensor in enumerate(transformed_imgs_cnn[0]):
                transformed_img_pil = TF.to_pil_image(transformed_img_tensor.clamp(0,1))
                file_path = cnn_person_dir / f"{img_path.stem}_proc{n}.png"
                transformed_img_pil.save(file_path)

        print(f"Completed {label_dir.name}: {len(list(updated_person_dir.iterdir()))} LP images, "
              f"{len(list(cnn_person_dir.iterdir()))} CNN images saved.")

# =========================
# Optional: start LP main
# =========================
# for i in range(5):
#     print(f"Starting LP iteration {i}...")
#     main_salience.main(lp=True, dataset_name=root)
