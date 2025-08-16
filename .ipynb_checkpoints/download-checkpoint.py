import os
import sys
import subprocess

# Ensure gdown is installed
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "gdown"])
    import gdown

# The shared-view URLs
urls = [
    "https://drive.google.com/file/d/1mFpBO1-XCgDOCQV01x92F3-I2hKVTuP1/view?usp=drive_link",  # CelebA
    "https://drive.google.com/file/d/1ud4OdpoWjULhqJQV50WZ9YF8dfHiIfo9/view?usp=drive_link",  # faces
    "https://drive.google.com/file/d/1q2xklkTf_x4h_28ZvcRIqd5f3kNJKBlt/view?usp=drive_link",  # cleaned faces
    "https://drive.google.com/file/d/1IpgrPpVpqv476u3O_Eqgd0NNK1vhcLeK/view?usp=drive_link", # dogs1k
]

# Desired filenames (match the original file types)
filenames = [
    "CelebA_HQ_facial_identity_dataset.zip",
    "faces.tar.gz",
    "cleaned_faces_dataset.zip",
    "dogs1k.zip",
]

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

for url, name in zip(urls, filenames):
    # Extract the file ID from the URL
    file_id = url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output_path = os.path.join(output_dir, name)
    
    print(f"Downloading {name}...")
    gdown.download(download_url, output_path, quiet=False)
    print(f"Saved to {output_path}\n")

print("Extracting files:")

import zipfile
import tarfile

# Extract CelebA_HQ_facial_identity_dataset.zip
with zipfile.ZipFile('data/CelebA_HQ_facial_identity_dataset.zip', 'r') as zip_ref:
    curr_path = "data/CelebA_HQ_facial_identity_dataset"
    os.makedirs(curr_path, exist_ok=True)
    zip_ref.extractall(path=curr_path)

# Extract faces.tar.gz
with tarfile.open('data/faces.tar.gz', 'r:gz') as tar_ref:
    curr_path = "data/faces"
    os.makedirs(curr_path, exist_ok=True)
    tar_ref.extractall(path=curr_path)

with zipfile.ZipFile('data/cleaned_faces_dataset.zip', 'r') as zip_ref:
    curr_path = "data/cleaned_faces_dataset"
    os.makedirs(curr_path, exist_ok=True)
    zip_ref.extractall(path=curr_path)

with zipfile.ZipFile('data/dogs1k.zip', 'r') as zip_ref:
    curr_path = "data/dogs1k"
    os.makedirs(curr_path, exist_ok=True)
    zip_ref.extractall(path=curr_path)

print("Extraction completed.")
