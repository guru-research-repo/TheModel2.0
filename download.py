import os
import sys
import subprocess
import zipfile
import tarfile

# Ensure gdown is installed
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "gdown"])
    import gdown

# Shared-view URLs for each dataset
urls = [
    "https://drive.google.com/file/d/1IpgrPpVpqv476u3O_Eqgd0NNK1vhcLeK/view?usp=drive_link" # dogs 1K
    # "https://drive.google.com/file/d/1mFpBO1-XCgDOCQV01x92F3-I2hKVTuP1/view?usp=drive_link",  # CelebA
    # "https://drive.google.com/file/d/1ud4OdpoWjULhqJQV50WZ9YF8dfHiIfo9/view?usp=drive_link",  # faces
    # "https://drive.google.com/file/d/1kOhJXRKaGpoQHKazFPG3QA8tLhBGNKOa/view?usp=drive_link",  # objects
    # "https://drive.google.com/file/d/137YEYgzi6qH5hXqpJOdqtyxe7hWgjAxZ/view?usp=drive_link",  # faces_cleaned
]

# Corresponding filenames
filenames = [
    "dogs1K.zip" # dogs 1K
    # "CelebA_HQ_facial_identity_dataset.zip",
    # "faces.tar.gz",
    # "ImageNet1k.zip",
    # "128_faces_manually_cleaned.zip", 
]

# Extraction target folders
extract_dirs = [
    "data/dogs1K"
    # "data/CelebA_HQ_facial_identity_dataset",
    # "data/faces",
    # "data/ImageNet1k",
    # "data/faces_cleaned",
]

# Create data folder
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Download each file if not already present
for url, name in zip(urls, filenames):
    output_path = os.path.join(output_dir, name)

    if os.path.exists(output_path):
        print(f"Skipping download of {name} (already exists).")
        continue

    file_id = url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    print(f"Downloading {name}...")
    gdown.download(download_url, output_path, quiet=False)
    print(f"Saved to {output_path}\n")

# Extract downloaded archives
print("Extracting files...\n")

for filename, extract_path in zip(filenames, extract_dirs):
    archive_path = os.path.join(output_dir, filename)

    # Skip extraction if target directory already has files
    if os.path.exists(extract_path) and any(os.scandir(extract_path)):
        print(f"Skipping extraction of {filename} (already extracted to {extract_path}).")
        continue

    os.makedirs(extract_path, exist_ok=True)

    if filename.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(path=extract_path)
            print(f"Extracted {filename} to {extract_path}")
    elif filename.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(path=extract_path)
            print(f"Extracted {filename} to {extract_path}")
    else:
        print(f"Unknown file format: {filename}")

print("\n All files downloaded and extracted.")
