from utils import *
from transformation import *

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------

# Choose the dataset to load:
#   - "celeb":  only supports splits "train" and "test"
#   - "faces":  supports identity counts {4, 8, 16, 32, 64, 128}
#               and splits "train", "valid", "test"
dataset_name   = "faces"  # Options: "celeb", "faces"
identity_count = 4        # Only used when dataset_name == "faces"
split          = "train"   # For "celeb": {"train", "test"};
                            # for "faces": {"train", "valid", "test"}

# ------------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------------

# load_dataset parameters:
#   dataset : str   -> which dataset to load ("celeb" or "faces")
#   identity: int   -> number of distinct identities (only for "faces")
#   task    : str   -> which split to load ("train", "valid", "test")
ds = load_dataset(
    dataset=dataset_name,
    identity=identity_count,
    task=split
) 

show_images([logpolar_cv2(ds[7][0]), logpolar_manual(ds[7][0])])