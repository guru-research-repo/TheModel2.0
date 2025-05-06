# import os

# # Path to the directory containing 128 identity folders
# identity_root = os.path.join("data","faces", "faces", "128_identities", "train")

# # Get all subdirectories (identity names)
# identity_names = [
#     name for name in os.listdir(identity_root)
#     if os.path.isdir(os.path.join(identity_root, name))
# ]

# # Sort alphabetically
# identity_names.sort()

# # Forward mapping: index -> name
# index_to_name = {i: name for i, name in enumerate(identity_names)}

# # Reverse mapping: name -> index
# name_to_index = {name: i for i, name in index_to_name.items()}

# # Optional: Print or save
# print("NAME TO INDEX",name_to_index)

# #To save as JSON (optional)
import json
# with open("reverse_identity_mapping.json", "w") as f:
#     json.dump(name_to_index, f, indent=2)
# Example sanity check
with open('reverse_identity_mapping.json', 'r') as f:
    rev_map = json.load(f)
    label_set = set(rev_map.values())
    assert label_set == set(range(len(label_set))), "Label mapping is non-contiguous or misaligned"
