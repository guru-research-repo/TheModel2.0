# import pandas as pd
# import matplotlib.pyplot as plt

# # Ensure numeric types
# df = pd.read_csv('training_history_20250610_051312.csv').apply(pd.to_numeric, errors='coerce')

# # Extract data
# epochs = df['epoch'].values
# train = df['train_mean'].values
# valid = df['valid_mean'].values
# valid_std = df['valid_std'].values
# test = df['test_mean'].values
# test_std = df['test_std'].values

# fig, ax = plt.subplots()

# # Plot mean lines
# ax.plot(epochs, train, label='Train', color='tab:blue')
# ax.plot(epochs, valid, label='Valid', color='tab:orange')
# ax.plot(epochs, test, label='Test', color='tab:green')

# # Shade ±1 std deviation
# ax.fill_between(epochs, valid - valid_std, valid + valid_std,
#                 color='tab:orange', alpha=0.2)
# ax.fill_between(epochs, test - test_std, test + test_std,
#                 color='tab:green', alpha=0.2)

# # Labels and legend
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy per Epoch with Standard Deviation Shading')
# ax.legend()
# ax.grid(True)
# plt.savefig("run5_mapping.png")
# # plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and clean the CSV
df = pd.read_csv("training_summary_avg_std.csv")
df = df.drop(index=[0, 1]).reset_index(drop=True)

# Rename columns
df.columns = [
    'index', 'train_mean', 'train_std',
    'val_upright_mean', 'val_upright_std',
    'val_inverted_mean', 'val_inverted_std'
]

# Convert to float
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Add epoch column
df['epoch'] = df.index

# Convert to NumPy float arrays for safe plotting
epoch = np.asarray(df['epoch'], dtype=np.float32)

train_mean = np.asarray(df['train_mean'], dtype=np.float32)
train_std = np.asarray(df['train_std'], dtype=np.float32)

val_upright_mean = np.asarray(df['val_upright_mean'], dtype=np.float32)
val_upright_std = np.asarray(df['val_upright_std'], dtype=np.float32)

val_inverted_mean = np.asarray(df['val_inverted_mean'], dtype=np.float32)
val_inverted_std = np.asarray(df['val_inverted_std'], dtype=np.float32)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epoch, train_mean, label='Train Accuracy', color='tab:blue')
plt.fill_between(epoch, train_mean - train_std, train_mean + train_std, alpha=0.2, color='tab:blue')

plt.plot(epoch, val_upright_mean, label='Validation Upright Accuracy', color='tab:orange')
plt.fill_between(epoch, val_upright_mean - val_upright_std, val_upright_mean + val_upright_std, alpha=0.2, color='tab:orange')

plt.plot(epoch, val_inverted_mean, label='Validation Inverted Accuracy', color='tab:green')
plt.fill_between(epoch, val_inverted_mean - val_inverted_std, val_inverted_mean + val_inverted_std, alpha=0.2, color='tab:green')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training, Validation Upright & Inverted Accuracies with Std Dev")
plt.yticks(np.arange(0, 1.21, 0.2))
plt.xticks(np.arange(0, 240, 40))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_glabella_mapping.png")
plt.show()
