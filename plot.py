import pandas as pd
import matplotlib.pyplot as plt

# Ensure numeric types
df = pd.read_csv('output/training_history.csv').apply(pd.to_numeric, errors='coerce')

# Extract data
epochs = df['epoch'].values
train = df['train_acc'].values
valid = df['valid_mean'].values
valid_std = df['valid_std'].values
test = df['test_mean'].values
test_std = df['test_std'].values

fig, ax = plt.subplots()

# Plot mean lines
# ax.plot(epochs, train, label='Train', color='tab:blue')
ax.plot(epochs, valid, label='Valid', color='tab:orange')
ax.plot(epochs, test, label='Test', color='tab:green')

# Shade ±1 std deviation
# ax.fill_between(epochs, valid - valid_std, valid + valid_std,
#                 color='tab:orange', alpha=0.2)
# ax.fill_between(epochs, test - test_std, test + test_std,
#                 color='tab:green', alpha=0.2)

# Labels and legend
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy per Epoch with Standard Deviation Shading')
ax.legend()
ax.grid(True)

plt.show()
