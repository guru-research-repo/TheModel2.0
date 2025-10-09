# %%
import pandas as pd
import matplotlib.pyplot as plt

num_ident = 32 # number of identities trained on
# num_fix = 64 # number of fixation poitns trained on 
sal = 'CNN' # LP or CNN
title = 'Fixations' if sal == 'LP' else 'Crops' 

# Ensure numeric types
df = pd.read_csv(f'{num_ident}_{sal}_aggregated_metrics_per_fixations.csv').apply(pd.to_numeric, errors='coerce')

# Extract data
fixs = df['fixation_points'].values
train = df['train_mean'].values
train_std = df['train_std'].values
valid = df['valid_mean'].values
valid_std = df['valid_std'].values
test = df['test_mean'].values
test_std = df['test_std'].values

fig, ax = plt.subplots()

# Plot bars w/ std
ax.bar(fixs, train, label='Train', color='tab:blue', yerr=train_std)
ax.bar(fixs, valid, label='Upright', color='tab:orange', yerr=valid_std)
ax.bar(fixs, test, label='Inverted', color='tab:green', yerr=test_std)

# Labels and legend
ax.set_xlabel('Number of Fixation Points')
ax.set_ylabel('Accuracy')
ax.set_title(f' {title} on {num_ident} Identities - {sal}')
ax.legend()
ax.grid(True)

plt.savefig(f'{num_ident}_{sal}_faces.png')

plt.show()

# %%