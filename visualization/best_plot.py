# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_ident = 32 # number of identities trained on
# num_fix = 64 # number of fixation poitns trained on 
sal = 'LP' # LP or CNN
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
width = 0.25
x = np.arange(len(fixs))

# Plot bars w/ std
ax.bar(x+0*width, train, label='Train', color='tab:blue', width=width, yerr=train_std)
ax.bar(x+1*width, valid, label='Upright', color='tab:orange', width=width, yerr=valid_std)
ax.bar(x+2*width, test, label='Inverted', color='tab:green', width=width, yerr=test_std)

# Labels and legend
ax.set_xlabel('Number of Fixation Points')
ax.set_ylabel('Accuracy')
ax.set_title(f' {title} on {num_ident} Identities - {sal}')
ax.set_xticks(x + width, fixs)
ax.legend()
ax.grid(True)
ax.set_ylim(0.0, 1.0) 
ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
plt.tight_layout()

plt.savefig(f'test.png')

plt.show()

# %%