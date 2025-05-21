import glob
import pandas as pd

# Collect data from all CSV files in the current directory
df_list = []
for fpath in glob.glob("*.csv"):
    try:
        df = pd.read_csv(fpath, usecols=['epoch', 'train_mean', 'valid_mean', 'test_mean'])
        df['source_file'] = fpath
        df_list.append(df)
    except ValueError:
        print(f"⚠️ Skipped '{fpath}': required columns not found.")

if not df_list:
    raise RuntimeError("No CSV files found with the required columns: 'epoch', 'train_mean', 'valid_mean', 'test_mean'.")

trains = []
valids = []
tests  = []
for df in df_list:
    trains.append(df["train_mean"])
    valids.append(df["valid_mean"])
    tests.append(df["test_mean"])

# 3) Concatenate across files (axis=1) so each column is one file’s series
train_df = pd.concat(trains, axis=1)
valid_df = pd.concat(valids, axis=1)
test_df  = pd.concat(tests,  axis=1)

# 4) Build the aggregated dict
agg_dict = {
    "epoch":      df_list[0]["epoch"],               # keep epoch
    # "identity":   df_list[0]["identity"],            # optional: first file’s identity
    "train_mean": train_df.mean(axis=1),             # across files
    "valid_mean": valid_df.mean(axis=1),
    "test_mean":  test_df.mean(axis=1),
    "train_std":  train_df.std(axis=1, ddof=0),      # population‐std
    "valid_std":  valid_df.std(axis=1, ddof=0),
    "test_std":   test_df.std(axis=1, ddof=0),
}

# 5) Convert to DataFrame and save
agg_df = pd.DataFrame(agg_dict)
agg_df.to_csv("aggregated_metrics_per_epoch.csv", index=False)

print("Saved per-epoch aggregates to 'aggregated_metrics_per_epoch.csv'")