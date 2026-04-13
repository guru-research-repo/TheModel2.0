# TheModel2.0

## This branch changes:
Creating preprocessed salience data instead of real-time transformations.

Using same fixation points for each base image instead of random fixation points every time an image is seen.

## Running this branch:

1. Run `python salience_preproccess.py` to create pre-processed fixation point data on local disk.
   - This assumes `faces_cleaned` data is already downloaded. If not downloaded, first run `download.py`.
2. Run `python main_salience.py` to run salience experiments
   - For each experiment, update line 20 to your desired salient count, e.g. `salient_counts = [4]`
   - For each experiment, update line 27 with the corresponding batch size for chosen salient counts, e.g. `batch_size = 64`

## Plotting

1. Once done running main code, move the .csv outputs from `/output` to `/visualization` 
   - Only do one type of output at a time (LP or CNN)
   - Make sure to remove any old .csv files from `/visualization` before plotting
2. Run `python foo.py` to aggregate data
   - Update line 5 based on the number of fixation points for the current experiment, e.g. `num_fix = 4`
   - Update line 6 for CNN vs LP experiment, e.g. `sal = 'LP'`
3. Run `python plot.py` to plot data
   - Update line 6 based on the number of fixation points for the current experiment, e.g. `num_fix = 4`
   - Update line 7 for CNN vs LP experiment, e.g. `sal = 'LP'`
