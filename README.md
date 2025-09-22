# TheModel2.0

## This branch changes:
Creating preprocessed salience data instead of real-time transformations.

Using same fixation points for each base image instead of random fixation points every time an image is seen.

## Running this branch:

1. Run `python salience_preproccess.py` to create pre-processed fixation point data on local disk.
2. Run `python main_salience.py` to run salience experiments
   - For each experiment, update line 19 to your desired salient count, e.g. `salient_counts = [4]`
   - For each experiment, update line 26 with the corresponding batch size for chosen salient counts, e.g. `batch_size = 64`

## Plotting

1. Once done running, move the .csv outputs from `/output` to `/visualization` 
- only do one type of output at a time (LP or CNN)
2. `python foo.py` to aggregate data
3. `python plot.py` to plot it
