# TheModel2.0

## Getting Started
Steps to run the project:

1. Clone the repository (if you haven't done it already):
   ```bash
   git clone https://your.repo.url/TheModel2.0.git
   cd TheModel2.0
2. Run `python download.py` to get all dataset downloaded. <br>
3. Each folder title is self-explanatory.<br>
   3.1. `CNN_on_Cars_Dataset` folder contains the code to run a baseline CNN ResNet18 model over the Cars Dataset (obtained from selecting the largest 128 car_model categories from the Comp_Cars dataset.<br>
   3.2. `LPNet_on_Cars_Dataset` folder contains the code to run the LPNet ResNet18 model over the same Cars Dataset to visualize the inversion effect.<br>
   3.3. `glabella_5runs_code` folder contains the code to run the LPNet ResNet18 model over the 'faces' Dataset. The pipeline follows 4 crops, glabella point detection using MediaPipe, rotation based on the point, foveation and log-polarisation based on the same point.<br>
   3.4 `DeepGaze_2` folder contains the code that can preprocess a `face` image by first finding 4 salient points (random 4 points out of the top-1000 salience points detected using DeepGaze 2) of the image, and then rotating, foveating and log-polarising based on the said fixation point. [No image cropping involved].<br>
   3.5. `DeepGaze_2` folder also contains 'Examples.ipynb' referenced from `https://github.com/matthias-k/DeepGaze` to visualize the outputs of DeepGaze 2 and DeepGaze 3. Clone the DeepGaze repo from their repo to successfully run the Examples.ipynb notebook.<br>
   3.6. `DeepGaze_2` folder also contains a notebook named `old_saliency.ipynb` that has the old Salience code from a previous Guron's repo. It visualizes the saliency heatmap and shows the most salience point.
