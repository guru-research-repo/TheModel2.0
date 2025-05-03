# TheModel2.0

## Getting Started
Steps to run the project:

1. Clone the repository (if you haven't done it already):
   ```bash
   git clone https://your.repo.url/TheModel2.0.git
   cd TheModel2.0
2. Run `python download.py` to get all dataset downloaded.
3. Create and activate a Conda environment
   ```bash
   conda create -n themodel2 python=3.10
   conda activate themodel2
4. Install PyTorch
   Make sure you have CUDA 12.8 support, then run:
   `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
5. Install the remaining dependencies (may need additional packages)
   `pip install numpy pandas pyyaml matplotlib opencv-python`
6. Run `python data_preprocess.py` to create pre-processed data on local disk.
7. From project root, run `python main.py` command.