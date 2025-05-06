# TheModel2.0

## Getting Started
Steps to run the project:

1. Clone the repository (if you haven't done it already):
   ```bash
   git clone https://your.repo.url/TheModel2.0.git
   cd TheModel2.0
2. Create and activate a Conda environment
   ```bash
   conda create -n themodel2 python=3.10
   conda activate themodel2
3. Install PyTorch
   Make sure you have CUDA 12.8 support, then run:
   `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
4. Install the remaining dependencies (may need additional packages)
   `pip install numpy pandas pyyaml matplotlib opencv-python tqdm`
5. Run `python download.py` to get all dataset downloaded.
6. Run `python data_preprocess.py` to create pre-processed data on local disk.
7. From project root, run `python main.py` command, or `torchrun --nproc_per_node=6 --master_port=29500 main_ddp.py`.
8. See the `output` folder for train, validation and test accuracy / standard deviation.

*Note*: main_ddp.py is used to rapidly train a model with multiple GPU, the epoch accuracy only reflects the average accuracy of one GPU, not the whole batch. <br>
*Note*: main_ddp_test.py updates batch accuracy among GPUs, but significantly impedes the training speed. This method is not recommended to use.