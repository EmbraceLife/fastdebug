# %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 6
# make sure fastkaggle is install and imported
import os

try: import fastkaggle
except ModuleNotFoundError:
    os.system("pip install -Uq fastkaggle")

from fastkaggle import *

# use fastdebug.utils 
if iskaggle: os.system("pip install nbdev snoop")

if iskaggle:
    path = "../input/fastdebugutils0"
    import sys
    sys.path
    sys.path.insert(1, path)
    import utils as fu
    from utils import *
else: 
    from fastdebug.utils import *
    import fastdebug.utils as fu

# import for dealing with vision problem
from fastai.vision.all import *

# download (if necessary and return the path of the dataset)
home = "/Users/Natsume/Documents/fastdebug/kaggle_datasets/"
comp = 'paddy-disease-classification' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
# path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
