# %%
# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb.

# %% auto 0
__all__ = ['home', 'comp', 'path', 'test_files', 'train_files']

# %% ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb 2
# this is a notebook for receiving code snippet from other notebooks

# %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 8
# make sure fastkaggle is install and imported
import os

# %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 9
try: import fastkaggle
except ModuleNotFoundError:
    os.system("pip install -Uq fastkaggle")

from fastkaggle import *

# %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 11
# use fastdebug.utils 
if iskaggle: os.system("pip install nbdev snoop")

# %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 12
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

# %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 14
# import for dealing with vision problem
from fastai.vision.all import *

# %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 49
# download (if necessary and return the path of the dataset)
home = "/Users/Natsume/Documents/fastdebug/kaggle_datasets/"
comp = 'paddy-disease-classification' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
# path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')

# %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 56
# set seed for reproducibility
set_seed(42)

# %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 65
# map the content of all subfolders of images
check_subfolders_img(path)

# %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 68
# to extract all images from a folder recursively (for subfolders)
test_files = get_image_files(path/"test_images")
train_files = get_image_files(path/"train_images")

# %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 78
# to display a random image from a path 
randomdisplay(train_files, 200)
randomdisplay(path/"train_images/dead_heart", 128)

# %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 88
# remove all images which fail to open
remove_failed(path)


# %%
@snoop
def remove_failed(path):
#     from fastai.vision.all import get_image_files, parallel
    print("before running remove_failed:")
    check_subfolders_img(path)
    failed = verify_images(get_image_files(path))
    print(f"total num: {len(get_image_files(path))}")
    print(f"num offailed: {len(failed)}")
    failed.map(Path.unlink)
    print()
    print("after running remove_failed:")
    check_subfolders_img(path)
# File:      ~/Documents/fastdebug/fastdebug/utils.py
# Type:      function


# %%
remove_failed(path)

# %%
