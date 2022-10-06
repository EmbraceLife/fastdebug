# Learning fastai with joy


```
#| hide
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>


## Search notes and notebooks

### search notebooks
The first step to learn fastai with joy is to make revision easier. I would like to be able to search learning points in fastai notebooks with ease.

If I want to read or run the notebook, I could click the second link to run the notebook on Kaggle.


```
#| column: page
fastnbs("how gradient accumulation work")
```


<style>.container { width:100% !important; }</style>



### <mark style="background-color: #ffff00">how</mark>  does <mark style="background-color: #ffff00">gradient</mark>  <mark style="background-color: #ffff00">accumulation</mark>  <mark style="background-color: #FFFF00">work</mark>  under the hood





<!-- #region -->
For instance, here's a basic example of a single epoch of a training loop without gradient accumulation:

```python
for x,y in dl:
    calc_loss(coeffs, x, y).backward()
    coeffs.data.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()
```

Here's the same thing, but with gradient accumulation added (assuming a target effective batch size of 64):

```python
count = 0            # track count of items seen since last weight update
for x,y in dl:
    count += len(x)  # update count based on this minibatch size
    calc_loss(coeffs, x, y).backward()
    if count>64:     # count is greater than accumulation target, so do weight update
        coeffs.data.sub_(coeffs.grad * lr)
        coeffs.grad.zero_()
        count=0      # reset count
```

The full implementation in fastai is only a few lines of code -- here's the [source code](https://github.com/fastai/fastai/blob/master/fastai/callback/training.py#L26).

To see the impact of gradient accumulation, consider this small model:
<!-- #endregion -->

```python
train('convnext_small_in22k', 128, epochs=1, accum=1, finetune=False)
```





[Open `0010_fastai_scaling_up_road_to_top_part_3` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/2022part1/0010_fastai_scaling_up_road_to_top_part_3.ipynb)



[Open `0010_fastai_scaling_up_road_to_top_part_3` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/scaling-up-road-to-the-top-part-3)


If `fastnbs` doesn't return anything to your query, it is because the search ability of `fastnbs` is minimum, I need to learn to improve it.  But don't worry the next function below `fastlistnbs` will assist you to continue searching. 

### list all notebook learning points
I would also like to view all the learning points (in the form of questions) of all the fastai notebooks I have studied. This is a long list, so press `cmd + f` and search keywords e.g., "ensemble" to find the relevant questions, and then use `fastnbs` to search and display the details like above.

press `cmd + o` to view them all without scrolling inside a small window


```
#| column: page
fastlistnbs() 
```

    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0007_fastai_how_random_forests_really_work.md
    ## Introduction
    ### why ensemble of decision trees, such as Random Forests and Gradient Boosting Machines
    ### how to set print options for numpy and import `fastai.imports`
    ## Data preprocessing
    ### how to get Titanic dataset ready for creating a decision tree model
    ## Binary splits
    ### what is binary splits and who does it work
    ### how to plot barplot and countplot with Seaborn
    ### Create a simplest model based on binary split
    ### how to do train and test split using `sklearn.model_selection.train_test_split`
    ### how to access dependent and independent values for both training set and test set 
    ### calc the prediction (the simplest so far)
    ### calc loss with mean absolute error using `sklearn.metrics.mean_absolute_error`
    ### how to do binary split on a continuous column rather than category column
    ### how to plot a boxenplot on survival and non-survival using `sns.logFare` column; how to a density plot with logFare using `sns.kdeplot` 
    ### how to find the binary split to calc predictions based on logFare using the boxenplot above
    ### see how good is this model and prediction using loss (mean absolute error)
    ### how does impurity measure the goodness of a split; how to create impurity as a measure for how good of a split
    ### how to create a score function for a single side of the split
    ### how to create the score function to measure the goodness of a binary split
    ### calc the impurity score for sex split and then logFare split
    ### how to make interactive on choose different split and calc score on continuous columns
    ### how to make interactive on choose different split and calc score on categorical columns
    ### how to make a list of all possible split points
    ### how to get the score for all possible splits of a particular column like Age; how to get the index for the lowest core
    ### how to write a function to return the best split value and its score on a particular column given the dataframe and the name of the column
    ### how to run this function on all columns of the dataset
    ### what is OneR classifier; why should it be a baseline to more sophisiticated models
    ## Creating a decision tree
    ### how is to do better than a OneR classifier which predict survival using sex? how about doing another OneR upon the first OneR classifier result (male group and female group)
    ### how to get the dataset splitted by sex in pandas dataframe
    ### how to find the best binary splits and score out of all columns in male dataset and then femal dataset
    ### what does a decision tree mean when the second binary split is done here
    ### how to do a decision tree automatically using sklearn
    ### how to visualize the decision tree above
    ## how is gini different from impurity 
    ### how to cacl gini
    ### how to wrap the process of preparing submission csv file for kaggle
    ### why no need to worry about dummy variables in decision trees
    ## The random forest
    ### what is random forest; what is bagging; what is the great insight behind it
    ### how to create uncorrelated trees using random subset of data
    ### how to make prediciton on each tree and take average on them, and then calc the loss
    ### how is sklearn's RandomForestClassifier differ from the forest from scratch above; how to do random forest with sklearn
    ## Conclusion
    ### how should we think of simple models like OneR and decision tree and randomforest
    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0004_fastai_how_neuralnet_work.md
    ## Fitting a function with *gradient descent*
    ### Is neuralnet just a math function? what does the function look like?
    ### why neuralnet is random at first and how to make neuralnet useful
    ### `plot_function`: how to plot a function with plt; how to create x input with torch.linspace; how to plot x, y, color and title with plt;
    ### how to create a particular quadratic function
    ### how to write a function `quad` to create any quadratic function
    ### how does `partial` and `quad` work to modify `quad` to a slightly different func?
    ### how to add noise to both mult and add of the neuralnet/function; how to create noise using `np.random.normal`
    ### how to create a random seed to ensure x and y are the same each run
    ## A numpy book recommended by Jeremy; what is a tensor
    ### how to scatterplot with plt
    ### how to plot a scatterplot and a line and slides for 3 params of the line func
    ### why need a loss function? how to write a mean absolute error function with torch.abs and mean
    ### how display and change loss by changing values of params with sliders of interactive plot
    ### A 15-min calculus video series recommended by Jeremy to watch first
    ## Automating gradient descent
    ### how derivatives automatically guide params to change for a lower loss
    ### how to create a mean absolute error function on any quadratic model
    ### how to create an random tensor with 3 values as initialized params
    ### how to calc gradients of params? 1. tell PyTorch to get ready for calculating gradients for these params; 2. calc loss; 3. calc the gradients with `loss.backward()`; 4. how to access params' gradients; 
    ### how to change params with gradients properly to lower loss¶
    ### why `with torch.no_grad():` when updating params with gradients
    ### how to do 10 iterations of updating params with gradients
    ## How a neural network approximates any given function
    ## how to combine a linear func with max(x, 0) into a rectified linear function; how to use torch.clip(y, 0.) to perform max(y, 0.)
    ## how to use partial to wrap rectified_linear to create a specific rectified_linear func
    ## how to use `F.relu` to replace `torch.clip` to create a rectified linear func; 
    ### create double and quaduple relu func/neuralnet
    ## How to recognise an owl
    ### deep learning basically is drawing squiggly lines infinitely given computation and time
    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0001_fastai_is_it_a_bird.md
    ## Useful Course sites
    ## How to use autoreload
    ## How to install and update libraries
    ## Know a little about the libraries
    ### what is fastai
    ### what is duckduckgo
    ## How to use fastdebug with fastai notebooks
    ### how to use fastdebug
    ### Did I document it in a notebook before?
    ### Did I document it in a src before?
    ## how to search and get a url of an image; how to download with an url; how to view an image;
    ### how to create folders using path; how to search and download images in folders; how to resize images 
    ## Train my model
    ### How to find and unlink images not properly downloaded
    ### How to create a DataLoaders with DataBlock; how to view data with it
    ### How to build my model with dataloaders and pretrained model; how to train my model
    ### How to predict with my model; how to avoid running cells in nbdev_prepare
    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0006_fastai_why_should_use_framework.md
    ## Introduction and set up
    ### what are the benefits of using fastai and PyTorch frameworks
    ### which fastai module is for tabular data; how to set float format display for pandas; how to set random seed;
    ## Prep the data
    ### no worry of dummy variables, normalization, missing values and so on if using fastai; interesting feature ideas from a nice Titanic feature notebook;
    ### how to create a tabular dataloaders with `TabularPandas` which handles all messing processing; how to set the parameters of `TabularPandas`
    ## Train the model
    ### how to create a tabular learner using tabular dataloader, metrics and layers
    ### how to find the learning rate automatically in fastai
    ### how to pick the best learning rate from the learning rate curve; how to train model 16 epochs using `learn.fit`
    ## Submit to Kaggle
    ### how to prepare test data including added new features
    ### how to apply all the processing steps of training data to test data with `learn.dls.test_dl`
    ### how to calc all predictions for test set using `learn.get_preds`
    ### how to prepare the results of test set into a csv file for kaggle submission; how to save into csv file without idx number
    ## Ensembling
    ### what is ensembling and why it is more robust than any single model
    ### how to create an ensemble function to create multiple models and generate predictions from each of them
    ### how to get the average predictions from all ensembed models
    ### how to create the csv file to Titanic competition
    ## Final thoughts
    ### Why you should use a framework like fastai
    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0010_fastai_scaling_up_road_to_top_part_3.md
    ## Memory and gradient accumulation
    ### how to get the train_val dataset folder/path ready; how to get the test set images files ready
    ### how to quickly train an ensemble of larger models with larger inputs on Kaggle
    ### how to find out the num of files in each disease class using `pandas.value_counts`
    ### how to choose a data folder which has the least num of image files for training
    ### how `fine_tune` differ from `fit_one_cycle`
    ### how to create a `train` function to do either fine_tune + tta or fit_one_cycle for all layers without freezing; how to add `gradient accumulation` to `train`
    ### what does gradient accumulation do?
    ### What benefits does gradient accumulation bring
    ### how does gradient accumulation work under the hood
    ### how to find out how much gpu memory is used; and how to free up the gpu memory
    ## Checking memory use
    ### how to check the gpu memory usage of large models with large image inputs
    ## Running the models
    ### how to use a dictionary to organize all models and their item and batch transformation setups
    ### how to train all the selected models with transformation setups and save the tta results into a list
    ## Ensembling
    ### how to save all the tta results (a list) into a pickle file
    ### how to get all the predictions from a list of results in which each result contains a prediction and a target for each row of test set
    ### why and how to double the weights for vit models in the ensembles
    ### what is the simplest way of doing ensembling
    ### how to all the classes or vocab of the dataset using dataloaders; how to prepare the csv for kaggle submission
    ### how to submit to kaggle using fastkaggle api
    ## Conclusion
    ### how fastai can superbly simply the codes and standardize processes
    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0008_fastai_first_steps_road_to_top_part_1.md
    ### how to install fastkaggle if not available
    ### how to iterate like a grandmaster
    ### what are the related walkthrus on paddy doctor competition
    ## Getting set up
    ### how to setup for fastkaggle; how to use fastkaggle to download dataset from kaggle; how to access the path
    ### which fastai module to use for vision problem; how to check files inside the dataset path; why Jeremy recommend not to use seed in your own analysis;
    ## Looking at the data
    ### how to access a subfolder by name using path from `setup_comp`; how to extract all image files from a folder
    ### how to create an image from an image file; how to access the size of an image; how to display it with specified size for viewing
    ### how to use `fastcore.parallel` to quickly access size of all images; how to count the occurance of each unique value in a pandas 
    ### how to create an image dataloaders; how to setup `item_tfms` and `batch_tfms` on image sizes; why to start with the smallest sizes first; how to display images in batch
    ## Our first model
    ### how to pick the first pretrained model for our model; how to build our model based on the selected pretrained model
    ### how to find the learning rate for our model
    ## Submitting to Kaggle
    ### how to check the kaggle submission sample csv file
    ### how to sort the files in the test set in the alphabetical order; how to create dataloaders for the test set based on the dataloaders of the training set
    ### how to make predictions for all test set; and what does `learn.get_preds` return
    ### how to access all the classes of labels with dataloaders
    ### how to map classes to each idx from the predictions
    ### how to save result into csv file
    ### how to submit to kaggle with fastkaggle api
    ## Conclusion
    ### what is the most important thing for your first model
    ## Addendum
    ### how to quickly push your local notebook to become kaggle notebook online
    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0009_fastai_small_models_road_to_the_top_part_2.md
    ## Going faster
    ### why kaggle gpu is much slower for training and how does fastai to fix it with `resize_images`
    ### how to create a new folder with `Path`
    ### how to resize all images (including those in subfolders) of `train_images` folder and save them into a new destination folder; max_size = 256 does shrink the total size by 4+, but question: how Jeremy pick 256 not 250; 
    ### how to create an image dataloaders using the resized image folder and specify the resize for each image item; how to display just 3 images in a batch
    ### how to wrap dataloaders creation, model creation, fine tuning together in a func `train` and return the trained model; how use model architecture, item transforms, and batch transforms, and num of epochs as the params of the `train` function;
    ## A ConvNeXt model
    ### How to tell whether a larger pretrained model would affect our training speed by reading GPU and CPU usage bar? why to pick convnext_small for our second model;
    ### how to load and use a new pretrained model in fastai
    ## Preprocessing experiments
    ### question: why trying different ways of cutting images could possibly improve model performance; what are the proper options for cutting images or preparing images
    ### how to try cutting image with `crop` instead of `squish` 
    ### what is transform image with padding and how does it differ from squish and crop
    ### question: how `resize(256, 192)` and `size(171, 128)` are determined
    ## Test time augmentation
    ### how does test time augmentation TTA work; question: what is the rationale behind TTA
    ### how to check the performance of our model on validation set
    ### how to display the transformations which have been done to a single image in the training set
    ### how to do TTA on validation set
    ### how to calc the error rate of the tta_preds
    ## Scaling up
    ### how to scale up on the model using padding and the tta approach in terms of image size and epoch number
    ### how to check the performance of the scaled up model using validation set
    ## Submission
    ### how to use TTA to predict instead of the usual `get_preds` to get predictions on the test set
    ### how to get the index of the predictions
    ### how to replace index with vocab or classes
    ### how to submit prediction csv to kaggle with comment using fastkaggle api
    ### how to push local notebook to Kaggle online
    ## Conclusion
    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0002_fastai_saving_a_basic_fastai_model.md
    ## what to import to handle vision problems in fastai
    ## how to download and decompress datasets prepared by fastai
    ## how to tell it is a cat by reading filename
    ## how to create dataloaders with `from_name_func`
    ## how to create a pretrained model with resnet18 and error_rate; how to fine tune it 3 epochs
    ## how to export model to a pickle file and download it from Kaggle
    ## how to convert ipynb to md
    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0005_fastai_linear_neuralnet_scratch.md
    ## how to not execute the entire notebook
    ## Introduction
    ## How to download kaggle dataset to your local machine or colab? how to ues kaggle api and zipfile to download data into specified folder; how to use `pathlib.Path` to create a path;
    ## how to set the print display option for numpy, torch and pandas
    ## Cleaning the data
    ### how to read csv file with pandas and `path/'subfolder_name'`
    ### why missing value is a problem? how to find out the num of missing values of each column with pandas?
    ### which value is most used to replace missing value? how to get mode for each column with pandas using `iloc[0]`
    ### how to use pandas `iloc` function
    ### how to fill missing values with mode without making a new copy with `pandas.fillna`
    ### how to get a quick summary of all the numeric columns with pandas and numpy
    ### what is long-tailed data in histogram and why it is a problem for neuralnet
    ### how to plot histogram with pandas on a single column
    ### how to fix long-tailed data with logarithm; why should logarithm work; how to handle zero values when applying logarithm
    ### how to get a quick summary of all the non-numeric columns with pandas
    ### when do we need dummy variables and how to create dummy variables with pandas
    ### how to check the first few rows of selected columns with pandas
    ### how to create dependent/target variable and independent/predictor variables in PyTorch tensors; how to create variables in tensor from pandas dataframe
    ### how to check the size (rows and columns) of independent variables in tensor
    ## Setting up a linear model
    ### how to create coefficients for each (column) of our independent variables; how to get random seed in torch; how to get the num of columns; how to create random number between -0.5 and 0.5;
    ### why no bias or a constant is needed for this Titanic dataset?
    ### why a column `Age` having higher values than other columns can cause problem for our model; how to solve this problem by making them the same scale; how to get the max value of each column with pandas dataframe max func
    ### what is maxtrix by vector operation (multiply or divide)
    ### How to calculate the prediction of a linear model
    ### how to look at the first 10 values of predictions
    ### how to calc mean absolute error
    ### how to calc predictions with a func `calc_preds`; how to calc loss with a func `calc_loss`
    ## Doing a gradient descent step
    ### How to cacl gradients for coefficients
    ### why set gradients to zero after each gradient descent step; how to set gradient to zero; how to do one iteration of training
    ### what does _ mean for `coeffs.sub_()` and `grad.zero_()`
    ## Training the linear model
    ### how to split the dataset by using train and valid idx produced by `fastai.data.transforms.RandomSplitter`
    ### how to udpate coefficients in a function `update_coeffs`
    ### how to do one epoch training in a function `one_epoch`
    ### how to initializing coefficients in a function `init_coeffs`
    ### how to integrate funcs above to form a function `train_model` on multiple epochs
    ### how to display coefficients of the model with func `show_coeffs`
    ## Measuring accuracy
    ### There are many possible loss options such as accuracy other than mean absolute error
    ### how to calc accuracy for the binary dependent variable
    ### how to wrap the process of calc accuracy using coeffs into a func `acc(coeffs)`
    ## Using sigmoid
    ### when will we be needing something like sigmoid
    ### how to write and plot a func like `sigmoid` using sympy
    ### how to update `calc_preds` by wrapping `torch.sigmoid` around prediction
    ## Submitting to Kaggle
    ### read test data using `pandas.read_csv`
    ### why and how to fill the missing value in Fare column with 0 instead of mode
    ### how to handle missing values, long-tailed distribution and dummies together for test data
    ### how to turn independent variable values into tensor
    ### how to make sure independent variable in test data share the same value scare with those in training data
    ### how to turn true or false into 1 or 0 and save them into a column
    ### how to select two columns of a dataframe and save them into a csv file using `to_csv`
    ### how to check the first few lines of the csv file using `!head`
    ## Using matrix product
    ### how to do matrix product `@` between a matrix and a vector with PyTorch; how to use `@` instead of doing multiplication and then addition together
    ### update `calc_preds` func using matrix multiplication `@`
    ### how to initialize coeffs and turn it into a matrix with a single column; question: but why make coeffs between 0 and 0.1 instead of -0.5 and 0.5
    ### how to turn a single column of dependent variable into a single column matrix or a column vector
    ### question: why set learning rate to be 100 for this Titanic model
    ## A neural network
    ### how to initialize coeffs for a neuralnet with two layers (including a hidden layer of n neurons) and the final output layer is a single neuron with a single coeff; question: how do `-0.5` and `-0.3` come from?
    ### how to update `calc_preds` for this 2 layer neuralnet using `F.relu`, matrix product `@`, and `torch.sigmoid`
    ### how to update coeffs layer by layer with `layer.sub_` and `layer.grad.zero_`
    ### question: how the learning rate is chosen (1.4 or 20) when training
    ## Deep learning
    ### how to move from neuralnet with one hidden layer to a deep learning
    ### why so many messy constants and how they block the progress of deep learning in the early days
    ### how to use `enumerate` to loop both idx and item
    ## Final thoughts
    ### How much similar or different between practical models and the models from scratch above
    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0003_fastai_which_image_model_best.md
    ## timm
    ## how to git clone TIMM analysis data; how to enter a directory with %cd
    ## how to read a csv file with pandas
    ## how to merge data with pandas; how to create new column with pandas; how to string extract with regex expression; how to select columns up to a particular column with pandas; how to do loc in pandas; how to select a group of columns using str.contains and regex
    ## Inference results
    ### how to scatterplot with plotly.express; how to set the plot's width, height, size, title, x, y, log_x, color, hover_name, hover_data; 
    ### how to scatterplot on a subgroup of data using regex and plotly
    ## Training results
    ### convert ipynb to md


### Search notes

I would also like to search my own fastai notes with ease. The `fastnotes` can search but very rough at the moment, and the notes need a lot of rewrite.


```
# fastnotes("how random forest work")
```


```

```
