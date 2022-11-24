---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

## Why reading/writing this one when we have @radek1's amazing beginner-friendly posts and discussions like this [one](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364062) already?
I guess I am a true beginner, so my knowing-nothingness is a perspective which amazingly helpful kagglers like @radek1 and @cdeotte can't have.

## What is a 'map' here?
Try to weave all the resources ([notebooks](https://www.kaggle.com/competitions/otto-recommender-system/code), [discussion](https://www.kaggle.com/competitions/otto-recommender-system/discussion) , linked resources) together from a beginner's perspective for beginners.

## What is the strategy to build a map?
By finding an amazing kaggler who is doing the competition as your guide and follow her/him closely, there are no shortage of great kagglers here, and it's @radek1 for me. Do follow him! To be honest, I don't think I could map out all of his sharing, as he is learning faster and the compounding power is scaring.

## What is a beginner? 
Sorry, I can only address a beginner like myself, who knows a little python and a little ML by watching part 1 course of fastai, has never really done a Kaggle competition and interested to try a Recommendation system one by learning from amazing kagglers through discussions and codes on Kaggle.

## What is Kaggle's OTTO competition and its dataset?

- A [brief](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363973) Intro to the competition and the dataset by @pnormann
	- [context](https://www.kaggle.com/competitions/otto-recommender-system/overview/description) of this competition
	- the [evaluation](https://www.kaggle.com/competitions/otto-recommender-system/overview/evaluation) function recall@20 
		- understanding the formula
			- If you read it patiently a few more times, like me you will be surprised that it is simpler than it looks
			- recall metric not reward duplicated aids [discussion](https://github.com/otto-de/recsys-dataset) is also confirmed by the formula
	- What is [the ground truth](https://www.kaggle.com/competitions/otto-recommender-system/overview/evaluation) in the test data
		- the [graph](https://github.com/otto-de/recsys-dataset/blob/main/.readme/ground_truth.png?raw=true) depicts it very well (just need to read it carefully a few more times)
	- cart or order without click [discussion](https://github.com/otto-de/recsys-dataset)

- a [detailed](https://github.com/otto-de/recsys-dataset) intro of OTTO dataset from github repo
	- [Key Features](https://github.com/otto-de/recsys-dataset/blob/main/README.md#key-features)
	- [Dataset Statistics](https://github.com/otto-de/recsys-dataset/blob/main/README.md#dataset-statistics)
	- [Get the Data](https://github.com/otto-de/recsys-dataset/blob/main/README.md#get-the-data)
	- [Data Format](https://github.com/otto-de/recsys-dataset/blob/main/README.md#data-format)
	- [Submission Format](https://github.com/otto-de/recsys-dataset/blob/main/README.md#submission-format)
	- [Prerequisites and Installation](https://github.com/otto-de/recsys-dataset/blob/main/README.md#prerequisites-and-installation)
	- [Evaluation](https://github.com/otto-de/recsys-dataset/blob/main/README.md#evaluation) (important)
		-  what is recall@20
		- [train test split](https://github.com/otto-de/recsys-dataset/blob/main/README.md#traintest-split) (important)
				- How competition host split train and test set visually with a [script](https://github.com/otto-de/recsys-dataset/blob/main/src/testset.py) provided
			- we can use the script to split our training set to do local validation
			- A [script](https://github.com/otto-de/recsys-dataset/blob/main/src/evaluate.py) is provided to calc recall@20 for each type and all
		- [recall@20 metrics calculation](https://github.com/otto-de/recsys-dataset/blob/main/README.md#metrics-calculation)
	- [FAQ](https://github.com/otto-de/recsys-dataset/blob/main/README.md#faq) (must read too)
		- How is a user `session` defined?
			- A session is all activity by a single user either in the train or the test set.
		- Are you allowed to train on the truncated test sessions?
			- Yes, for the scope of the competition, you may use all the data we provided.
		- How is Recall@20 calculated if the ground truth contains more than 20 labels?
			- If you predict 20 items correctly out of the ground truth labels, you will still score 1.0.


## Recommended resources to get started by amazing Kagglers

- [Andrew Ng Reccommender Systems](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363624) by @andradaolteanu
	- Ng's lectures with [timestamps](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363624#2021900) by me

- [Resources for getting started with recommender systems](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363814) by @radek1 
	- Xavier lectures with [timestamps](https://www.kaggle.com/competitions/otto-recommender-system/discussion/365039) by me
	- What is session based recommendations 
	- A blog [post](https://medium.com/nvidia-merlin/transformers4rec-4523cc7d8fa8) about Transformers for session based recommendations
	- great [advice](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363814#2023718) to a beginner like me by @radek1  
	- @radek1 [will look into](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363814#2015722) notebooks on merlin models (Transformer4Rec)

- [Recommendation Systems for Large Datasets](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364721) by @ravishah1 introduces two steps of doing Recsys
	- How large is OTTO dataset
	- What is Candidate Generation
	- What is Ranking model with links to 4 models

- [2 recsys libraries](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364062) shared by @radek1 (I copied directly from Radek below, not yet read the resources)
	-  [Transformers4Rec](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363980) by NVIDIA, deep learning session-based recommendation models, as recommended by [@snnclsr](https://www.kaggle.com/snnclsr)! 🚀 
	- [A post on RecBole](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363603), a very interesting looking Recommendation Model library implemented in PyTorch 🔥(thanks [@hidehisaarai1213](https://www.kaggle.com/hidehisaarai1213)) 


## Questions I should know the answers

- what challenges does huge number of aids (1.5 milliion unique products) bring us ([high cardinality of targets](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364722) by @radek1)? 
	- softmax not working well
	- a layer with 1.5million activation is too much computation

- how to approach the high cardinality of targets problem above? [summarized](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364722) by @radek1
	- an RNN or a Transformer trained on sequences and outputting single (or multiple) labels
	- implement sampled softmax in pytorch to overcome limitation of softmax (has anyone done it yet?)
	- instead of co-visitation matrix, @radek1 planed to train a matrix factorization model and share (maybe already done, need to find it)
	- With embeddings, you can use nearest neighbor search to make the whole idea of cardinality go away. (maybe done, I am searching for it)
	- "candidate rerank" models can handle high cardinality problem [mentioned](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364722#2021088) by @cdeotte

- What is [test data leakage](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363939)?  (use future events to help predict past events) by @cdeotte 

- Can training using test data here help improve the model? (yes, evidence from [notebook](https://www.kaggle.com/code/cdeotte/test-data-leak-lb-boost))

- What exactly we are asked to predict in the competition?
		- "The goal of this competition is to predict e-commerce clicks, cart additions, and orders. You'll build a multi-objective recommender system based on previous events in a user session." - [description page](https://www.kaggle.com/competitions/otto-recommender-system/overview/description)
		- "For each `session` in the test data, your task it to predict the `aid` values for each `type` that occur after the last timestamp `ts` the test session. In other words, the test data contains sessions truncated by timestamp, and you are to predict what occurs after the point of truncation." - [evaluation page](https://www.kaggle.com/competitions/otto-recommender-system/overview/evaluation)
		- "and predicted aidspredicted aids are the predictions for each session-type (e.g., each row in the submission file) _truncated after the first 20 predictions_." - [evaluation page](https://www.kaggle.com/competitions/otto-recommender-system/overview/evaluation)
	- a detailed [answer](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363939#2017432) by @pnormann the organizer and read the comments [chronologically](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363939#2016227) will be helpful too

- How is the test set we use different from the full test set? [randomly truncated](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363939#2017433)  by @pnormann

- What does the truncated test set we got look like?
	- [illuminating visualization](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363939#2016413) by @cdeotte

- How to use future events predict past events 
	- It is important to ponder on the [picture](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/main/Nov-2022/leak.png) annotated by @cdeotte on "use future events to predict past events" with "how?"
	- todo

- What is a session
	- [answer](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363554#2015486) given by @pnormann
	- helpful [comment](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363554#2015468) by @radek1

- What is a "cold start" problem of RecSys and why OTTO is one of them?
	- intuitive [analysis](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363874) on what by @narsil
	- on [why](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363874#2033243) by @cdeotte

- predicting the same aid twice when ground truth has it duplicated too, won't increase the hits? 
	- see [discussion](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363965#2016269) 

- Do sessions from training set and test set overlap? How is the test set made by truncating the test set (or/and training set)
	- [discussion](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363874#2015825) by @radek1 and @narsil
	- detailed [analysis](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363965) by @radek1

- How the test set is truncated (left part is for us, the right part is hidden for scoring competition)? 
	- [analysis](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363874#2015821) and official test split code shows it is [random](https://github.com/otto-de/recsys-dataset/blob/main/src/testset.py#L25) by @radek1
	- [answer](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363874#2016257) by @pnormann 
	- the sessions in the test set may not end within one week, but events outside the one week won't appear in test set.

- What does a "custom sub-session logic which could help modeling" would look like? 
	- [discussion](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363874#2017479)
- why we should not discard but treasure longer sessions?
	- great [analysis](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364375) by @radek1
	- my [rephrase](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364375#2033760)
- How people do analysis on the dataset to find interesting things? 
	- see the [topic](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364064) and first few comments for examples
- If ground truth has more than 20 items and you predict the first 20 items correctly, will you score 1?
	- yes and [confirmed](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364064#2018475) by @pnormann
- when a test session start with carts or orders, how do we make predictions on clicks?
	- [answer](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364064#2019209) and [code](https://github.com/otto-de/recsys-dataset/blob/main/src/labels.py#L5) on ground truth by @pnormann (dive in later)
- How to calculate the recall@20 metric of otto competition?
	- a nice [summary](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364530) of the confusion on the metrics by @dehokanta
	- created the [dataset](https://www.kaggle.com/datasets/radek1/otto-train-and-test-data-for-local-validation) for local validation using organizer's script @radek1
	- [evidence](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364534) from organizer's [script](https://github.com/otto-de/recsys-dataset/blob/main/src/evaluate.py#L83) to clear the metric confusion @radek
- Can Merlin DataLoaders make processing otto super faster? 
	- [answer](https://twitter.com/radekosmulski/status/1593378877216522240) by @radek1
- How the `ts` in Radek's parquet otto dataset differ from that of original jsonl otto dataset?
	- "**IMPORTANT CAVEAT:** Please note that in my dataset, in order to minimize its size without losing information (unless you care about milliseconds, but I assume seconds should be enough 🙂), I divide the `ts` column by 1000. Notebooks on Kaggle by other people use unmodified data, so if you would want to run this validation with that code, you need to either make changes to the code or multiply the `ts` column by 1000."
	- [which dataset](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991#2034277) did Radek refer to? (I found it [here](https://www.kaggle.com/datasets/radek1/otto-train-and-test-data-for-local-validation))
		- it's a different dataset: "otto_train_and_test_data_for_local_validation"
		- another dataset which does not have `ts/1000` is Otto Full Optimized Memory Footprint version 1 (version 2 also has `(ts/1000).astype(np.32)`)
		- "This is another version of the Otto dataset, where I have minimized the size of it even further without losing information. Among other things, I divided the ts column by 1000 (this way, I could store it as np.int32)."
- Otto Full Optimized Memory Footprint, how to use a specified [version](https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint)?
	- click Data Explorer to see different version options
	- but how to [use a specific version](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991#2034424) in Kaggle notebook? (usually a more patient and careful look after a good rest can solve it)
- How does @radek1 use `process_data.ipynb` to convert `train.jsonl` to `train.parquet` to achieve huge RAM usage reduction?
	- [why](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843#2024279) we can't process the data on Kaggle due to RAM shortage by me
	- convert `type` column from `string` to `uint8`, [9 times more RAM](https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files/comments#2025187) can be saved by @danielliao
	- [half of RAM can be saved](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991#2034507) by dividing `ts` column by 1000 and convert `ts` to `int32` by @danielliao
	- what's the evidence dividing 1000 on ts can save RAM or disk? [I got the evidence here](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991#2034291) 
	- the [updated](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843#2034437) `process_data.ipynb`, and the [original one](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843#2015604) without dividing `ts` by 1000 and convert from int64 to int32
	- doing the above conversion [only affect accuracy in milisecond not second accuracy](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991#2034529)
	- my experimental process_data [notebook](https://www.kaggle.com/code/danielliao/process-data-otto) processing `test.jsonl`
- How powerful can the last 20 aids of each test session as prediction be? 
	- my [notebook](https://www.kaggle.com/danielliao/akaggle-otto-rd-last-20-aids) built upon @radek1's notebook
	- submit to public leaderboard and run it on local validation
- how to use parquet OTTO dataset by Radek's [notebook](https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files?scriptVersionId=109945227) and my [version](https://www.kaggle.com/danielliao/kaggle-access-parquet-otto)
- otto_train_and_test_data_for_local_validation
	- where is the dataset [here](https://www.kaggle.com/datasets/radek1/otto-train-and-test-data-for-local-validation).
	- [how](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364534) the dataset above is created? @radek1
		- I created the dataset passing the competition train data to the script and used the last 7 days to create the test set.
- [Interesting findings](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364534) by @radek1 when creating the **otto_train_and_test_data_for_local_validation** dataset for local validation
	- "Generally, the interesting bit here is that the test set created on the last week of the train set seems considerably easier to predict on than what we have in the competition." - I wonder why?
	- "Another interesting observation s regarding the calculation of the metric. It is not a mean of per-row recall, rather it is recall calculated on hits across the entire dataset! (might be an important piece of information for anyone implementing the competition metric themselves!)" - what is the implication?
- [compare](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364534) @radek1 implementation and the src of the `evaluation.py` from organizer
- A great [example](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991#2023511) of how to incrementally building a notebook on @radek1's local validation setup
	- where we shall see Radek's dataset with `ts/1000` in Chris's notebook
	- how @cdeotte use the local validation setup to test [how it looks](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991#2023756) between local CV and public leaderboard on different versions of notebooks
- What to do when your eda found an issue in the dataset? example
	- "There are "new" AIDs (18785) in the validation test set comparing to validations train set). For the original train and test set this is not the case." 
	- [how it is solved](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991#2023756)
- How to do local validation on OTTO dataset
	- the setup [discussion](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991), [radek's notebook](https://www.kaggle.com/code/radek1/a-robust-local-validation-framework), 
	- and my experiment [notebook](https://www.kaggle.com/danielliao/kaggle-local-validation-framework-otto) 🔥 (todo)
- How does co-visitation matrix model work
	- simplified and improved notebook by @radek
	- my experiment [notebook](https://www.kaggle.com/danielliao/kaggle-covisitation-matrix-otto) (todo)

## todos: 

notebook: https://www.kaggle.com/code/radek1/polars-proof-of-concept-lgbm-ranker

local validation tracks public LB perfecty -- here is the setup [discussion](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991), [notebook](https://www.kaggle.com/code/radek1/a-robust-local-validation-framework)

- [Notebooks recap](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364062) by @radek1 (I copied directly from Radek)
	- The outstanding [co-visitation matrix](https://www.kaggle.com/code/vslaykovsky/co-visitation-matrix) by [@vslaykovsky](https://www.kaggle.com/vslaykovsky)!
	-   A notebook by [@cdeotte](https://www.kaggle.com/cdeotte) building on the co-visitation matrix 👆 and demonstrating the power of training on the test data! (maybe training is too strong of a word, rather using the leak in your calculations)
	-   My take on the above approach that is simplified and that runs on a `parquet` dataset (without having to read in json), read [more here](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364210)
	-   the last 20 AIDs are very powerful! ([original code](https://www.kaggle.com/code/ttahara/last-aid-20), [simplified without need for chunking](https://www.kaggle.com/code/radek1/last-20-aids))
	-   An overview of how to set up [local validation](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991)
	-   Local Validation is key to improving results -- no one worked on this so [I implemented one here](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364216)






## Notebooks

- EDA by Radek's [notebook](https://www.kaggle.com/code/radek1/eda-an-overview-of-the-full-dataset) (annotation done)
- co-visitation matrix by radek's [notebook](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364210) (annotation done)
- 
- 15x Faster Co-Visitation Matrices using RAPIDS cuDF! [discussion](https://www.kaggle.com/competitions/otto-recommender-system/discussion/365369), [notebook](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-573)
-   the last 20 AIDs are very powerful! ([original code](https://www.kaggle.com/code/ttahara/last-aid-20), [simplified without need for chunking](https://www.kaggle.com/code/radek1/last-20-aids)
-   A notebook by [@cdeotte](https://www.kaggle.com/cdeotte) building on the co-visitation matrix 👆 and demonstrating the power of training on the test data! (maybe training is too strong of a word, rather using the leak in your calculations)





## others
- [papers related](https://www.kaggle.com/competitions/otto-recommender-system/discussion/365716) by @azminetoushikwasi
- some [context](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363939#2016280) on why it is impractical to prevent people to use test data to train by @pietromaldini1


## journey

### jn: becoming a DL practitioner or anything is a marathon, in order to keep working, I need to good health, which needs good sleep, which needs to have peace with my goal and progress everyday. I think I am in good path and I should allow myself to have peace! /2022-11-17

### jn: to understand better of Kaggle discussions when I'm doing the map, I should first to extract the birdview logic from covisitation matrix model. /2022-11-17


### jn: always update the kaggle notebooks to review code and notebook logics and then directly download into fastai_notebooks folder and use fastlistnbs to access them /2022-11-18
### jn: todo (tomorrow) - update co-visitation matrix and local validation notebooks to Kaggle, use name with kaggle for fastdebug access /2022-11-18
### jn: my first gold comment is my timestamp note on recsys intro videos on Andrew Ng. A famous topic and the usefulness of the comment both are important for rating /2022-11-19
### jn: doing the map without a notebook and code is all over the places, I want to use notebook/codes to unite the pieces together, and I can just use index notebook to find them all /2022-11-19
### jn: todo tomorrow - need to work on two more notebooks https://www.kaggle.com/code/danielliao/polars-proof-of-concept-lgbm-ranker/edit and https://www.kaggle.com/code/danielliao/matrix-factorization-pytorch-merlin-dataloader/edit /2022-11-19



## symbols to use
🔥 👆💡
