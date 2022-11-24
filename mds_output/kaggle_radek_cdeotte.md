##### [How to thrive in this competition without going crazy](https://www.kaggle.com/competitions/otto-recommender-system/discussion/367503) -- 1 out of 2 important truths ❤️‍🔥

### Don't let the complexity crash you

If you are not cautious, the complexity of this competition will crush you. On the surface, we are presented with just a sequence of actions but once you start going deeper, this competition becomes EXTREMELY complex very quickly.

### The essential question to work on a kaggle competition

What to work on to improve your LB standing? 

What should be the shape and structure of the winning model?

### The first essential truth you need to follow to thrive in this competition:

Make good use of the fabulous resource that this forum is. 

In particular, [@cdeotte](https://www.kaggle.com/cdeotte) is on a mission to guide us to the light, and reading his comments is as high an ROI activity as it gets!

Here are a couple of things Chris said that are worth their weight in gold 🥇:

### How to structure the solution to this competition:

use candidate generation + Ranker to find the best 20 candidates.

[what does candidate generation and ranker do and how do they work together to give us a solution](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364721). 

You probably should opt to construct a reranker how I show [here: how co-visitation matrix and LGBM Ranker work together to be a solution](https://www.kaggle.com/code/radek1/polars-proof-of-concept-lgbm-ranker).

While my code is a good start, there are a lot of conceptual challenges that await you along the way!

Again, Chris answers some of the most pressing questions 😄

### How can a reranker beat the manual heuristic (reranking covisitation matrices using handcrafted rules)

> If the features are designed correctly, the reranker should always beat heuristics.

This is from a discussion [here: how to use co-visitation matrix to work with LGBM Ranker and how to do feature engineering to make the solution work better than co-visitation matrix on its own](https://www.kaggle.com/competitions/otto-recommender-system/discussion/366474#2032493).

You might be thinking -- "my reranker HAS some inherent limitations, even as I provide it the same data that the heuristics use, it still doesn't perform better!". But that would be WRONG!

I have been in that spot and have nearly given up completely on the reranker approach. **Essentially, I had no clue how to dig myself out of the complexity of creating a good reranker**. But that is where Chris's comment came to the rescue!

Once I had my north star that a reranker should always beat a manual heuristic, I knew where to focus to fix the problem! I had to fix the data I was feeding my reranker and I were off to the races!

### How to improve your results

Again, Chris drops the absolutely gold in [his comment here ABSOLUTE GOLD: two ways to improve CV and LB](https://www.kaggle.com/competitions/otto-recommender-system/discussion/365369#2036565) .

Do you need to invest a bunch of time into the covisitation matrices? Absolutely no! But you absolutely can adopt that reasoning in thinking what data to feed to your reranker! (though probably spending a bit more time on improving those co-visitation matrices can go a long way, like Chris suggests 🙂)

### How do I know what features to create for my reranker?

If only a person with 19 gold medals on Kaggle explained this to me, that would be great!

Wait a second! That is exactly what Chris does in his comment [here: how to create features for users and items ](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575/comments#2030893). 🔥

# Summary

The comments from Chris are extremely valuable. You will not learn how to think about solving a ML problem by reading how this or that algorithm works. That will only be a part of the solution. This tacit knowledge that Chris shares is super valuable.

Thank you, Chris! 😄

And that is truth number 1 (out of 2) that I am using to thrive in this competition. Will post the truth number 2 in a day or so!

**Would appreciate your upvote on this post if you found it useful 🙏 

Read the 2nd part here: [How to thrive in this competition without going crazy (part 2 of 2) ❤️‍🔥](https://www.kaggle.com/competitions/otto-recommender-system/discussion/367754)

**A couple of related resources that you may find useful:**

-   [🐘 the elephant in the room -- high cardinality of targets and what to do about this](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364722)
-   [📖 What are some good resources to learn about how gradient boosted tree ranking models work?](https://www.kaggle.com/competitions/otto-recommender-system/discussion/366477)
-   [💡 Can you beat static rules with a ranker model without additional features?](https://www.kaggle.com/competitions/otto-recommender-system/discussion/366474)
-   [💡 Best hyperparams for the co-visitation matrix based on HPO study with 30 runs](https://www.kaggle.com/competitions/otto-recommender-system/discussion/365153)
-   [📅 Dataset for local validation created using organizer's repository (parquet files)](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364534)
-   [💡 [polars] Proof of concept: LGBM Ranker🧪🧪🧪](https://www.kaggle.com/code/radek1/polars-proof-of-concept-lgbm-ranker)
-   [💡A robust local validation framework 🚀🚀🚀](https://www.kaggle.com/code/radek1/a-robust-local-validation-framework)
-   [📈 [EDA] An overview of the full dataset](https://www.kaggle.com/code/radek1/eda-an-overview-of-the-full-dataset)

### What will be the winning solution look like for otto competition

haha, thanks for highlighting my posts. The winning solution from Kaggle's last recommender system competition (i.e. H&M Personalized Fashion Recommendations) was a "candidate rerank" model described [here](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070). (All top solutions were "candidate rerank"). Based on the high cardinality of the items in Otto comp (i.e. 1.8 million unique items!), I also believe the winning solution in Otto comp will be a "candidate rerank" model.

An easy and quick way to explore what "candidates" are important and what "rerank features" are important is to make heuristic models like my public notebook. It is a fun way to explore the Otto data and brainstorm what information is important. Afterward, you can use the same "candidates" but then replace heuristics with GBT rerank model and you will have a very strong solution for Otto comp!

### The complexity of the solution could be
Thank you very much Chris for everything that you are sharing in this competition and the above comment! 🙂🙏

It is one thing to sort of have a vague idea of how to go about arriving at a solution and a completely different thing to have someone super experienced guide you along the way via their forum posts. It makes a huge difference, thank you so much again! 🙂

I am _somewhat_ following the overall trajectory of the H&M solution you linked to in your comment. Including monitoring the `HR` at various stages as mentioned there 😄

But I am quite surprised how complex of a software engineering project a solution to a Kaggle competition ends up being 🙂 Will continue to chip away at the solution every now and then as time permits, quite fortunate that I joined this competition so early on 😄

But can already get a feeling for how complex the pipeline is shaping up to be! Oh man. To not be crushed by the conceptual complexity of the solution and the code required to implement it is really something 🙂 Quite a fun challenge!

Thanks again for all the help that you are giving us 🙏

### How to handle the complexity 

Yes, everything gets huge. 

These large scale recommender system models require train data with millions of users where each user has thousands of candidates. It can be overwhelming. 

One approach is to create training data chunk by chunk and save to disk. Then load from disk to train your GBT reranker. 

Also during inference, we can infer chunk by chunk.

Using heuristics is a fun simpler solution which doesn't require all the train pipeline transformation and train model training. So its a nice way to start. 

But eventually, top solutions will require dealing with the massive amounts of data and feature generation and model training.

## journey
### jn: I should definitely work on @cdeotte's notebooks too /2022-11-23
### jn: as the solution will be candidate generations with heuristics like co-visitation matrix and then rerank models like XGB or LGBM Rankers, I should make sure to get my codes right on them /2022-11-23
### jn: as @cdeotte provides the full suite solution, I should start now on rerank [notebook](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575/notebook)  and compute validation [notebook](https://www.kaggle.com/code/cdeotte/compute-validation-score-cv-565?scriptVersionId=111214251) /2022-11-23
