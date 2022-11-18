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


post by @ravishah1 from [otto-recommender-system](https://www.kaggle.com/competitions/otto-recommender-system)

# The Concept

If you didn't realize, this dataset it quite large. In fact the train dataset consists of:

-   12,899,779 sessions
-   1,855,603 items
-   216,716,096 events
-   194,720,954 clicks
-   16,896,191 carts
-   5,098,951 orders

[more info about the dataset](https://github.com/otto-de/recsys-dataset)

### rd: recsys - otto - big pic - what is candidate generation
So as you might expect, just dumping the raw data into a model is not going to very very effective or efficient. That is why sites like YouTube with billions of items of content use a technique known as candidates generation. This technique has been applied in previous recommendation system challenges on Kaggle, and I think it can be applied here to.

This image should give you an overview of the process:

![[recsys_overview.png]]


Here are some criteria you can use to select you candidates:

-   previously purchased items
-   repurchased items
-   overall most popular items
-   similar items based on some sort of clustering technique
-   similar items based on something such as a co-visitation matrix

After using techniques like these, you should have much fewer items for each session, so you should be able to input these into a ranker model.

### rd: recsys - otto - big pic - what is ranking

Now that you have your candidates, you need to generate features for your items. This is kind of tricky in this competition because we only have the article id (we don't have the product name, price, or other useful info). Thus, your features will probably be the reason the item was selected as a candidate. You can also make numerical features related to the candidates generation such as the number of times a item was viewed or repurchased, etc.

Now that you have feature engineered items, you want to feed your options to a ranker models. Examples of ranker models include:

-   [LGBMRanker](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html)
-   [XGBRanker](https://medium.com/predictly-on-tech/learning-to-rank-using-xgboost-83de0166229d)
-   [Ranking with sklearn](https://towardsdatascience.com/learning-to-rank-with-python-scikit-learn-327a5cfd81f)
-   [Neural Network Ranker](https://maroo.cs.umass.edu/getpdf.php?id=1373)

You can then take your highest ranked items and submit them as your recommendations.

### jn: when I saw this big picture, I can feel there are a lot of new stuff to learn here. It is tempting and can lead to many hours go without result. However, I will follow Radek's advice to stay with the Kaggle discussion and kernels, make sure I make the most out of them first before I explore new waters /2022-11-13

### jn: I have finished the annotation of Radek's co-visitation matrix simplified notebook, and have a good feel of what does co-visitation matrix do here. But still there is more work on it can be learnt, e.g., all the previous notebooks [1st](https://www.kaggle.com/code/vslaykovsky/co-visitation-matrix) [2nd](https://www.kaggle.com/code/cdeotte/test-data-leak-lb-boost), [3nd](https://www.kaggle.com/code/ingvarasgalinskas/item-type-vs-multiple-clicks-vs-latest-items) which Radek based to built the simplifed notebook /2022-11-13

### jn: I need to finish up the annotation on Radek's EDA notebook properly /2022-11-13

### jn: following Radek's advice, I realize that there are also other great kagglers to learn from like [Chris Deotte](https://www.kaggle.com/code/cdeotte/test-data-leak-lb-boost) /2022-11-13
