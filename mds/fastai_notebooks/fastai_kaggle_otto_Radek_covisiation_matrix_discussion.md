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




# What is the co-visiation matrix, really?

### rd: recsys - otto - get started - what are unigram, bigram, trigram models
It is very interesting to think of modern techniques in the context of their roots.

For instance, when thinking about RNNs we should consider unigram, bigram, and trigram models.

What were they?

They estimated the probability of a word given the words that came before.

We see "Radek is a **_____**".

In a trigram model, we would consider the chain "Radek", "is", and "a" and could find the most likely word to come next.

The easiest way would be to count the occurrences of "Radek is a" and the words that came after and pick the one most common.

### rd: recsys - otto - get started - How RNN improve on uni-bi-trigram models
So why RNNs are such a great improvement?

Well, there are not that many "Radek is a **_____**" in most text corpora!

With RNNs (or word2vec) we could operate on embeddings and look at many more words in sequence.

Instead of looking at only "Radek is a " we can use "Radek was an **_____**", "Tommy is a **______**" as examples to learn from, etc.

Embeddings take values of a variable of high cardinality and project them to a representation that ideally captures similarities.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F83267%2F70e2a7f1b9621cd0afb2b815a7b3e01b%2Fyou_shall_know_a_word.png?generation=1668122115555809&alt=media)

### rd: recsys - otto - get started - How co-visitation matrix relate to embeddings?
So how does this relate to the co-visitation matrix?

A co-visitation matrix counts the co-occurrence of two actions in close proximity.

If a user bought A and shortly after bought B, we store these values together.

We calculate counts and use them to estimate the probability of future actions based on recent history.

### rd: recsys - otto - get started - limitations and resembles of co-visitation matrix
It is quite important to understand what is happening in the co-visitation matrix approach…

Since it suffers from the same issues as our trigram example!

Plus what does the co-visitation matrix resemble?

You are right, it is akin to doing Matrix Factorization by counting!

It is really fun that this competition exposed this heuristic (the co-visitation matrix) that I have not been aware of before! 🙏

And if you'd like to learn more about the co-visitation matrix and jump straight into the competition, you might find these two notebooks useful. Please upvote if you do!

-   [co-visitation matrix - simplified, imprvd logic 🔥](https://www.kaggle.com/code/radek1/co-visitation-matrix-simplified-imprvd-logic)
-   [💡 [Howto] Full dataset as parquet/csv files](https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files)
-   [💡A robust local validation framework 🚀🚀🚀](https://www.kaggle.com/code/radek1/a-robust-local-validation-framework)

Thank you! 🙌

---

## Discussion

### rd: recsys - otto - get started - build word2vec as a resemblence of co-visitation matrix
[Chris Deotte](https://www.kaggle.com/cdeotte) • (2nd in this Competition) 

Training Word2Vec would be very helpful. That would produce embeddings such that similar items (with co-visitation) would have distance similar embeddings. It would help us understand what these anonymous item ids represent.

Then we could do lots of EDA like how do the items that users purchase over time change? How do items purchased in the morning differ from items purchased in the evening. Etc etc. For each of these analysis, we could plot 2D pictures with clusters of dots (using their Word2Vec embedding 2D projection)

[Chris Deotte](https://www.kaggle.com/cdeotte) • (2nd in this Competition) 

Co-visitation matrices are the natural statistical approach. It is a way to compute conditional probabilities. Given a user has interacted with item A, it computes the most likely future items that this user will interact with.

And we can make many more specific variants such as given a user has **ordered** item A, what is the most likely item that this user will **click, cart, order** etc. Or given a user has visited this item **in the morning**, what is the most likely item that this user will **order**. Etc etc.

Of course if we train a reranker model it will learn all these varieties of conditional probabilities on its own.

That is some amazing information [@cdeotte](https://www.kaggle.com/cdeotte), as always! 🙂 Thank you so very much for sharing all this with us 🙏

---

[Sinan Calisir](https://www.kaggle.com/snnclsr) • (124th in this Competition) 

How funny that I was playing with the Word2Vec and its variants and then saw this post :D


Yeah, it would be really interesting to see what word2vec can do in this competition 🙂 Might be it could be quite nice for candidate generation or some sort of similarity scoring for reranking 🤔

Anyhow, cool that we were both circulating around the same topic! 🙌

Yes, exactly! I was also thinking of it for the candidate generation. 🤜🤛

Word2Vec would also be great for EDA. It (with 2D projection via UMAP or TSNE) would help us visualize these anonymous items.


### jn: started the notebook [co-visitation matrix - simplified, imprvd logic 🔥](https://www.kaggle.com/code/radek1/co-visitation-matrix-simplified-imprvd-logic) and half way through. At first the notebook was intimidating even though Radek made the codes easy to follow, and after experiment line by line I can taste the refresh and interesting flavor from this notebook /2022-11-11

### jn: in the discussion word2vec is frequently mentioned, I will at some point [search embedding](https://forums.fast.ai/t/exploring-fastai-with-excel-and-python/97426/4?u=daniel) in fastai lectures (videos) for related content (search [colab](https://colab.research.google.com/drive/102_vWdSfRxw8SI61CED1B9uVE2cJxpCC?usp=sharing)) /2022-11-11

### jn: another useful notebook study group [repo](https://github.com/jcatanza/Fastai-A-Code-First-Introduction-To-Natural-Language-Processing-TWiML-Study-Group) on NLP course by Rachel /2022-11-12

### jn: read radek explaining what is co-visitation matrix on [twitter](https://twitter.com/radekosmulski/status/1590909701797007360) for a second time (first time on Kaggle), it certainly makes more sense to me after last night's work on the notebook. The next notebook to explore is train-validation split notebook recommended [here](https://twitter.com/radekosmulski/status/1590909730469294080?s=20&t=hTs07NKjbCWpz5sAXxJLwg) /2022-11-11

### jn: radek's summary on Xavier's lecture 1-2 on [twitter](https://twitter.com/radekosmulski/status/1565716248083566592) /2022-11-12

### jn: I don't believe I will succeed as a DL practitioner no matter how great examples or paths Jeremy, Radek etc have set for me. But because it's such as a great pleasure to witness these amazing human beings' stories, I will try my best to follow their paths in particular Radek's anyway despite how much I disbelieve I could make it /2022-11-12

### jn: Radek and Jeremy said it numerously to share your work publicly. I think one way I feel good to share is to tweet a list of what I learnt from each of Radek's kaggle notebooks. /2022-11-12
