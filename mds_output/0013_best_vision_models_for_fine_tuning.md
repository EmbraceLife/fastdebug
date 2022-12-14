# 0013_best_vision_models_for_fine_tuning
---
skip_exec: true
---
## Introduction

### best vision models for training from scratch vs for fine tuning

In a recent notebook I tried to answer the question "[Which image models are best?](https://www.kaggle.com/code/jhoward/which-image-models-are-best)" This showed which models in Ross Wightman's [PyTorch Image Models](https://timm.fast.ai/) (*timm*) were the fastest and most accurate for training from scratch with Imagenet.

However, this is not what most of us use models for. Most of us fine-tune pretrained models. Therefore, what most of us really want to know is which models are the fastest and most accurate for fine-tuning. However, this analysis has not, to my knowledge, previously existed.

Therefore I teamed up with [Thomas Capelle](https://tcapelle.github.io/about/) of [Weights and Biases](https://wandb.ai/) to answer this question. In this notebook, I present our results.

## The analysis

### how to evaluate or compare models for fine tuning

There are two key dimensions on which datasets can vary when it comes to how well they fine-tune a model:

1. How similar they are to the pre-trained model's dataset
2. How large they are.

Therefore, we decided to test on two datasets that were very different on both of these axes. We tested pre-trained models that were trained on Imagenet, and tested fine-tuning on two different datasets:

1. The [Oxford IIT-Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which is very similar to Imagenet. Imagenet contains many pictures of animals, and each picture is a photo in which the animal is the main subject. IIT-Pet contains nearly 15,000 images, that are also of this type.
2. The [Kaggle Planet](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data) sample contains 1,000 satellite images of Earth. There are no images of this kind in Imagenet.

So these two datasets are of very different sizes, and very different in terms of their similarity to Imagenet. Furthermore, they have different types of labels - Planet is a multi-label problem, whereas IIT-Pet is a single label problem.

### how to use Weights and Biases with fastai

To test the fine-tuning accuracy of different models, Thomas put together [this script](https://github.com/tcapelle/fastai_timm/blob/main/fine_tune.py). The basic script contains the standard 4 lines of code needed for fastai image recognition models, plus some code to handle various configuration options, such as learning rate and batch size. It was particularly easy to handle in fastai since fastai supports all timm models directly.

Then, to allow us to easily try different configuration options, Thomas created Weights and Biases (*wandb*) YAML files such as [this one](https://github.com/tcapelle/fastai_timm/blob/main/sweep_planets_lr.yaml). This takes advantage of the convenient [wandb "sweeps"](https://wandb.ai/site/sweeps) feature which tries a range of different levels of a model input and tracks the results.

wandb makes it really easy for a group of people to run these kinds of analyses on whatever GPUs they have access to. When you create a sweep using the command-line wandb client, it gives you a command to run to have a computer run experiments for the project. You run that same command on each computer where you want to run experiments. The wandb client automatically ensures that each computer runs different parts of the sweep, and has each on report back its results to the wandb server. You can look at the progress in the wandb web GUI at any time during or after the run. I've got three GPUs in my PC at home, so I ran three copies of the client, with each using a different GPU. Thomas also ran the client on a [Paperspace Gradient](https://gradient.run/notebooks) server.

I liked this approach because I could start and stop the clients any time I wanted, and wandb would automatically handle keeping all the results in sync. When I restarted a client, it would automatically grab from the server whatever the next set of sweep settings were needed. Furthermore, the integration in fastai is really exceptional, thanks particularly to [Boris Dayma](https://github.com/borisdayma), who worked tirelessly to ensure that wandb automatically tracks every aspect of all fastai data processing, model architectures, and optimisation.

## Hyperparameters

### how to decide hyperparameters to create all the possible and meaningful models for testing

We decided to try out all the timm models which had reasonable performance on timm, and which are capable of working with 224x224 px images. We ended up with a list of 86 models and variants to try.

Our first step was to find a good set of hyper-parameters for each model variant and for each dataset. Our experience at fast.ai has been that there's generally not much difference between models and datasets in terms of what hyperparameter settings work well -- and that experience was repeated in this project. Based on some initial sweeps across a smaller number of representative models, on which we found little variation in optimal hyperparameters, in our final sweep we included all combinations of the following options:

- Learning rate (AdamW): 0.008 and 0.02
- Resize method: [Squish](https://docs.fast.ai/vision.augment.html#Resize)
- Pooling type: [Concat](https://docs.fast.ai/layers.html#AdaptiveConcatPool2d) and Average Pooling

For other parameters, we used defaults that we've previously found at fast.ai to be reliable across a range of models and datasets (see the fastai docs for details).

## Analysis

### how to analyse the sweep results from W&B

Let's take a look at the data. I've put a CSV of the results into a gist:


```
from fastai.vision.all import *
import plotly.express as px

url = 'https://gist.githubusercontent.com/jph00/959aaf8695e723246b5e21f3cd5deb02/raw/sweep.csv'
```

For each model variant and dataset, for each hyperparameter setting, we did three runs. For the final sweep, we just used the hyperparameter settings listed above.

For each model variant and dataset, I create a group with the minimum error and fit time, and GPU memory use if used. I use the minimum because there might be some reason that a particular run didn't do so well (e.g. maybe there was some resource contention), and I'm mainly interested in knowing what the best case results for a model can be.

I create a "score" which, somewhat arbitrarily combines the accuracy and speed into a single number. I tried a few options until I came up with something that closely matched my own opinions about the tradeoffs between the two. (Feel free of course to fork this notebook and adjust how that's calculated.)


```
df = pd.read_csv(url)
df['family'] = df.model_name.str.extract('^([a-z]+?(?:v2)?)(?:\d|_|$)')
df.loc[df.family=='swinv2', 'family'] = 'swin'
pt_all = df.pivot_table(values=['error_rate','fit_time','GPU_mem'], index=['dataset', 'family', 'model_name'],
                        aggfunc=np.min).reset_index()
pt_all['score'] = pt_all.error_rate*(pt_all.fit_time+80)
```

### IIT Pet

Here's the top 15 models on the IIT Pet dataset, ordered by score:


```
pt = pt_all[pt_all.dataset=='pets'].sort_values('score').reset_index(drop=True)
pt.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>family</th>
      <th>model_name</th>
      <th>GPU_mem</th>
      <th>error_rate</th>
      <th>fit_time</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pets</td>
      <td>convnext</td>
      <td>convnext_tiny_in22k</td>
      <td>2.660156</td>
      <td>0.044655</td>
      <td>94.557838</td>
      <td>7.794874</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pets</td>
      <td>swin</td>
      <td>swin_s3_tiny_224</td>
      <td>3.126953</td>
      <td>0.041949</td>
      <td>112.282200</td>
      <td>8.065961</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pets</td>
      <td>convnext</td>
      <td>convnext_tiny</td>
      <td>2.660156</td>
      <td>0.047361</td>
      <td>92.761599</td>
      <td>8.182216</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pets</td>
      <td>vit</td>
      <td>vit_small_r26_s32_224</td>
      <td>3.367188</td>
      <td>0.045332</td>
      <td>103.240067</td>
      <td>8.306554</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pets</td>
      <td>mobilevit</td>
      <td>mobilevit_s</td>
      <td>2.781250</td>
      <td>0.046685</td>
      <td>100.770686</td>
      <td>8.439222</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pets</td>
      <td>resnetv2</td>
      <td>resnetv2_50x1_bit_distilled</td>
      <td>3.892578</td>
      <td>0.047361</td>
      <td>105.952172</td>
      <td>8.806939</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pets</td>
      <td>vit</td>
      <td>vit_small_patch16_224</td>
      <td>2.111328</td>
      <td>0.054804</td>
      <td>80.739517</td>
      <td>8.809135</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pets</td>
      <td>swin</td>
      <td>swin_tiny_patch4_window7_224</td>
      <td>2.796875</td>
      <td>0.048038</td>
      <td>105.797015</td>
      <td>8.925296</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pets</td>
      <td>swin</td>
      <td>swinv2_cr_tiny_ns_224</td>
      <td>3.302734</td>
      <td>0.042625</td>
      <td>129.435368</td>
      <td>8.927222</td>
    </tr>
    <tr>
      <th>9</th>
      <td>pets</td>
      <td>resnetrs</td>
      <td>resnetrs50</td>
      <td>2.419922</td>
      <td>0.047361</td>
      <td>109.549398</td>
      <td>8.977309</td>
    </tr>
    <tr>
      <th>10</th>
      <td>pets</td>
      <td>levit</td>
      <td>levit_384</td>
      <td>1.699219</td>
      <td>0.054127</td>
      <td>86.199098</td>
      <td>8.995895</td>
    </tr>
    <tr>
      <th>11</th>
      <td>pets</td>
      <td>resnet</td>
      <td>resnet26d</td>
      <td>1.412109</td>
      <td>0.060216</td>
      <td>69.395598</td>
      <td>8.996078</td>
    </tr>
    <tr>
      <th>12</th>
      <td>pets</td>
      <td>convnext</td>
      <td>convnext_tiny_hnf</td>
      <td>2.970703</td>
      <td>0.049391</td>
      <td>103.014163</td>
      <td>9.039269</td>
    </tr>
    <tr>
      <th>13</th>
      <td>pets</td>
      <td>regnety</td>
      <td>regnety_006</td>
      <td>0.914062</td>
      <td>0.052097</td>
      <td>93.912189</td>
      <td>9.060380</td>
    </tr>
    <tr>
      <th>14</th>
      <td>pets</td>
      <td>levit</td>
      <td>levit_256</td>
      <td>1.031250</td>
      <td>0.056157</td>
      <td>82.682410</td>
      <td>9.135755</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, the [convnext](https://arxiv.org/abs/2201.03545), [swin](https://arxiv.org/abs/2103.14030), and [vit](https://arxiv.org/abs/2010.11929) families are fairly dominent. The excellent showing of `convnext_tiny` matches my view that we should think of this as our default baseline for image recognition today. It's fast, accurate, and not too much of a memory hog. (And according to Ross Wightman, it could be even faster if NVIDIA and PyTorch make some changes to better optimise the operations it relies on!)

`vit_small_patch16` is also a good option -- it's faster and leaner on memory than `convnext_tiny`, although there is some performance cost too.

Interestingly, resnets are still a great option -- especially the [`resnet26d`](https://arxiv.org/abs/1812.01187) variant, which is the fastest in our top 15.

Here's a quick visual representation of the seven model families which look best in the above analysis (the "fit lines" are just there to help visually show where the different families are -- they don't necessarily actually follow a linear fit):


```
w,h = 900,700
faves = ['vit','convnext','resnet','levit', 'regnetx', 'swin']
pt2 = pt[pt.family.isin(faves)]
px.scatter(pt2, width=w, height=h, x='fit_time', y='error_rate', color='family', hover_name='model_name', trendline="ols",)
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.12.1.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




<div>                            <div id="e9abb41c-7f0b-4782-b53f-bc1deb85b090" class="plotly-graph-div" style="height:700px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("e9abb41c-7f0b-4782-b53f-bc1deb85b090")) {                    Plotly.newPlot(                        "e9abb41c-7f0b-4782-b53f-bc1deb85b090",                        [{"hovertemplate":"<b>%{hovertext}</b><br><br>family=convnext<br>fit_time=%{x}<br>error_rate=%{y}<extra></extra>","hovertext":["convnext_tiny_in22k","convnext_tiny","convnext_tiny_hnf","convnext_small","convnext_small_in22k","convnext_base_in22k","convnext_base","convnext_large_in22k"],"legendgroup":"convnext","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"convnext","orientation":"v","showlegend":true,"x":[94.55783775995953,92.76159947301494,103.01416272699134,140.69350739900256,141.08317097002873,178.8063996139681,177.59854573803025,286.3457520679804],"xaxis":"x","y":[0.0446549654006958,0.0473613142967224,0.0493910908699035,0.0453315377235411,0.0480378866195679,0.0412719845771789,0.0419485569000244,0.0392422080039978],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>OLS trendline</b><br>error_rate = -4.58499e-05 * fit_time + 0.0516176<br>R<sup>2</sup>=0.676602<br><br>family=convnext<br>fit_time=%{x}<br>error_rate=%{y} <b>(trend)</b><extra></extra>","legendgroup":"convnext","marker":{"color":"#636efa","symbol":"circle"},"mode":"lines","name":"convnext","showlegend":false,"x":[92.76159947301494,94.55783775995953,103.01416272699134,140.69350739900256,141.08317097002873,177.59854573803025,178.8063996139681,286.3457520679804],"xaxis":"x","y":[0.047364491670095224,0.04728213426653234,0.0468944123406439,0.04516681694436925,0.04514895089608044,0.043474723440917115,0.04341934342267087,0.03848867141032261],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>%{hovertext}</b><br><br>family=swin<br>fit_time=%{x}<br>error_rate=%{y}<extra></extra>","hovertext":["swin_s3_tiny_224","swin_tiny_patch4_window7_224","swinv2_cr_tiny_ns_224","swin_small_patch4_window7_224","swin_s3_small_224","swin_base_patch4_window7_224","swinv2_cr_small_ns_224","swinv2_cr_small_224","swin_base_patch4_window7_224_in22k","swin_s3_base_224","swin_large_patch4_window7_224","swin_large_patch4_window7_224_in22k"],"legendgroup":"swin","marker":{"color":"#EF553B","symbol":"circle"},"mode":"markers","name":"swin","orientation":"v","showlegend":true,"x":[112.28219997999258,105.7970150890178,129.43536768300692,162.27144744596444,197.57776981400093,201.50697606999893,201.5679091179627,200.8026947770268,201.5516522770049,248.17078953096643,321.7958480840316,321.60895527602406],"xaxis":"x","y":[0.0419485569000244,0.0480378866195679,0.0426251888275146,0.0392422080039978,0.0378890633583067,0.0405954122543336,0.0412719845771789,0.0433017611503599,0.0453315377235411,0.0405954122543336,0.0372124314308167,0.0446549654006958],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>OLS trendline</b><br>error_rate = -1.49081e-05 * fit_time + 0.0448793<br>R<sup>2</sup>=0.112560<br><br>family=swin<br>fit_time=%{x}<br>error_rate=%{y} <b>(trend)</b><extra></extra>","legendgroup":"swin","marker":{"color":"#EF553B","symbol":"circle"},"mode":"lines","name":"swin","showlegend":false,"x":[105.7970150890178,112.28219997999258,129.43536768300692,162.27144744596444,197.57776981400093,200.8026947770268,201.50697606999893,201.5516522770049,201.5679091179627,248.17078953096643,321.60895527602406,321.7958480840316],"xaxis":"x","y":[0.043302018321227195,0.043205336346500756,0.04294961470490975,0.04246009018292185,0.041933738964837,0.04188566136658628,0.04187516185002474,0.04187449581135518,0.041874253452268584,0.04117949168648768,0.04008466601784702,0.04008187979570469],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>%{hovertext}</b><br><br>family=vit<br>fit_time=%{x}<br>error_rate=%{y}<extra></extra>","hovertext":["vit_small_r26_s32_224","vit_small_patch16_224","vit_base_patch32_224","vit_tiny_patch16_224","vit_small_patch32_224","vit_base_patch16_224_miil","vit_base_patch16_224_sam","vit_base_patch32_224_sam","vit_tiny_r_s16_p8_224","vit_base_patch16_224"],"legendgroup":"vit","marker":{"color":"#00cc96","symbol":"circle"},"mode":"markers","name":"vit","orientation":"v","showlegend":true,"x":[103.24006690399256,80.73951682000188,72.87590981600806,65.67020213900832,68.4788694360177,147.4005719649722,150.23862834600732,72.32148186000995,68.8997921749833,150.4993650689721],"xaxis":"x","y":[0.0453315377235411,0.0548037886619567,0.0608931183815002,0.0642760396003724,0.065629243850708,0.0446549654006958,0.0460081100463868,0.0771312713623048,0.0791610479354858,0.0541272163391112],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>OLS trendline</b><br>error_rate = -0.000252624 * fit_time + 0.083968<br>R<sup>2</sup>=0.555792<br><br>family=vit<br>fit_time=%{x}<br>error_rate=%{y} <b>(trend)</b><extra></extra>","legendgroup":"vit","marker":{"color":"#00cc96","symbol":"circle"},"mode":"lines","name":"vit","showlegend":false,"x":[65.67020213900832,68.4788694360177,68.8997921749833,72.32148186000995,72.87590981600806,80.73951682000188,103.24006690399256,147.4005719649722,150.23862834600732,150.4993650689721],"xaxis":"x","y":[0.0673781154752632,0.06666857931917934,0.06656224422476216,0.06569784403431927,0.06555778234700473,0.06357124820232019,0.05788707413394033,0.04673108031258135,0.046014119774929516,0.045948251477762624],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>%{hovertext}</b><br><br>family=levit<br>fit_time=%{x}<br>error_rate=%{y}<extra></extra>","hovertext":["levit_384","levit_256","levit_192","levit_128","levit_128s"],"legendgroup":"levit","marker":{"color":"#ab63fa","symbol":"circle"},"mode":"markers","name":"levit","orientation":"v","showlegend":true,"x":[86.19909829896642,82.68240976100788,82.3857870750362,82.81964537699241,71.38045354594942],"xaxis":"x","y":[0.0541272163391112,0.0561569929122923,0.0608931183815002,0.07780784368515,0.0926928520202635],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>OLS trendline</b><br>error_rate = -0.0025218 * fit_time + 0.272838<br>R<sup>2</sup>=0.745347<br><br>family=levit<br>fit_time=%{x}<br>error_rate=%{y} <b>(trend)</b><extra></extra>","legendgroup":"levit","marker":{"color":"#ab63fa","symbol":"circle"},"mode":"lines","name":"levit","showlegend":false,"x":[71.38045354594942,82.3857870750362,82.68240976100788,82.81964537699241,86.19909829896642],"xaxis":"x","y":[0.09282995574023578,0.06507665584038097,0.0643286314084894,0.0639825500110203,0.05546023033819089],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>%{hovertext}</b><br><br>family=resnet<br>fit_time=%{x}<br>error_rate=%{y}<extra></extra>","hovertext":["resnet26d","resnet50_gn","resnet50d","resnet26","resnet34","resnet18","resnet34d","resnet18d","resnet50","resnet101","resnet152"],"legendgroup":"resnet","marker":{"color":"#FFA15A","symbol":"circle"},"mode":"markers","name":"resnet","orientation":"v","showlegend":true,"x":[69.39559792500222,111.03789425600552,92.989515033958,64.3980961269699,66.93234487896552,53.40542413300136,71.63126915198518,53.41355074098101,86.83967336296337,129.06840408500284,178.05023195099784],"xaxis":"x","y":[0.06021648645401,0.0493910908699035,0.0554803609848022,0.0676590204238891,0.0703653693199157,0.0784844160079956,0.0703653693199157,0.0859269499778747,0.0723951458930968,0.0629228949546814,0.0527740120887756],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>OLS trendline</b><br>error_rate = -0.000207342 * fit_time + 0.0844171<br>R<sup>2</sup>=0.497991<br><br>family=resnet<br>fit_time=%{x}<br>error_rate=%{y} <b>(trend)</b><extra></extra>","legendgroup":"resnet","marker":{"color":"#FFA15A","symbol":"circle"},"mode":"lines","name":"resnet","showlegend":false,"x":[53.40542413300136,53.41355074098101,64.3980961269699,66.93234487896552,69.39559792500222,71.63126915198518,86.83967336296337,92.989515033958,111.03789425600552,129.06840408500284,178.05023195099784],"xaxis":"x","y":[0.07334389821553083,0.0733422132262461,0.0710646527336547,0.07053919586374122,0.07002845940412729,0.06956491027377415,0.06641156533603405,0.06513644325010637,0.061394251468198394,0.057655764766674514,0.04749976175677307],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>%{hovertext}</b><br><br>family=regnetx<br>fit_time=%{x}<br>error_rate=%{y}<extra></extra>","hovertext":["regnetx_032","regnetx_016","regnetx_080","regnetx_040","regnetx_006","regnetx_064","regnetx_008","regnetx_120","regnetx_004","regnetx_002","regnetx_160"],"legendgroup":"regnetx","marker":{"color":"#19d3f3","symbol":"circle"},"mode":"markers","name":"regnetx","orientation":"v","showlegend":true,"x":[123.79527464497367,88.65808749297867,130.6214182979893,127.91163991601206,78.59255497000413,150.43287114502164,81.93718523101415,194.2259087840212,90.86267638701248,69.06612294499064,217.3980512759881],"xaxis":"x","y":[0.0473613142967224,0.0595399141311645,0.0514208674430848,0.0541272163391112,0.0710419416427612,0.0493910908699035,0.0703653693199157,0.0439783334732055,0.0764546394348144,0.0926928520202635,0.0487144589424132],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>OLS trendline</b><br>error_rate = -0.000245533 * fit_time + 0.0906744<br>R<sup>2</sup>=0.610272<br><br>family=regnetx<br>fit_time=%{x}<br>error_rate=%{y} <b>(trend)</b><extra></extra>","legendgroup":"regnetx","marker":{"color":"#19d3f3","symbol":"circle"},"mode":"lines","name":"regnetx","showlegend":false,"x":[69.06612294499064,78.59255497000413,81.93718523101415,88.65808749297867,90.86267638701248,123.79527464497367,127.91163991601206,130.6214182979893,150.43287114502164,194.2259087840212,217.3980512759881],"xaxis":"x","y":[0.07371631286388115,0.0713772558303137,0.07055603746480638,0.06890583163050346,0.06836453147267157,0.06027847938237425,0.059267774313081376,0.05860243327384125,0.053738060337009695,0.04298540788099835,0.037295873463878956],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"fit_time"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"error_rate"}},"legend":{"title":{"text":"family"},"tracegroupgap":0},"margin":{"t":60},"height":700,"width":900},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('e9abb41c-7f0b-4782-b53f-bc1deb85b090');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


This chart shows that there's a big drop-off in performance towards the far left. It seems like there's a big compromise if we want the fastest possible model. It also seems that the best models in terms of accuracy, convnext and swin, aren't able to make great use of the larger capacity of larger models. So an ensemble of smaller models may be effective in some situations.

Note that `vit` doesn't include any larger/slower models, since they only work with larger images. We would recommend trying larger models on your dataset if you have larger images and the resources to handle them.

I particularly like using fast and small models, since I wanted to be able to iterate rapidly to try lots of ideas (see [this notebook](https://www.kaggle.com/code/jhoward/iterate-like-a-grandmaster) for more on this). Here's the top models (based on accuracy) that are smaller and faster than the median model:


```
pt.query("(GPU_mem<2.7) & (fit_time<110)").sort_values("error_rate").head(15).reset_index(drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>family</th>
      <th>model_name</th>
      <th>GPU_mem</th>
      <th>error_rate</th>
      <th>fit_time</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pets</td>
      <td>convnext</td>
      <td>convnext_tiny_in22k</td>
      <td>2.660156</td>
      <td>0.044655</td>
      <td>94.557838</td>
      <td>7.794874</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pets</td>
      <td>convnext</td>
      <td>convnext_tiny</td>
      <td>2.660156</td>
      <td>0.047361</td>
      <td>92.761599</td>
      <td>8.182216</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pets</td>
      <td>resnetrs</td>
      <td>resnetrs50</td>
      <td>2.419922</td>
      <td>0.047361</td>
      <td>109.549398</td>
      <td>8.977309</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pets</td>
      <td>regnety</td>
      <td>regnety_006</td>
      <td>0.914062</td>
      <td>0.052097</td>
      <td>93.912189</td>
      <td>9.060380</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pets</td>
      <td>levit</td>
      <td>levit_384</td>
      <td>1.699219</td>
      <td>0.054127</td>
      <td>86.199098</td>
      <td>8.995895</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pets</td>
      <td>vit</td>
      <td>vit_small_patch16_224</td>
      <td>2.111328</td>
      <td>0.054804</td>
      <td>80.739517</td>
      <td>8.809135</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pets</td>
      <td>resnet</td>
      <td>resnet50d</td>
      <td>2.037109</td>
      <td>0.055480</td>
      <td>92.989515</td>
      <td>9.597521</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pets</td>
      <td>levit</td>
      <td>levit_256</td>
      <td>1.031250</td>
      <td>0.056157</td>
      <td>82.682410</td>
      <td>9.135755</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pets</td>
      <td>regnetx</td>
      <td>regnetx_016</td>
      <td>1.369141</td>
      <td>0.059540</td>
      <td>88.658087</td>
      <td>10.041888</td>
    </tr>
    <tr>
      <th>9</th>
      <td>pets</td>
      <td>resnet</td>
      <td>resnet26d</td>
      <td>1.412109</td>
      <td>0.060216</td>
      <td>69.395598</td>
      <td>8.996078</td>
    </tr>
    <tr>
      <th>10</th>
      <td>pets</td>
      <td>levit</td>
      <td>levit_192</td>
      <td>0.781250</td>
      <td>0.060893</td>
      <td>82.385787</td>
      <td>9.888177</td>
    </tr>
    <tr>
      <th>11</th>
      <td>pets</td>
      <td>resnetblur</td>
      <td>resnetblur50</td>
      <td>2.195312</td>
      <td>0.061570</td>
      <td>96.008735</td>
      <td>10.836803</td>
    </tr>
    <tr>
      <th>12</th>
      <td>pets</td>
      <td>mobilevit</td>
      <td>mobilevit_xs</td>
      <td>2.349609</td>
      <td>0.062923</td>
      <td>98.758011</td>
      <td>11.247972</td>
    </tr>
    <tr>
      <th>13</th>
      <td>pets</td>
      <td>vit</td>
      <td>vit_tiny_patch16_224</td>
      <td>1.074219</td>
      <td>0.064276</td>
      <td>65.670202</td>
      <td>9.363104</td>
    </tr>
    <tr>
      <th>14</th>
      <td>pets</td>
      <td>regnety</td>
      <td>regnety_008</td>
      <td>1.044922</td>
      <td>0.064953</td>
      <td>94.741903</td>
      <td>11.349943</td>
    </tr>
  </tbody>
</table>
</div>



...and here's the top 15 models that are the very fastest and most memory efficient:


```
pt.query("(GPU_mem<1.6) & (fit_time<90)").sort_values("error_rate").head(15).reset_index(drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>family</th>
      <th>model_name</th>
      <th>GPU_mem</th>
      <th>error_rate</th>
      <th>fit_time</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pets</td>
      <td>levit</td>
      <td>levit_256</td>
      <td>1.031250</td>
      <td>0.056157</td>
      <td>82.682410</td>
      <td>9.135755</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pets</td>
      <td>regnetx</td>
      <td>regnetx_016</td>
      <td>1.369141</td>
      <td>0.059540</td>
      <td>88.658087</td>
      <td>10.041888</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pets</td>
      <td>resnet</td>
      <td>resnet26d</td>
      <td>1.412109</td>
      <td>0.060216</td>
      <td>69.395598</td>
      <td>8.996078</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pets</td>
      <td>levit</td>
      <td>levit_192</td>
      <td>0.781250</td>
      <td>0.060893</td>
      <td>82.385787</td>
      <td>9.888177</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pets</td>
      <td>vit</td>
      <td>vit_tiny_patch16_224</td>
      <td>1.074219</td>
      <td>0.064276</td>
      <td>65.670202</td>
      <td>9.363104</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pets</td>
      <td>vit</td>
      <td>vit_small_patch32_224</td>
      <td>0.775391</td>
      <td>0.065629</td>
      <td>68.478869</td>
      <td>9.744556</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pets</td>
      <td>efficientnet</td>
      <td>efficientnet_es_pruned</td>
      <td>1.507812</td>
      <td>0.066306</td>
      <td>69.601242</td>
      <td>9.919432</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pets</td>
      <td>efficientnet</td>
      <td>efficientnet_es</td>
      <td>1.507812</td>
      <td>0.066306</td>
      <td>69.822634</td>
      <td>9.934112</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pets</td>
      <td>resnet</td>
      <td>resnet26</td>
      <td>1.291016</td>
      <td>0.067659</td>
      <td>64.398096</td>
      <td>9.769834</td>
    </tr>
    <tr>
      <th>9</th>
      <td>pets</td>
      <td>resnet</td>
      <td>resnet34</td>
      <td>0.951172</td>
      <td>0.070365</td>
      <td>66.932345</td>
      <td>10.338949</td>
    </tr>
    <tr>
      <th>10</th>
      <td>pets</td>
      <td>resnet</td>
      <td>resnet34d</td>
      <td>1.056641</td>
      <td>0.070365</td>
      <td>71.631269</td>
      <td>10.669590</td>
    </tr>
    <tr>
      <th>11</th>
      <td>pets</td>
      <td>regnetx</td>
      <td>regnetx_008</td>
      <td>0.976562</td>
      <td>0.070365</td>
      <td>81.937185</td>
      <td>11.394770</td>
    </tr>
    <tr>
      <th>12</th>
      <td>pets</td>
      <td>regnetx</td>
      <td>regnetx_006</td>
      <td>0.730469</td>
      <td>0.071042</td>
      <td>78.592555</td>
      <td>11.266723</td>
    </tr>
    <tr>
      <th>13</th>
      <td>pets</td>
      <td>mobilevit</td>
      <td>mobilevit_xxs</td>
      <td>1.152344</td>
      <td>0.073072</td>
      <td>88.449456</td>
      <td>12.308891</td>
    </tr>
    <tr>
      <th>14</th>
      <td>pets</td>
      <td>levit</td>
      <td>levit_128</td>
      <td>0.650391</td>
      <td>0.077808</td>
      <td>82.819645</td>
      <td>12.668646</td>
    </tr>
  </tbody>
</table>
</div>



[ResNet-RS](https://arxiv.org/abs/2103.07579) performs well here, with lower memory use than convnext but nonetheless high accuracy. A version trained on the larger Imagenet-22k dataset (like `convnext_tiny_in22k` would presumably do even better, and may top the charts!)

[RegNet-y](https://arxiv.org/abs/2003.13678) is impressively miserly in terms of memory use, whilst still achieving high accuracy.

### Planet

Here's the top-15 for Planet:


```
pt = pt_all[pt_all.dataset=='planet'].sort_values('score').reset_index(drop=True)
pt.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>family</th>
      <th>model_name</th>
      <th>GPU_mem</th>
      <th>error_rate</th>
      <th>fit_time</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_small_patch16_224</td>
      <td>2.121094</td>
      <td>0.035000</td>
      <td>20.075387</td>
      <td>3.502641</td>
    </tr>
    <tr>
      <th>1</th>
      <td>planet</td>
      <td>swin</td>
      <td>swin_base_patch4_window7_224_in22k</td>
      <td>6.283203</td>
      <td>0.031177</td>
      <td>37.115593</td>
      <td>3.651255</td>
    </tr>
    <tr>
      <th>2</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_small_patch32_224</td>
      <td>0.775391</td>
      <td>0.038529</td>
      <td>17.817797</td>
      <td>3.768855</td>
    </tr>
    <tr>
      <th>3</th>
      <td>planet</td>
      <td>convnext</td>
      <td>convnext_tiny_in22k</td>
      <td>2.660156</td>
      <td>0.037647</td>
      <td>22.014424</td>
      <td>3.840538</td>
    </tr>
    <tr>
      <th>4</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_base_patch32_224</td>
      <td>2.755859</td>
      <td>0.038823</td>
      <td>19.060116</td>
      <td>3.845859</td>
    </tr>
    <tr>
      <th>5</th>
      <td>planet</td>
      <td>swin</td>
      <td>swinv2_cr_tiny_ns_224</td>
      <td>3.302734</td>
      <td>0.036176</td>
      <td>26.547731</td>
      <td>3.854518</td>
    </tr>
    <tr>
      <th>6</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_base_patch32_224_sam</td>
      <td>2.755859</td>
      <td>0.039412</td>
      <td>18.567447</td>
      <td>3.884713</td>
    </tr>
    <tr>
      <th>7</th>
      <td>planet</td>
      <td>swin</td>
      <td>swin_tiny_patch4_window7_224</td>
      <td>2.796875</td>
      <td>0.036765</td>
      <td>25.790094</td>
      <td>3.889339</td>
    </tr>
    <tr>
      <th>8</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_base_patch16_224_miil</td>
      <td>4.853516</td>
      <td>0.036471</td>
      <td>28.131062</td>
      <td>3.943604</td>
    </tr>
    <tr>
      <th>9</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_base_patch16_224</td>
      <td>4.853516</td>
      <td>0.036176</td>
      <td>29.274090</td>
      <td>3.953148</td>
    </tr>
    <tr>
      <th>10</th>
      <td>planet</td>
      <td>convnext</td>
      <td>convnext_small_in22k</td>
      <td>4.210938</td>
      <td>0.036471</td>
      <td>28.446879</td>
      <td>3.955122</td>
    </tr>
    <tr>
      <th>11</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_small_r26_s32_224</td>
      <td>3.367188</td>
      <td>0.038529</td>
      <td>23.008444</td>
      <td>3.968847</td>
    </tr>
    <tr>
      <th>12</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_tiny_patch16_224</td>
      <td>1.070312</td>
      <td>0.040588</td>
      <td>18.103888</td>
      <td>3.981860</td>
    </tr>
    <tr>
      <th>13</th>
      <td>planet</td>
      <td>swin</td>
      <td>swin_small_patch4_window7_224</td>
      <td>4.486328</td>
      <td>0.035588</td>
      <td>31.928643</td>
      <td>3.983339</td>
    </tr>
    <tr>
      <th>14</th>
      <td>planet</td>
      <td>swin</td>
      <td>swin_s3_tiny_224</td>
      <td>3.126953</td>
      <td>0.038235</td>
      <td>24.459997</td>
      <td>3.994054</td>
    </tr>
  </tbody>
</table>
</div>



Interestingly, the results look quite different: *vit* and *swin* take most of the top positions in terms of the combination of accuracy and speed. `vit_small_patch32` is a particular standout with its extremely low memory use and also the fastest in the top 15.

Because this dataset is so different to Imagenet, what we're testing here is more about how quickly and data-efficiently a model can learn new features that it hasn't seen before. We can see that the transformers-based architectures able to do that better than any other model. `convnext_tiny` still puts in a good performance, but it's a bit let down by it's relatively poor speed -- hopefully we'll see NVIDIA speed it up in the future, because in theory it's a light-weight architecture which should be able to do better.

The downside of vit and swin models, like most transformers-based models, is that they can only handle one input image size. Of course, we can always squish or crop or pad our input images to the required size, but this can have a significant impact on performance. For instance, recently in looking at the [Kaggle Paddy Disease](https://www.kaggle.com/competitions/paddy-disease-classification) competition I've found that the ability of convnext models to handle dynamically sized inputs to be very convenient.

Here's a chart of the seven top families, this time for the Planet dataset:


```
pt2 = pt[pt.family.isin(faves)]
px.scatter(pt2, width=w, height=h, x='fit_time', y='error_rate', color='family', hover_name='model_name', trendline="ols")
```


<div>                            <div id="eaa7fca0-9795-4e7c-9708-6b6efb1c7ce3" class="plotly-graph-div" style="height:700px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("eaa7fca0-9795-4e7c-9708-6b6efb1c7ce3")) {                    Plotly.newPlot(                        "eaa7fca0-9795-4e7c-9708-6b6efb1c7ce3",                        [{"hovertemplate":"<b>%{hovertext}</b><br><br>family=vit<br>fit_time=%{x}<br>error_rate=%{y}<extra></extra>","hovertext":["vit_small_patch16_224","vit_small_patch32_224","vit_base_patch32_224","vit_base_patch32_224_sam","vit_base_patch16_224_miil","vit_base_patch16_224","vit_small_r26_s32_224","vit_tiny_patch16_224","vit_base_patch16_224_sam","vit_tiny_r_s16_p8_224"],"legendgroup":"vit","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"vit","orientation":"v","showlegend":true,"x":[20.075386599986818,17.817797187017277,19.0601161980303,18.56744659197284,28.1310620289878,29.2740897089825,23.00844355300069,18.103888006007764,30.0218978819903,20.520312173990533],"xaxis":"x","y":[0.0350000262260435,0.038529336452484,0.0388234853744507,0.039411723613739,0.036470592021942,0.0361764430999755,0.038529336452484,0.0405882000923155,0.0373529195785522,0.0417646765708924],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>OLS trendline</b><br>error_rate = -0.000225004 * fit_time + 0.0433178<br>R<sup>2</sup>=0.279882<br><br>family=vit<br>fit_time=%{x}<br>error_rate=%{y} <b>(trend)</b><extra></extra>","legendgroup":"vit","marker":{"color":"#636efa","symbol":"circle"},"mode":"lines","name":"vit","showlegend":false,"x":[17.817797187017277,18.103888006007764,18.56744659197284,19.0601161980303,20.075386599986818,20.520312173990533,23.00844355300069,28.1310620289878,29.2740897089825,30.0218978819903],"xaxis":"x","y":[0.039308747645665124,0.03924437609137966,0.03914007359475903,0.03902922100465133,0.03880078118954441,0.03870067119356119,0.03814083189253608,0.03698822268097755,0.03673103697815701,0.03656277721164737],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>%{hovertext}</b><br><br>family=swin<br>fit_time=%{x}<br>error_rate=%{y}<extra></extra>","hovertext":["swin_base_patch4_window7_224_in22k","swinv2_cr_tiny_ns_224","swin_tiny_patch4_window7_224","swin_small_patch4_window7_224","swin_s3_tiny_224","swinv2_cr_small_ns_224","swin_large_patch4_window7_224_in22k","swin_s3_small_224","swinv2_cr_small_224","swin_base_patch4_window7_224","swin_s3_base_224","swin_large_patch4_window7_224"],"legendgroup":"swin","marker":{"color":"#EF553B","symbol":"circle"},"mode":"markers","name":"swin","orientation":"v","showlegend":true,"x":[37.1155928770313,26.54773105797358,25.79009427002165,31.92864254495361,24.45999662100803,35.8855170509778,52.94627756503178,35.36353248998057,37.25085559103172,37.62205783399986,41.811679941019975,52.09745338198263],"xaxis":"x","y":[0.031176507472992,0.0361764430999755,0.036764681339264,0.0355882048606871,0.0382352471351623,0.0358823537826538,0.0314705371856689,0.036764681339264,0.0361764430999755,0.0376470685005188,0.0370587706565855,0.0352941155433655],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>OLS trendline</b><br>error_rate = -0.000125765 * fit_time + 0.0402853<br>R<sup>2</sup>=0.274385<br><br>family=swin<br>fit_time=%{x}<br>error_rate=%{y} <b>(trend)</b><extra></extra>","legendgroup":"swin","marker":{"color":"#EF553B","symbol":"circle"},"mode":"lines","name":"swin","showlegend":false,"x":[24.45999662100803,25.79009427002165,26.54773105797358,31.92864254495361,35.36353248998057,35.8855170509778,37.1155928770313,37.25085559103172,37.62205783399986,41.811679941019975,52.09745338198263,52.94627756503178],"xaxis":"x","y":[0.03720905624287752,0.03704177621150936,0.036946491849682045,0.036269760300687244,0.03583777059062618,0.035772123084368224,0.035617422320164904,0.03560041097437558,0.03555372664041432,0.03502681786946596,0.03373322524855874,0.03362647268338294],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>%{hovertext}</b><br><br>family=convnext<br>fit_time=%{x}<br>error_rate=%{y}<extra></extra>","hovertext":["convnext_tiny_in22k","convnext_small_in22k","convnext_tiny_hnf","convnext_base_in22k","convnext_tiny","convnext_base","convnext_small","convnext_large_in22k"],"legendgroup":"convnext","marker":{"color":"#00cc96","symbol":"circle"},"mode":"markers","name":"convnext","orientation":"v","showlegend":true,"x":[22.014423568965867,28.44687945500482,24.75937563698972,34.22033763502259,23.18080709700007,31.467404578987043,26.913248616969213,47.05334026599303],"xaxis":"x","y":[0.037647008895874,0.036470592021942,0.038529336452484,0.0355882048606871,0.0397058129310607,0.037647008895874,0.0405882000923155,0.0370587706565855],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>OLS trendline</b><br>error_rate = -9.59488e-05 * fit_time + 0.0407595<br>R<sup>2</sup>=0.221635<br><br>family=convnext<br>fit_time=%{x}<br>error_rate=%{y} <b>(trend)</b><extra></extra>","legendgroup":"convnext","marker":{"color":"#00cc96","symbol":"circle"},"mode":"lines","name":"convnext","showlegend":false,"x":[22.014423568965867,23.18080709700007,24.75937563698972,26.913248616969213,28.44687945500482,31.467404578987043,34.22033763502259,47.05334026599303],"xaxis":"x","y":[0.03864725529156526,0.03853534223391931,0.03838388053391425,0.03817721908407642,0.03803006910102242,0.03774025344931221,0.03747611292572747,0.036244802187285347],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>%{hovertext}</b><br><br>family=resnet<br>fit_time=%{x}<br>error_rate=%{y}<extra></extra>","hovertext":["resnet18","resnet26","resnet26d","resnet50_gn","resnet34","resnet18d","resnet50d","resnet34d","resnet101","resnet50","resnet152"],"legendgroup":"resnet","marker":{"color":"#ab63fa","symbol":"circle"},"mode":"markers","name":"resnet","orientation":"v","showlegend":true,"x":[17.189185384951998,17.83223271300085,20.341083280975,24.831026803003624,19.8849368430092,17.317234788963106,22.132747628027573,18.742670573003124,28.63429423799971,21.53141612198669,31.76075899699936],"xaxis":"x","y":[0.0426470041275024,0.0441175699234007,0.0444117188453674,0.0432352423667907,0.0488234758377076,0.054117739200592,0.0549999475479126,0.0576471090316772,0.053823471069336,0.0605883002281187,0.0552942156791688],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>OLS trendline</b><br>error_rate = 0.000363223 * fit_time + 0.042951<br>R<sup>2</sup>=0.072562<br><br>family=resnet<br>fit_time=%{x}<br>error_rate=%{y} <b>(trend)</b><extra></extra>","legendgroup":"resnet","marker":{"color":"#ab63fa","symbol":"circle"},"mode":"lines","name":"resnet","showlegend":false,"x":[17.189185384951998,17.317234788963106,17.83223271300085,18.742670573003124,19.8849368430092,20.341083280975,21.53141612198669,22.132747628027573,24.831026803003624,28.63429423799971,31.76075899699936],"xaxis":"x","y":[0.04919446381243072,0.04924097426592473,0.04942803321540167,0.04975872493609992,0.050173622003674555,0.05033930475599431,0.050771660694483946,0.050990077962898325,0.05197015427837012,0.05335158744103683,0.05448719049125865],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>%{hovertext}</b><br><br>family=regnetx<br>fit_time=%{x}<br>error_rate=%{y}<extra></extra>","hovertext":["regnetx_080","regnetx_016","regnetx_064","regnetx_040","regnetx_120","regnetx_008","regnetx_160","regnetx_032","regnetx_002","regnetx_006","regnetx_004"],"legendgroup":"regnetx","marker":{"color":"#FFA15A","symbol":"circle"},"mode":"markers","name":"regnetx","orientation":"v","showlegend":true,"x":[27.766882728028577,22.21239931700984,29.76441209804034,25.664181158994325,34.610666284977924,20.212097682990137,37.67724299401743,26.180230233003385,18.394935377000365,19.35444484697655,20.875265374023],"xaxis":"x","y":[0.0414705872535705,0.0441176891326904,0.0414705872535705,0.0438234806060792,0.0408822894096375,0.0482352375984191,0.0417646765708924,0.0473529100418091,0.0511764287948608,0.0517646670341491,0.0550000071525573],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>OLS trendline</b><br>error_rate = -0.000612538 * fit_time + 0.0618392<br>R<sup>2</sup>=0.627016<br><br>family=regnetx<br>fit_time=%{x}<br>error_rate=%{y} <b>(trend)</b><extra></extra>","legendgroup":"regnetx","marker":{"color":"#FFA15A","symbol":"circle"},"mode":"lines","name":"regnetx","showlegend":false,"x":[18.394935377000365,19.35444484697655,20.212097682990137,20.875265374023,22.21239931700984,25.664181158994325,26.180230233003385,27.766882728028577,29.76441209804034,34.610666284977924,37.67724299401743],"xaxis":"x","y":[0.05057157099023192,0.0499838352314749,0.049458490504718775,0.049052275268425156,0.04823323026977533,0.04611888363387104,0.04580278410222587,0.04483089957453018,0.04360733745590676,0.04063882388640562,0.03876042993067043],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>%{hovertext}</b><br><br>family=levit<br>fit_time=%{x}<br>error_rate=%{y}<extra></extra>","hovertext":["levit_384","levit_256","levit_192","levit_128s","levit_128"],"legendgroup":"levit","marker":{"color":"#19d3f3","symbol":"circle"},"mode":"markers","name":"levit","orientation":"v","showlegend":true,"x":[21.410115082981065,22.119607886997983,20.173096203012392,18.47770341602154,19.63021514203865],"xaxis":"x","y":[0.0455881357192992,0.054411768913269,0.0582353472709655,0.0600000023841857,0.060588240623474],"yaxis":"y","type":"scatter"},{"hovertemplate":"<b>OLS trendline</b><br>error_rate = -0.00301116 * fit_time + 0.117078<br>R<sup>2</sup>=0.493400<br><br>family=levit<br>fit_time=%{x}<br>error_rate=%{y} <b>(trend)</b><extra></extra>","legendgroup":"levit","marker":{"color":"#19d3f3","symbol":"circle"},"mode":"lines","name":"levit","showlegend":false,"x":[18.47770341602154,19.63021514203865,20.173096203012392,21.410115082981065,22.119607886997983],"xaxis":"x","y":[0.06143905711924147,0.05796866275426701,0.05633396235831923,0.05260910364018194,0.05047270903918359],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"fit_time"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"error_rate"}},"legend":{"title":{"text":"family"},"tracegroupgap":0},"margin":{"t":60},"height":700,"width":900},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('eaa7fca0-9795-4e7c-9708-6b6efb1c7ce3');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


One striking feature is that for this dataset, there's little correlation between model size and performance. Regnetx and vit are the only families that show much of a relationship here. This suggests that if you have data that's very different to your pretrained model's data, that you might want to focus on smaller models. This makes intuitive sense, since these models have more new features to learn, and if they're too big they're either going to overfit, or fail to utilise their capacity effectively.

Here's the most accurate small and fast models on the Planet dataset:


```
pt.query("(GPU_mem<2.7) & (fit_time<25)").sort_values("error_rate").head(15).reset_index(drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>family</th>
      <th>model_name</th>
      <th>GPU_mem</th>
      <th>error_rate</th>
      <th>fit_time</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_small_patch16_224</td>
      <td>2.121094</td>
      <td>0.035000</td>
      <td>20.075387</td>
      <td>3.502641</td>
    </tr>
    <tr>
      <th>1</th>
      <td>planet</td>
      <td>convnext</td>
      <td>convnext_tiny_in22k</td>
      <td>2.660156</td>
      <td>0.037647</td>
      <td>22.014424</td>
      <td>3.840538</td>
    </tr>
    <tr>
      <th>2</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_small_patch32_224</td>
      <td>0.775391</td>
      <td>0.038529</td>
      <td>17.817797</td>
      <td>3.768855</td>
    </tr>
    <tr>
      <th>3</th>
      <td>planet</td>
      <td>convnext</td>
      <td>convnext_tiny</td>
      <td>2.660156</td>
      <td>0.039706</td>
      <td>23.180807</td>
      <td>4.096878</td>
    </tr>
    <tr>
      <th>4</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_tiny_patch16_224</td>
      <td>1.070312</td>
      <td>0.040588</td>
      <td>18.103888</td>
      <td>3.981860</td>
    </tr>
    <tr>
      <th>5</th>
      <td>planet</td>
      <td>mobilevit</td>
      <td>mobilevit_xxs</td>
      <td>1.152344</td>
      <td>0.041471</td>
      <td>20.329964</td>
      <td>4.160743</td>
    </tr>
    <tr>
      <th>6</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_tiny_r_s16_p8_224</td>
      <td>0.785156</td>
      <td>0.041765</td>
      <td>20.520312</td>
      <td>4.198198</td>
    </tr>
    <tr>
      <th>7</th>
      <td>planet</td>
      <td>resnetblur</td>
      <td>resnetblur50</td>
      <td>2.195312</td>
      <td>0.042353</td>
      <td>21.530770</td>
      <td>4.300124</td>
    </tr>
    <tr>
      <th>8</th>
      <td>planet</td>
      <td>resnet</td>
      <td>resnet18</td>
      <td>0.634766</td>
      <td>0.042647</td>
      <td>17.189185</td>
      <td>4.144828</td>
    </tr>
    <tr>
      <th>9</th>
      <td>planet</td>
      <td>resnetrs</td>
      <td>resnetrs50</td>
      <td>2.419922</td>
      <td>0.043823</td>
      <td>23.490568</td>
      <td>4.535317</td>
    </tr>
    <tr>
      <th>10</th>
      <td>planet</td>
      <td>resnet</td>
      <td>resnet26</td>
      <td>1.289062</td>
      <td>0.044118</td>
      <td>17.832233</td>
      <td>4.316120</td>
    </tr>
    <tr>
      <th>11</th>
      <td>planet</td>
      <td>regnetx</td>
      <td>regnetx_016</td>
      <td>1.367188</td>
      <td>0.044118</td>
      <td>22.212399</td>
      <td>4.509375</td>
    </tr>
    <tr>
      <th>12</th>
      <td>planet</td>
      <td>resnet</td>
      <td>resnet26d</td>
      <td>1.412109</td>
      <td>0.044412</td>
      <td>20.341083</td>
      <td>4.456320</td>
    </tr>
    <tr>
      <th>13</th>
      <td>planet</td>
      <td>regnety</td>
      <td>regnety_006</td>
      <td>0.914062</td>
      <td>0.045000</td>
      <td>22.715365</td>
      <td>4.622193</td>
    </tr>
    <tr>
      <th>14</th>
      <td>planet</td>
      <td>levit</td>
      <td>levit_384</td>
      <td>1.699219</td>
      <td>0.045588</td>
      <td>21.410115</td>
      <td>4.623098</td>
    </tr>
  </tbody>
</table>
</div>



`convnext_tiny` is still the most accurate option amongst architectures that don't have a fixed resolution. Resnet 18 has very low memory use, is fast, and is still quite accurate.

Here's the subset of ultra lean/fast models on the Planet dataset:


```
pt.query("(GPU_mem<1.6) & (fit_time<21)").sort_values("error_rate").head(15).reset_index(drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>family</th>
      <th>model_name</th>
      <th>GPU_mem</th>
      <th>error_rate</th>
      <th>fit_time</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_small_patch32_224</td>
      <td>0.775391</td>
      <td>0.038529</td>
      <td>17.817797</td>
      <td>3.768855</td>
    </tr>
    <tr>
      <th>1</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_tiny_patch16_224</td>
      <td>1.070312</td>
      <td>0.040588</td>
      <td>18.103888</td>
      <td>3.981860</td>
    </tr>
    <tr>
      <th>2</th>
      <td>planet</td>
      <td>mobilevit</td>
      <td>mobilevit_xxs</td>
      <td>1.152344</td>
      <td>0.041471</td>
      <td>20.329964</td>
      <td>4.160743</td>
    </tr>
    <tr>
      <th>3</th>
      <td>planet</td>
      <td>vit</td>
      <td>vit_tiny_r_s16_p8_224</td>
      <td>0.785156</td>
      <td>0.041765</td>
      <td>20.520312</td>
      <td>4.198198</td>
    </tr>
    <tr>
      <th>4</th>
      <td>planet</td>
      <td>resnet</td>
      <td>resnet18</td>
      <td>0.634766</td>
      <td>0.042647</td>
      <td>17.189185</td>
      <td>4.144828</td>
    </tr>
    <tr>
      <th>5</th>
      <td>planet</td>
      <td>resnet</td>
      <td>resnet26</td>
      <td>1.289062</td>
      <td>0.044118</td>
      <td>17.832233</td>
      <td>4.316120</td>
    </tr>
    <tr>
      <th>6</th>
      <td>planet</td>
      <td>resnet</td>
      <td>resnet26d</td>
      <td>1.412109</td>
      <td>0.044412</td>
      <td>20.341083</td>
      <td>4.456320</td>
    </tr>
    <tr>
      <th>7</th>
      <td>planet</td>
      <td>efficientnet</td>
      <td>efficientnet_es</td>
      <td>1.507812</td>
      <td>0.046176</td>
      <td>17.470632</td>
      <td>4.500840</td>
    </tr>
    <tr>
      <th>8</th>
      <td>planet</td>
      <td>regnetx</td>
      <td>regnetx_008</td>
      <td>0.974609</td>
      <td>0.048235</td>
      <td>20.212098</td>
      <td>4.833754</td>
    </tr>
    <tr>
      <th>9</th>
      <td>planet</td>
      <td>resnet</td>
      <td>resnet34</td>
      <td>0.949219</td>
      <td>0.048823</td>
      <td>19.884937</td>
      <td>4.876730</td>
    </tr>
    <tr>
      <th>10</th>
      <td>planet</td>
      <td>efficientnet</td>
      <td>efficientnet_es_pruned</td>
      <td>1.507812</td>
      <td>0.050294</td>
      <td>17.644619</td>
      <td>4.910943</td>
    </tr>
    <tr>
      <th>11</th>
      <td>planet</td>
      <td>regnety</td>
      <td>regnety_002</td>
      <td>0.490234</td>
      <td>0.050882</td>
      <td>20.417092</td>
      <td>5.109463</td>
    </tr>
    <tr>
      <th>12</th>
      <td>planet</td>
      <td>regnetx</td>
      <td>regnetx_002</td>
      <td>0.462891</td>
      <td>0.051176</td>
      <td>18.394935</td>
      <td>5.035501</td>
    </tr>
    <tr>
      <th>13</th>
      <td>planet</td>
      <td>regnetx</td>
      <td>regnetx_006</td>
      <td>0.730469</td>
      <td>0.051765</td>
      <td>19.354445</td>
      <td>5.143050</td>
    </tr>
    <tr>
      <th>14</th>
      <td>planet</td>
      <td>efficientnet</td>
      <td>efficientnet_lite0</td>
      <td>1.494141</td>
      <td>0.052059</td>
      <td>16.381403</td>
      <td>5.017507</td>
    </tr>
  </tbody>
</table>
</div>



## Conclusions

It really seems like it's time for a changing of the guard when it comes to computer vision models. There are, as at the time of writing (June 2022) three very clear winners when it comes to fine-tuning pretrained models:

- [convnext](https://arxiv.org/abs/2201.03545)
- [vit](https://arxiv.org/abs/2010.11929)
- [swin](https://arxiv.org/abs/2103.14030) (and [v2](https://arxiv.org/abs/2111.09883)).

[Tanishq Abraham](https://www.kaggle.com/tanlikesmath) studied the top results of a [recent Kaggle computer vision competition](https://www.kaggle.com/c/petfinder-pawpularity-score) and found that the above three approaches did indeed appear to the best approaches. However, there were two other architectures which were also very strong in that competition, but which aren't in our top models above:

- [EfficientNet](https://arxiv.org/abs/1905.11946) and [v2](https://arxiv.org/abs/2104.00298)
- [BEiT](https://arxiv.org/abs/2106.08254).

BEiT isn't there because it's too big to fit on my GPU (even the smallest BEiT model is too big!) This is fixable with gradient accumulation, so perhaps in a future iteration we'll add it in. EfficientNet didn't have any variants that were fast and accurate enough to appear in the top 15 on either dataset. However, it's notoriously fiddly to train, so there might well be some set of hyperparameters that would work for these datasets. Having said that, I'm mainly interested in knowing which architectures can be trained quickly and easily without to much mucking around, so perhaps EfficientNet doesn't really fit here anyway!

Thankfully, it's easy to try lots of different models, especially if you use fastai and timm, because it's literally as easy as changing the model name in one place in your code. Your existing hyperparameters are most likely going to continue to work fine regardless of what model you try. And it's particularly easy if you use [wandb](https://wandb.ai/), since you can start and stop experiments at any time and they'll all be automatically tracked and managed for you.

If you found this notebook useful, please remember to click the little up-arrow at the top to upvote it, since I like to know when people have found my work useful, and it helps others find it too. And if you have any questions or comments, please pop them below -- I read every comment I receive!


```

```
