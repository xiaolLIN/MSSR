# MSSR
The source code for our WSDM 2024 paper [**"Multi-Sequence Attentive User Representation Learning for Side-information Integrated Sequential Recommendation"**]



## Preparation

Our code is based on PyTorch 1.8.1 and runnable for both windows and ubuntu server. Required python packages:

> + numpy==1.21.6
> + scipy==1.7.3 
> + torch==1.13.1+cu116
> + tensorboard==2.11.2


## Usage

In this project, we provide the dataset Toys `./dataset/Amazon_Toys_and_Games`.  <br>
If you want to realize more reproduction results, please download datasets from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or their [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). And put the files in `./dataset/` like the following.

```
$ tree
.
├── Amazon_Beauty
│   ├── Amazon_Beauty.inter
│   └── Amazon_Beauty.item
├── Amazon_Toys_and_Games
│   ├── Amazon_Toys_and_Games.inter
│   └── Amazon_Toys_and_Games.item
├── Amazon_Sports_and_Outdoors
│   ├── Amazon_Sports_and_Outdoors.inter
│   └── Amazon_Sports_and_Outdoors.item
└── yelp
    ├── README.md
    ├── yelp.inter
    ├── yelp.item
    └── yelp.user

```

Run `run_MSSR.sh`. After training and evaluation, check out the final metrics in the "result.txt".


## Credit
This repo is based on [RecBole](https://github.com/RUCAIBox/RecBole).

