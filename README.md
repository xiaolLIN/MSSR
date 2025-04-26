# MSSR
The source code for our WSDM 2024 paper [**"Multi-Sequence Attentive User Representation Learning for Side-information Integrated Sequential Recommendation"**](https://dl.acm.org/doi/10.1145/3616855.3635815)



## Preparation
We train and evaluate our MSSR using a Tesla V100 PCIe GPU with 32 GB memory. <br>
Our code is based on PyTorch, and requires the following python packages:

> + numpy==1.21.6
> + scipy==1.7.3 
> + torch==1.13.1+cu116
> + tensorboard==2.11.2


## Usage

We provide two datasets, Beauty `./dataset/Amazon_Beauty` and Toys `./dataset/Amazon_Toys_and_Games`.  <br>
Please download the other two datasets from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or their [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). And put the files in `./dataset/` like the following.

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

## Questions
If you have any questions, please send an email to Xiaolin Lin (linxiaolin2021@email.szu.edu.com).

## Acknowledgement
This repository is based on [RecBole](https://github.com/RUCAIBox/RecBole) and [DIF-SR](https://github.com/AIM-SE/DIF-SR).

## Citation
```
@inproceedings{lin2024MSSR,
  author       = {Xiaolin Lin and
                  Jinwei Luo and
                  Junwei Pan and
                  Weike Pan and
                  Zhong Ming and
                  Xun Liu and
                  Shudong Huang and
                  Jie Jiang}, 
  title        = {Multi-Sequence Attentive User Representation Learning for Side-information Integrated Sequential Recommendation},
  booktitle    = {Proceedings of the 17th {ACM} International Conference on Web Search and Data Mining},
  pages        = {414--423}, 
  year         = {2024}
}
```
