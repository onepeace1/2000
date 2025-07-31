<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> modified version of (ICLR'24) Time-LLM: Time Series Forecasting by Reprogramming Large Language Models </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/KimMeen/Time-LLM?color=green)
![](https://img.shields.io/github/stars/KimMeen/Time-LLM?color=yellow)
![](https://img.shields.io/github/forks/KimMeen/Time-LLM?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>

<p align="center">

<img src="./figures/logo.png" width="70">

</p>

## Requirements
Use python 3.11 from MiniConda

- torch==2.2.2
- accelerate==0.28.0
- einops==0.7.0
- matplotlib==3.7.0
- numpy==1.23.5
- pandas==1.5.3
- scikit_learn==1.2.2
- scipy==1.12.0
- tqdm==4.65.0
- peft==0.4.0
- transformers==4.31.0
- deepspeed==0.14.0
- sentencepiece==0.2.0

To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
We collected daily stcok price data of companies in S&P100 via yahoo finance. Note that one could add more data if available, and 
preprocess it if one wants to add external information in prompt or time series data.

## Quick Demos
1. Download datasets and place them under `./dataset`
2. Tune the model. FOr stock price prediction, we provide experiment script for demonstration purpose under the folder `./scripts`.
   
```bash
bash ./scripts/TimeLLM_ETTM4.sh 
```

## Detailed usage

Please refer to ```run_main.py```, ```run_m4.py``` and ```run_pretrain.py``` for the detailed description of each hyperparameter.
Note that for stock price prediction via HyperClovaX, we changed 'patch length'=32, embedded dim='2048' 


## Acknowledgement
Our implementation is largely based on [TimeLLM](https://github.com/KimMeen/Time-LLM)  as the code base and have modified it to our purposes mainly for using HyperClovaX and stock price prediction. We thank the authors for sharing their implementations and related resources.
