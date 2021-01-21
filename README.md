# SAM The Trader
## Description
 SAM is a trading bot primarliy using a deep reinfrocment learning algorithm to learn to discover new trading tactics in high frequency data. Currenlty SAM has been     
 trained using over 500 tickers and intra-day data. 

## Setup
#### Clone repository
 git clone https://github.com/ava6969/sam-the-trader.git
 cd sam-the-trader

#### Install Environment
conda env create -f environment.yml

## Actions
 SAM can only make take 7 actions which is fractional based on customizable max shares 
 For Example
 if max shares is 3
 actions are [-3 -2 -1 0 1 2 3]
 
## Observation
SAM observes and embedding of the following data:
  - OHLCV
  - 6 tech indicators[MACD MA EMA ATR ROC vwap]
  - cash left, sharp ratio and stocks owned

## Fetch New data
 - python downloader.py -h
 
## Train from scratch
Create a yaml configuration in the folder /configs/
Run configuration using
- python train.py -c CONFIG_NAME

## Evaluate Trained model 
for more details load
- python evaluate.py -h

## Backtesting
After a model has been evaluated a backtest file is stored and ready to be tested
using jupyter notebook, run the BackTester.ipynb
Run and select checkpoint to backtest

## Results
- With a starting capital of 25000$, SAM made 
- SAM can trade over 400 stocks well
- 

## Limitations
- SAM hasnt been tested on other frequency yet, but it is assumed it will generalize
- SAM doenst show risk aversion startegies naturally as it was trained using returns as its objective function
- SAM doesnt cxontainany portfolio allocation model or recommendation system, so ticker to trade is selected manually.
