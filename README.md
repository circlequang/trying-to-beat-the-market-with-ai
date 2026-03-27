# 📉 Trying to Beat the Market with AI: A 3-Year Developer Diary

Welcome to my graveyard of burnt demo accounts, overfitted models, and shattered dreams. 

For the past 3 years, I have felt like Indiana Jones searching for the "Holy Grail" of algorithmic trading. It has been an absolute emotional rollercoaster: from the euphoric, sleepless nights of watching a backtest print money, to the soul-crushing reality of watching the live market destroy my bot the very next day. 

I am a software engineer who, back in 2023, thought predicting the financial markets (Crypto & Forex) would be as straightforward as writing a facial recognition script. Fast forward to today, after years of relentless coding, training, and testing, I am still "trying."

This repository is an open-source diary of my journey. It documents every failed experiment, from basic Machine Learning to complex Reinforcement Learning, complete with the code and the hard-learned lessons.

## 🤔 Why share the code? (The Open-Source Philosophy)

In quantitative trading, there is an unwritten rule: **If a strategy works, you keep it a secret (to prevent Alpha Decay). If it fails, you share it.** I am not here to sell you a "Holy Grail" trading bot. If I ever build a model that consistently beats the market, I will keep the final hyperparameters and reward functions to myself. However, I am open-sourcing my **failed models, data pipelines, and backtesting frameworks**. 

Why? Because witnessing exactly how powerful algorithms like `XGBoost` or `DQN` fail in real-world time-series data is incredibly valuable. I hope this repository saves you hundreds of hours of GPU renting and prevents you from making the same expensive mistakes I did.

## 📖 The Journey (Table of Contents)

This repository is structured alongside a series of in-depth articles I publish on Medium. You can read the story there, and inspect the code here.

### [Part 1: The Sweet Illusion of Basic Machine Learning](./Part-1-Linear-and-XGBoost)
* **Algorithms:** Linear Regression, SARIMA, XGBoost.
* **The Story:** Treating Bitcoin data like a static image classification problem. I combined three models thinking ensemble learning would save me. 
* **The Result:** Win rate didn't matter because the Risk:Reward (R:R) was terrible. Daily volatility destroyed the account. XGBoost is great for Kaggle, but the crypto market ate it alive.
* 🔗 Read the full post-mortem on Medium: **[Comming soon](https://circlequang.medium.com/)**

### [Part 2: The Trap of Deep Learning](./Part-2-Trap-of-Deep-Learning) *(Coming Soon)*
* **Algorithms:** RNN, LSTM.
* **The Story:** Moving to Deep Learning. Achieving a 60% win rate during sideway markets and feeling like a genius, until the market revealed its true nature.
* **The Result:** The brutal reality of severe overfitting, the sideway illusion, and other hidden variables that neural networks failed to capture.
* 🔗 Read the full post-mortem on Medium: **[Comming soon](https://circlequang.medium.com/)**

### [Part 3: A Taste of Victory in Forex](./Part-3-EURUSD-Winning-Phase) *(Coming Soon)*
* **Market:** EURUSD.
* **The Story:** Shifting from Crypto to Forex. Actually achieving a 10% profit in one month. The brief moment I thought I finally found the Grail.
* **The Result:** The reality of "Regime Change." The market structure shifted, and the bot bled out.
* 🔗 Read the full post-mortem on Medium: **[Comming soon](https://circlequang.medium.com/)**

### [Part 4: The Reinforcement Learning Blackhole](./Part-4-RL-Failures) *(Coming Soon)*
* **Algorithms:** DQN, SAC, Huggingface RL models.
* **The Story:** Teaching an AI to trade like playing a video game. 
* **The Result:** The absolute nightmare of designing a proper Reward Function. The agent learned how to "cheat" the backtester rather than trade successfully.
* 🔗 Read the full post-mortem on Medium: **[Comming soon](https://circlequang.medium.com/)**

### [Part 5: Back to Square One - The PPO Bet](./Part-5-PPO-Bitcoin) *(Ongoing - 2026)*
* **Algorithms:** Proximal Policy Optimization (PPO).
* **The Story:** Current ongoing research. Returning to Bitcoin with a more mature, cautious approach. Will it survive? Let's find out.
* 🔗 Read the full post-mortem on Medium: **[Comming soon](https://circlequang.medium.com/)**

## 🛠️ How to use this repository

Each folder corresponds to a specific phase of the journey. Inside, you will find:
1. Jupyter Notebooks (`.ipynb`) with the exact code used.
2. Training logs and backtest charts showing where the model failed.
3. A mini `README.md` explaining the fatal flaw of that specific approach.

*Note: API keys and sensitive paths have been removed. The code from 2023 remains mostly un-refactored to preserve the "authenticity" of my mindset at that time.*

## ☕ Support the Project

Training Deep Learning and Reinforcement Learning models requires massive datasets and endless hours of GPU compute. If my documented failures have saved you from losing money or wasting time, consider supporting my ongoing PPO server costs!

* **[☕ Buy me a Coffee on Ko-fi](https://ko-fi.com/circlequang)**

**Looking for clean data?** Data scientists hate cleaning data. If you want to skip the painful process of scraping, cleaning, and formatting years of OHLCV tick data, you can buy my pre-cleaned datasets and backtest templates directly on my **[Ko-fi Shop](https://ko-fi.com/circlequang)** *(Coming Soon)*.

---
*Disclaimer: Nothing in this repository is financial advice. The code provided is for educational and research purposes only. Do not use these models with real money.*
