# Experimental Study of Deep Learning Models for Short-Term Crypto Price Forecasting

## Project Metadata
### Authors
- **Team:** g202401140 - Waseem Mohamad
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
The paper that I have chosen to work on is "Deep learning and NLP in cryptocurrency forecasting: Integrating financial, blockchain, and social media data" by Gurgul, Lessmann and Härdle. This paper is inspired by the recent events in the cryptocurrency world and is related to cryptocurrency forecasting using application of latest techniques.

With the rapid growth of cryptocurrency markets over the last decade, it has become a center of attention for both Financial Technology (FinTech) enthusiasts and computer scientists due to their high volatility and sensitivity to multifaceted drivers. Contrary to traditional assets, the prices of cryptocurrency get affected by blockchain activity, social media sentiments, and influential individuals' opinions while also getting some influence by financial indicators. Artificial Intelligence (AI) and its applications, like Machine Learning (ML) approaches, have gained notoriety for this domain, specifically Deep Learning (DL) techniques as they can model complicated nonlinear dependencies and exploit the highly dimensional data.
   
The paper mentions that the recent studies have surpassed the univariate price series to multimodal approaches that integrate heterogeneous signals, such as blockchain data and textual sentiments (social media). Especially, techniques like transformers and advanced Natural Language Processing (NLP) models have demonstrated promising capabilities in extracting nuanced features from text and merging them with numerial predictors. With these state-of-the-art techniques, many opportunities open up to design more robust forecasting frameworks that better adapt to volatile regimes and market shocks.

## Problem Statement
The paper states that the traditional predicting models often struggle to capture the complex drivers of cryptocurrency markets as they solely depend on the historical prices and volume data of assets. Furthermore, the researchers convey that even existing multimodal techniques face pivotal constraints. The paper integrates financial, blockchain and social media data but also employs comparatively conventional NLP methods and basic fusion strategies. In conclusion, cross modal interactions may be underutilized and external macroeconomic influences are ignored.

This further leads to models that may perform well under stable conditions but lack robustness during turbulent events such as regulatory announcements, global macro shocks or even sudden shifts in investor sentiment. Therefore, according to the paper, a need for forecasting frameworks is required that combine richer data, leverage advanced architectures and provide resilience across different market conditions or environments.

## Application Area and Project Domain
The main application area of this paper is "Forecasting" the assets prices, mainly cryptocurrencies, which is a field of high relevance to crypto enthusiasts, investors, traders and even policymakers. In general, the domain lies at the intersection of DL, FinTech and multimodal data science. Also, the paper leverages blockchain analytics, social media NLP, and financial time series modeling to sow the seeds for forecasts that are both technically rigorous and practically useful. Other than the crypto field, the methods explored in this paper contribute to the wider body of research on multimodal forecasting, highlighting how the heterogeneous data sources can be combined to improve the predictions in complex and dynamic systems.

## What is the paper trying to do, and what are you planning to do?
The paper tries to demonstrate that incorporating blockchain and social sentiment alongside financial dataa improves the predictive accuracy. The models mentioned in the paper used DL and NLP to integrate the multimodal signals while also showing the benefits of a holistic approach.

I will not only try to replicate the work done by the authors but also extend on its research and implmentation in terms of both methodological sophistication and practical applicability while demonstrating measurable improvements in accuracy, robustness and real world applicability. This will eventually narrow the gap between the academic forecasting models and decision making tools that could be deployed in live trading environments. The planned improvements might include the following or even more:
1. **Enhanced Textual Sentiment Extraction:** The traditional NLP approach used in the paper might be replaced with advanced finance related transformer model, such as [FinBERT](https://github.com/ProsusAI/finBERT), as this type of models capture subtle details of financial language that makes it complex and interesting alongwith sentiment shifts that conventional techniques often miss which further leads to more accurate sentiment features from social media and news sources.
2. **Evaluation Framework:** In addition to what the paper has demonstrated, the likes of traditional metrics such as RMSE, MAE, MAPE etc., model performance would be evaluated using different direction of accuracy and simulated trading strategies. Moreover, backtesting would be also be incorporated to evaluate how the model's forecasts would perform in a real world trading scenario as it will allow for assessment of cumulative returns and risk-adjusted performance metrics which will eventually bridge the gap between predictive accuracy and practical trading value.
3. **Incorporation of various Signals:** A factor that was overlooked in the paper was integrating macroeconomic variables like interest rates, inflation data, stock and global risk indices into the forecasting framework. These features provide a valuable context especially during periods of market shakedown (due to the mentioned factors) and can further improve model robustness.
4. **Multimodal Fusion Strategies:** The paper makes use of simple concatenation and feature stacking in its models while a better approach could have been the implementation of attention based transformer fusion layers or cross-modal transformers as these architectures dynamically learn the relative importance of different data types over the time and under constantly changing market conditions. This type of architecture will also potentially improve the performance of the framework during high volatility in the market or during market shifting periods.

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/251___ICS_590___202401140.pdf)

### Reference Paper
- [Deep learning and NLP in cryptocurrency forecasting: Integrating financial, blockchain, and social media data](/1-s2.0-S0169207025000147-main.pdf)
- [LINK](https://www.sciencedirect.com/science/article/pii/S0169207025000147)

### Reference Dataset
- [BTC, ETH, Yahoo Finance, Google New and Trends CSV files](/csv_files-20251124T112916Z-1-001.zip)


## Project Technicalities

### Terminologies
- **Time Series Forecasting:** Predicting future values in sequential data using historical patterns.
- **Multimodal Features:** Combining diverse inputs such as price data, Google Trends, news statistics, and macroeconomic indicators.
- **Technical Indicators:** Engineered features derived from OHLCV data such as RSI, MACD, ATR, and moving averages.
- **Sequence Model:** A model (LSTM, CNN LSTM, Transformer) that processes temporal windows of data.
- **Sliding Window:** A technique to convert continuous time series into fixed length sequences for training.
- **Walk Forward Validation:** Time aware evaluation where the model is trained on past data and tested on a future window.
- **AUC:** A performance metric used for binary classification tasks such as up or down movement prediction.
- **Sharpe Ratio:** A measure of risk adjusted returns used to evaluate trading strategies.
- **Extrema Detection:** Identifying local maxima and minima over short windows as a classification target.

### Problem Statements
- **Problem 1:** Predicting short term price movements in volatile crypto markets remains difficult due to nonlinear and noisy dynamics.
- **Problem 2:** Many deep learning studies rely on paid APIs or proprietary sentiment datasets, limiting reproducibility.
- **Problem 3:** The performance of advanced architectures such as transformers is unclear under limited data density and daily resolution.

### Loopholes or Research Areas
- **Evaluation Metrics:** Financial forecasting studies often lack consistent cross task metrics, especially for direction and extrema prediction.
- **Output Consistency:** Models may overfit high noise crypto data, reducing stability across different market regimes.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible thats why using Collab is one of the options. Also, transformer based models require greater compute and do not always outperform simpler architectures on daily data.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Model Architecture Selection:** Evaluate multiple neural architectures (MLP, LSTM, deep LSTM, CNN LSTM, Transformer) to determine which structure best fits short horizon crypto dynamics.
2. **Enhanced Feature Engineering:** Integrate a broad multimodal feature set including OHLCV derived indicators, Google Trends signals, news headline statistics, and macroeconomic variables.
3. **Robust Evaluation with Walk Forward Splits:** Apply expanding window validation to simulate real trading conditions and test generalization across evolving market regimes.

### Proposed Solution: Code-Based Implementation
This repository implements the full forecasting pipeline in modular Python scripts. The workflow includes:

- **Multimodal Data Acquisition:** Custom scripts collect Kaggle OHLCV, Google News headline metadata, Google Trends indices, and Yahoo Finance macroeconomic series.
- **Feature Engineering Toolkit:** Generates technical indicators, sentiment proxies, z score transformations, volatility measures, local extrema labels, and binary direction targets.
- **Models:** Implements MLP, LSTM, deep LSTM, CNN LSTM, and Transformer based models for classification and regression tasks.
- **Backtesting Engine:** Evaluates trading performance using equity curves, Sharpe ratio, and comparison with buy and hold benchmarks.

### Key Components
- **`Data Acquisition.ipynb`**: Handles Kaggle price loading, Google News scraping, Trends data collection, and macroeconomic series processing.
- **`Fungineering.ipynb`**: Builds technical indicators, sentiment proxies, return features, and supervised learning targets along with the implementations of MLP, LSTM, CNN etc.

## Model Workflow
The forecasting workflow follows a structured multistage process:

1. **Input:**
   - **Historical OHLCV Data:** Daily price and volume for BTC and ETH.
   - **Technical Indicators:** RSI, MACD, ATR, OBV, Bollinger Bands, stochastic oscillator, and moving average ratios.
   - **News and Trends Features:** Google News headline counts, sentiment proxies, and Google Trends interest indices.
   - **Macroeconomic Variables:** S&P 500, VIX, and gold prices.

2. **Processing Pipeline:**
   - **Sequence Construction:** Sliding windows of 30 to 60 days are created for sequence models.
   - **Standardization:** Features are z score normalized using training only statistics.
   - **Target Generation:** Next day direction, local extrema, or regression targets are attached.

3. **Modeling:**
   - LSTM and CNN LSTM models capture temporal structure.
   - MLP models operate on flattened features for stability under noisy data.
   - Transformers apply multi head attention for long range dependency modeling.


4. **Output:**
   - **Classification Outputs:** Up or down probabilities or extrema labels.
   - **Regression Outputs:** Next day normalized closing price.
   - **Backtest Results:** Equity curve, Sharpe ratio, and performance relative to buy and hold.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/ForecastAI-Crypto-Bridging-Blockchain-Trends-Social-Whispers.git crypto-forecasting
    cd crypto-forecasting
    ```

2. **Set Up the Environment in Colab:**
    The notebooks are set up in such a way that it should run one by one. Just need to change the path under the heading `Mount Collab and CD`

## Suggestion
Instead of running the code, since it would take a lot of time, `pkl` files can be used to view quick plots and all other history of the model such as MSE, MAE etc!

## Acknowledgments
- **Open-Source Communities:** Gratitude to contributors of TensorFlow, Keras, PyTorch, scikit learn, and related libraries, and Kaggle datasets used in this project.
- **Individuals:** Special thanks to Dr. Muzammil, family and friends for constant support and guidance throughout the project.
- **Resource Providers:** Gratitude to Google Colab for providing the computational resources necessary for this project for free.
