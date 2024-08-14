# MLAI Green Battery Hack - Team Skvalp

## Competition Overview

The goal of [MLAI Green Battery Hack](https://www.mlai.au/hackathon) is to develop a strategy to optimise battery operations within an energy market simulation. Our challenge is to create algorithms, or policies, that expertly manage batery charging and discharging in response to real-time market data and incoming energy from a simulated solar panel.

![](design/setup.png)

## Accomplishments

We are proud to have achieved a final ranking of **4th place** in the competition amongst the many teams formed in Melbourne and Sydney. Our team, Skvalp, have incorporated a similar approach as the 1st place winners team, which will be discussed in the following [section](#strategy-overview). Despite passing the tests and getting our code to work for the provisional leaderboard, we ran into some technical issues due to utilising a function that was not documented properly for participants' usage. We had to disable a function in order for our algorithm to work properly in the final leaderboard simulation, which resulted in a lower score than expected. However, we are proud of our team's efforts and the final result as the process was very rewarding for me.

![](result/provisional_ranking.png)
Screenshot of the provisional leaderboard rankings before the cut-off deadline for code submissions, taken at 4:59PM on 21st April 2024. We were so excited that we're provisionally **1st**!

![](result/final_ranking.png)
Screenshot of the final leaderboard rankings after the competition ended, check it on this [link](https://www.mlai.au/hackathon#!).


## Methodology
Disclaimer: As some of our team members are working on a start-up company, due to proprietary reasons, I have removed the code used in optimising the actions of our batteries used in our strategy.

## Time Series Forecasting with sktime and Darts
Two powerful Python libraries for time series forecasting were implemented: sktime and darts.

### sktime
[sktime](https://www.sktime.net/en/stable/) is a unified framework for machine learning with time series. It provides a collection of algorithm implementations for timeseries classification, regression, clustering, and forecasting. Key features include:
- A scikit-learn compatible API
- Support for various time series learning tasks
- Composable machine learning pipelines
- Extensible framework for custom algorithm development

### Darts
[darts](https://unit8co.github.io/darts/) (Dataframe-based Automated Regression for Time Series) is a Python library for easy manipulation and forecasting of time series. It offers:

- A wide range of forecasting models, including statistical, machine learning, and deep learning approaches
- Tools for preprocessing, transforming, and feature engineering time series data
- Built-in support for backtesting and model evaluation
- Scalability for large datasets

After hacking with these libraries for a few days, Darts was chosen as the main library for our framework. Darts in our scenario offers several advantages:

- Efficient handling of multi-step forecasting
- Support for past covariates without requiring full model retraining
- Faster processing times, which is crucial given our 5-minute constraint

### Price Prediction Models

The system uses an ensemble of regression models for price prediction:

1. XGBoost models
2. CatBoost model

These models are trained on historical price data, demand, and temperature information. The system switches between 12-hour and 24-hour models initially based on the amount of available historical data.

### Optimisation Algorithm

The `SreshtaGyan` class in `simple.py` implements the main optimization logic:

1. Generates price forecasts using ML models and basic forecasting techniques
2. Incorporates solar power forecasts from external sources
3. Calculates optimal battery charge/discharge strategies using the `find_optimal_discharge` function
4. Considers dynamic State of Energy (SOE) limits for better battery management

## Hackathon Details

For more information about the guidelines provided by the hackathon, please check out [hackathon.md](hackathon.md).

## Contributors

Thanks to all the team members of Skvalp who have contributed to this project:

- Jithin George, [@j-georgeAU](https://github.com/j-georgeAU)
- Vishnu Vinayamohanan, [@vihnuav08](https://github.com/vishnuav08)
- Bigi Philip, [@bigiphilip](https://github.com/bigiphilip)
- Eric Kim, [@EricKim9724](https://github.com/EricKim9724)
- Victor Goh, [@victorwkb](https://github.com/victorwkb)

Special thanks to [MLAI Aus](https://www.mlai.au/) for always organising such amazing hackathons and the [sponsors]() that made this happen!