from collections import deque

import numpy as np
import pandas as pd

from policies.policy import Policy
from sklearn.preprocessing import MinMaxScaler
from sktime.datatypes import convert_to
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.impute import Imputer
from xgboost import XGBRegressor


class MLBasedPricePredictor(Policy):
    def __init__(self, window_size=12 * 24):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)

        self.scaler = TabularToSeriesAdaptor(MinMaxScaler())
        self.xgb = XGBRegressor()
        self.forecaster = make_reduction(
            self.xgb, window_length=288, strategy="recursive"
        )
        self.pipe = ForecastingPipeline(
            steps=[
                ("imputer", Imputer(method="ffill")),
                ("scaler", TabularToSeriesAdaptor(MinMaxScaler())),
                ("forecaster", self.forecaster),
            ]
        )
        self.fh = ForecastingHorizon(np.arange(1, 13), is_relative=True)

    def act(self, external_state, internal_state):
        market_price = external_state["price"]
        self.price_history.append(market_price)

        n_input = int(24 * 12)  # last 12 hr
        if len(self.price_history) >= n_input:
            # do the pp based policy
            y_train = pd.DataFrame(
                {
                    "price": self.price_history,
                }
            )
            y_train = convert_to(y_train, to_type="pd.Series")

            y_hat = self.pipe.fit_predict(y=y_train, fh=self.fh)
            y_hat = y_hat.values.reshape(len(y_hat), 1)

            return y_hat
        return None

    def load_historical(self, external_states: pd.DataFrame):
        for price in external_states["price"].values:
            self.price_history.append(price)
