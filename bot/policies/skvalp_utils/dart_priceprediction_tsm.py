from collections import deque
import os

from darts import TimeSeries
from darts.models import TSMixerModel
import numpy as np
import pandas as pd

from policies.policy import Policy


def _load_model(file_name: str):
    # Get the current working directory
    script_directory = os.path.dirname(os.path.realpath(__file__))
    # Construct the path to your model file
    model_file_path = os.path.join(script_directory, file_name)
    # Load the model
    return TSMixerModel.load(model_file_path)


def set_index(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df.set_index(keys="timestamp", inplace=True, drop=True)
    return df


class MLBasedPricePredictor(Policy):
    def __init__(self, window_size=12 * 24, fh=36):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)
        self.temp_air_history = deque(maxlen=window_size)
        self.demand_total_history = deque(maxlen=window_size)
        self.model = _load_model("TSM_24hrL_3hr")
        # another model to test
        # self.model = _load_model("TSM_1month_24hrL_3hr")
        self.fh = fh  # forecast horizon of 3 hours

    def act(self, external_state, internal_state):
        market_price = external_state["price"]
        self.price_history.append(market_price)
        demand_total = external_state["demand"]
        self.demand_total_history.append(demand_total)
        temp_air = external_state["temp_air"]
        self.temp_air_history.append(temp_air)
        self.timestamp_history.append(external_state["timestamp"])

        # start after 24 hours
        n_input = int(24 * 12)
        if len(self.price_history) >= n_input:
            # do the pp based policy
            df = pd.DataFrame(
                {
                    "price": self.price_history,
                    "demand": self.demand_total_history,
                    "temp_air": self.temp_air_history,
                    "timestamp": self.timestamp_history,
                }
            )
            df = set_index(df)
            series = TimeSeries.from_dataframe(df)
            y = series["price"]
            X = series.drop_columns("price")

            y_pred = self.model.predict(series=y, past_covariates=X, n=self.fh)

            y_hat = y_pred.values()

            y_hat_final = np.append(external_state["price"], y_hat)

            return y_hat_final
        return None

    def load_historical(self, external_states: pd.DataFrame):
        for price in external_states["price"].values:
            self.price_history.append(price)
