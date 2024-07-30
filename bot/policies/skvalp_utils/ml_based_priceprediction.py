import pandas as pd
import os
from collections import deque
import numpy as np
from policies.policy import Policy
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def _load_model(file_name:str):
    # Get the current working directory
    script_directory = os.path.dirname(os.path.realpath(__file__))
    # Construct the path to your model file
    model_file_path = os.path.join(script_directory, file_name)
    # Load the model
    return tf.keras.models.load_model(model_file_path)

class MLBasedPricePredictor(Policy):
    def __init__(self, window_size=12*24):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        solar_window = 12*24
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.demand_total_history =  deque(maxlen=window_size)
        self.temp_air_history =  deque(maxlen=window_size)
        
        self.model = _load_model('model_12hr_n_nns_r2.keras')
        self.scaler = MinMaxScaler()
        # solar model

    def act(self, external_state, internal_state):
        market_price = external_state['price']
        self.price_history.append(market_price)
        demand_total = external_state['demand']
        self.demand_total_history.append(demand_total)
        temp_air = external_state['temp_air']
        self.temp_air_history.append(temp_air)
        
        n_input = int(12*12) #last 12 hr
        if(len(self.price_history) >= n_input):
            #do the pp based policy
            data_input = pd.DataFrame({'price_imputed': self.price_history, 'demand' :  self.demand_total_history, 'temp_air'  : self.temp_air_history })           
            data_input.dropna(inplace = True)

            data_input = data_input.tail(n_input) # Fetching the last n_input rows from the df
            
            if(data_input.shape[0] < n_input):
                first_data_point = data_input.head(1)
                # first_data_point.append(data_input)
                dups = pd.DataFrame(np.repeat(first_data_point, n_input - data_input.shape[0], axis=0), columns=first_data_point.columns)
                data_input = pd.concat([dups, data_input], axis = 0)
            
            n_features = data_input.shape[1]
            data_input = data_input.values.reshape(len(data_input), n_features)
            self.scaler.fit(data_input)
            scaled_data = self.scaler.transform(data_input)
        
            y_hat_scaled = self.model.predict(scaled_data.reshape(1, n_input, n_features), batch_size = 1, verbose = False)
            y_hat_scaled = y_hat_scaled.reshape(y_hat_scaled.shape[1], 1)
            #to make to the size of scaler fit
            y_hat_scaled = np.repeat(y_hat_scaled, n_features, axis = 1)
            y_hat = self.scaler.inverse_transform(y_hat_scaled)[:,0]
            #next 1 hr data
            y_hat = np.append(external_state['price'], y_hat)
            y_hat = y_hat.reshape(len(y_hat), 1)
            
        # Solar prediction
            #DO your OPTIMIZATION  BELOW
            return y_hat
        return None

    def load_historical(self, external_states: pd.DataFrame):   
        for price in external_states['price'].values:
            self.price_history.append(price)