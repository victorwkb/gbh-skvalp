"""
Creates a simple 24 hour price and solar forecast based on the data provided
 in validation_data.csv
"""

from pandas import (
    DataFrame,
    read_csv,
    to_datetime, 
    date_range,
    concat)
from datetime import timedelta, datetime, time
from policies.skvalp_utils.utils import current_settlement_period_start_utc
from os import path, getcwd
import logging
from collections import deque
from typing import Union
import pandas as pd


logs = logging.getLogger()
def read_csv_(file_name:str)->DataFrame:
    cwd = getcwd()
    file_dir = path.join(cwd,"bot","data")
    df = read_csv(path.join(file_dir,file_name))
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

class BasicForecaster:
    def __init__(self, days_to_store:int = 30) -> None:
        df = read_csv_("training_data.csv")
        actual_data = df[["timestamp", "price", "pv_power"]].dropna()
        # Ensure the index is in datetime format
        actual_data["timestamp"] = to_datetime(actual_data["timestamp"])
        actual_data.set_index("timestamp", inplace=True)
        quelength = days_to_store*288
        data_curtailed = actual_data.tail(quelength)
        # actual_data = actual_data[(actual_data.index.month >= 4) & (actual_data.index.month < 6)]
        
        self._price_history = deque(data_curtailed["price"].to_list(),
                                    maxlen=quelength)
        self._timestamps = deque(data_curtailed.index.to_list(),
                                 maxlen=quelength)
        self._pv_power = deque(data_curtailed["pv_power"].to_list(),
                               maxlen=quelength)

        # Group by time of day and calculate mean
        # self._time_of_day_avg = actual_data.groupby(
        #     actual_data.index.time).mean() #type: ignore
       
    def _get_data_as_pd(self)->DataFrame:
        # Create a dictionary from deques
        data = {'timestamp': list(self._timestamps),
                'price': list(self._price_history),
                'pv_power': list(self._pv_power)}
        # Create a DataFrame from the dictionary
        df = DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df
        
       
    def _get_time_of_day_avg(self):
        df = self._get_data_as_pd()
        df.index = pd.to_datetime(df.index,  utc=True)
        # Remove intervals with price > 5000 AUD    
        df_filtered = df.loc[df['price'] <= 5000, :]
        try:
            grouped_mean = df_filtered.groupby(df_filtered.index.time).mean()
        except:
            print(df_filtered.index.time)
            raise ValueError
        return grouped_mean
    
    #TODO: define functions to get x percentile price 
    #TODO: define function to get start_time and end_time when price is lower than X percentikle
    #TODO: define function to get start_time and end_time when price is greater than X percentikle
    
    
    def _x_percentile_price(self, df:DataFrame|None = None, percentile = 0.95):
        
        df = df or self._get_data_as_pd()
        # Calculate the 95th percentile price
        percentile_val = df['price'].quantile(percentile)
        return percentile_val
    
    
    def _find_continuous_sequences(self, forecast, percentile, upper = True):
        
        df = forecast.copy()
        df.set_index("interval_start_time", inplace = True)
        sequences = []
        current_start = None
        current_end = None
        
        if upper:
            df = df[df['price'] >= percentile]
        else:
            df = df[df['price'] <= percentile]
            
        

        # Iterate through the DataFrame sorted by index
        for index, row in df.sort_index().iterrows():
            if current_start is None:
                # First element, set start and end
                current_start = index
                current_end = index
            else:
                
                expected_next_time = (current_end + timedelta(minutes=5)).time()
                if index == expected_next_time:
                    # Update end time 
                    current_end = index
                else:
                    # End of sequence, append to results and reset
                    sequences.append((current_start, current_end))
                    current_start = index
                    current_end = index

        # Append the last sequence if it exists
        if current_start is not None:
            sequences.append((current_start, current_end))

        return sequences
    
    
    def _update_soe(self, soe_df, sequence, updated_val, col_name = "soe_high"):
        
        # Iterate over the sequence
        for i in range(0, len(sequence), 2):
            start_time_1, end_time_1 = sequence[i]
            # Check if start_time_1 is before 12:00
            start_idx = soe_df.index.searchsorted(pd.Timestamp(start_time_1.strftime('%H:%M')))
            end_idx = soe_df.index.searchsorted(pd.Timestamp(end_time_1.strftime('%H:%M')))
            # Update 'soe' column within the specified range
            soe_df.loc[soe_df.index[start_idx:end_idx], col_name] = updated_val
                
                
        return soe_df
        
    
    
    def get_soft_soe_limits(self, final_pf):
        
        
        soe_dummy_df = pd.DataFrame(index=pd.date_range("00:00", "23:55", freq="5min"))
        soe_dummy_df['soe_high'] = 12.5
        soe_dummy_df['soe_low'] = 1
        
        p95 = self._x_percentile_price(None, 0.95)
        p90 = self._x_percentile_price(None, 0.90)
        p10 = self._x_percentile_price(None, 0.10)
        p5  = self._x_percentile_price(None, 0.05)
        
        p95_sequence = self._find_continuous_sequences(final_pf, p95)
        p90_sequence = self._find_continuous_sequences(final_pf, p90)
        p10_sequence = self._find_continuous_sequences(final_pf, p10, False)
        p5_sequence  = self._find_continuous_sequences(final_pf, p5, False)
        
        
        soe_dummy_df = self._update_soe(soe_dummy_df, p95_sequence, 13)
        soe_dummy_df = self._update_soe(soe_dummy_df, p5_sequence, 0, 'soe_low')
        
        return soe_dummy_df
         
    def simple_forecast(self,
                        forecast_start_utc:Union[datetime, None]=None,
                        price:float = 0.0,
                        pv_power:float = 0.0
                        )->DataFrame:
        if forecast_start_utc is not None:
            self._timestamps.append(forecast_start_utc)
            self._price_history.append(price)
            self._pv_power.append(pv_power)
        forecast = self._get_time_of_day_avg()
                
        # price_above_high_df, price_below_low_df = self._x_percentile_price(forecast, percentile = 0.95)
        # high_price_event_start_end = self._find_continuous_sequences(price_above_high_df)
        # low_price_event_start_end = self._find_continuous_sequences(price_below_low_df)
           
        if forecast_start_utc is None:
            forecast_start_utc = current_settlement_period_start_utc()
        # Step 1: Generate datetime index for specific dates with 5-minute resolution
        # Create a datetime object with a fixed date and the time from forecast.index[0]
        forecast_index = forecast.index[0]
        forecast_index_datetime = datetime.combine(forecast_start_utc.date(), forecast_index)
        forecast_start_utc = forecast_start_utc.replace(tzinfo=None)
        desired_shift = forecast_start_utc-forecast_index_datetime
        date_range_ = date_range(
            start=forecast_index_datetime,
            periods=len(forecast),
            freq='5min')
        simple_forecast_df = DataFrame()
        simple_forecast_df["interval_start_time"] = date_range_
        simple_forecast_df["price"] = list(forecast["price"])
        simple_forecast_df["pv_power"] = list(forecast["pv_power"])
        simple_forecast_df["inteval_duration"] = timedelta(minutes=5)
        total_prediction_duration = simple_forecast_df["inteval_duration"].sum()
        next_day_df = simple_forecast_df.copy()
        next_day_df["interval_start_time"] = next_day_df[
            "interval_start_time"] + total_prediction_duration
        appended_df = concat([simple_forecast_df, next_day_df])
        shifted_df = appended_df.shift(-int(desired_shift.total_seconds()/300))
        return shifted_df[0:len(forecast)]
        
