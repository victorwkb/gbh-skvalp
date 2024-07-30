from datetime import datetime, timedelta
from policies.policy import Policy
import logging
import pandas as pd
import sys, os
from dateutil import parser
from collections import deque


current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, 'skvalp_utils'))

from basic_predictions import BasicForecaster
from utils import current_settlement_period_start_utc, floor_to_nearest_5_minutes
from optimiser_ideal_bess_pv_tariff import find_optimal_discharge
# from ml_based_priceprediction import MLBasedPricePredictor
#from dart_priceprediction import MLBasedPricePredictor
from dart_priceprediction_comb import MLBasedPricePredictor

from environment import TIMESTAMP_KEY


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)


class SreshtaGyan(Policy):
    def __init__(self, **kwargs):
        super().__init__()
        self._round_trip_efficiency=kwargs.get("efficiency", 1.0)
        self._battery_capacity = kwargs.get("bess_capacity",15)
        self._max_grid_import = kwargs.get("max_grid_import", 100)
        self._max_grid_export = kwargs.get("max_grid_export", 100)
        #TODO: find SOE limit vectors here before enabling soft limits
        # Simple price forecast for the next 24 hours
        self._consider_dynamic_soe_upper_limit = kwargs.get(
            "consider_dynamic_soe_upper_limit", False)
        self._consider_dynamic_soe_lower_limit = kwargs.get(
            "consider_dynamic_soe_lower_limit", False)
        use_ML_forecaster = kwargs.get("use_ml_forecaster", True)
        self._forecaster = BasicForecaster()
        self._simulation_start_utc = kwargs.get("start_time_utc",
                                                current_settlement_period_start_utc())
        self._detected_input_timezone:str|None = None
        self._last_market_info:dict|None=None
        
        self.ml_price_forecast_obj = MLBasedPricePredictor() if use_ML_forecaster else None
        
        self.solar_1hour_forecast  = deque(maxlen=12)
        self.solar_2hour_forecast  = deque(maxlen=24)
        self.solar_24hour_forecast  = deque(maxlen=288)
        
    def _detect_and_covert_input_timezone_to_utc(self,external_state:dict|pd.Series)->dict:
        if isinstance(external_state, pd.Series):
            external_state = external_state.to_dict()
        timestamp = external_state[TIMESTAMP_KEY]
        # timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        timestamp = parser.parse(timestamp)
        hour_of_day = timestamp.hour
        sun_is_up = (external_state['pv_power'] > 0.001)
        if self._detected_input_timezone is None:
            # Try detect the timezone
            if (hour_of_day == 23) and sun_is_up:
                # if there is solar power at midnight, then it is UTC
                self._detected_input_timezone = "utc"
                logger.info("Input timezone detected as UTC")
            if (hour_of_day == 14) and sun_is_up:
                # if there is solar power at midnight, then it is UTC
                self._detected_input_timezone = "nem_time"
                logger.info("Input timezone detected as nem_time")
            else:
                external_state["interval_start_time"] = timestamp
        if self._detected_input_timezone is not None:
            if self._detected_input_timezone == "utc":
                if  (hour_of_day == 14) and sun_is_up:
                    logger.error("there shouldn't be solar power in SA during midday UTC")
                external_state["interval_start_time"] = timestamp
            elif self._detected_input_timezone == "nem_time":
                if (hour_of_day == 23) and sun_is_up:
                    logger.error("there shouldn't be solar power in SA during midnight AEST")
                external_state["interval_start_time"] = timestamp-timedelta(hours=10)
            else:
                raise ValueError("Unknown key for detected timezone")
        return external_state
    
    def _rundown_price(self,remaining_steps:int,price_forecast_list:list[float]):
        if remaining_steps >=288:
            return 0.0
        prev_24_hour_data = list(self.ml_price_forecast_obj.price_history)
        req_histosic_data = prev_24_hour_data[-(288-remaining_steps):]
        req_forecast_data = price_forecast_list[:remaining_steps]
        return (sum(req_histosic_data)+sum(req_forecast_data))/288
    
    def act(self, external_state, internal_state):
        
        try:
        
            external_data =  external_state
            # enhance info with further details about the battery
            internal_state["capacity"] = self._battery_capacity
            internal_state["efficiency"] = self._round_trip_efficiency
            internal_state["max_grid_import"] = self._max_grid_import
            internal_state["max_grid_export"] = self._max_grid_export
            internal_state["max_discharge_rate"] = internal_state["max_charge_rate"]
            external_state=self._detect_and_covert_input_timezone_to_utc(external_state)
            external_state=self._update_market_info(external_state)
            self.solar_1hour_forecast.append(external_state['pv_power_forecast_1h'])
            self.solar_2hour_forecast.append(external_state['pv_power_forecast_2h'])
            self.solar_24hour_forecast.append(external_state['pv_power_forecast_24h'])
            remaining_steps = internal_state["remaining_steps"]
            # Example logic: return some calculated action based on the observation and info
            """
            generating price forecast
            """
            offset_naive_time = external_state["interval_start_time"].replace(tzinfo=None)
            
            # ML based price forecast for the next 1 or 3 hours in 5 mins resolution
            if self.ml_price_forecast_obj is not None:
                ml_price_forecast = self.ml_price_forecast_obj.act(external_data, internal_state)
            else:
                ml_price_forecast = None

            forecast = self._forecaster.simple_forecast(
                forecast_start_utc=external_state["interval_start_time"],
                price=external_state['price'],
                pv_power = external_state['pv_power']
                )
            
            
            #--------------------------------------------------------
            # Using the solcast forecast data
            #--------------------------------------------------------
            
            pv_forecast_update = None

            if len(self.solar_1hour_forecast) >= 12 and len(self.solar_2hour_forecast) < 24:
                pv_forecast_update = list(self.solar_1hour_forecast) 
                
            elif len(self.solar_2hour_forecast) >= 24 and len(self.solar_24hour_forecast) < 288:
                            
                pv_forecast_update = list(self.solar_1hour_forecast)  + list(self.solar_2hour_forecast)[12:]

            elif len(self.solar_24hour_forecast) >= 288:
                            
                pv_forecast_update = list(self.solar_1hour_forecast)  + list(self.solar_2hour_forecast)[12:] + list(self.solar_24hour_forecast)[24:]
                            
            
            if pv_forecast_update is not None:
                try:
                    pv_forecast_update.insert(0, external_state['pv_power'])
                    length = min(len(pv_forecast_update), 288)
                    forecast.loc[:length, "pv_power"] = pv_forecast_update[:length]
                except:
                    pass
            
                
            # print("-------------------------")    
            print(internal_state["remaining_steps"])
            # print(self.solar_1hour_forecast)
            # print(self.solar_2hour_forecast)
            # print(pv_forecast_update)
            # print("-------------------------")
            
            # combining the ML based forecast with the 24 hour forecast
            if ml_price_forecast is not None and len(ml_price_forecast) > 0:
                if len(ml_price_forecast) > len(forecast):
                    logger.warning("Potential forecast resolution mismatch")
                else:
                    logger.debug("Appending ML price forecast to 24 hour forecast")
                    forecast.loc[:len(ml_price_forecast)-1, "price"] = ml_price_forecast.flatten().tolist()  
                        
            forecast.loc[forecast['interval_start_time'] == offset_naive_time,
                            "price"] = external_state['price']
            forecast.loc[forecast['interval_start_time'] == offset_naive_time,
                    "pv_power"] = external_state['pv_power']
            

            # Clipping the forecast for last case
            # Get the first n rows if length greater than n, else take the whole DataFrame
            if len(forecast) > remaining_steps:
                forecast = forecast.head(remaining_steps)

            rundown_price = self._rundown_price(
                remaining_steps=remaining_steps,
                price_forecast_list=forecast['price'].tolist())
            
            '''       
            implementing the optimizer logic
            
            '''
            #TODO: pass the SOE limit vectors here if enabling soft limits
            soe_limits = self._forecaster.get_soft_soe_limits(forecast)
            solar_kW_to_battery, charge_kW = find_optimal_discharge(forecasts=forecast,
                        bess_info=internal_state,
                        rundown_price=rundown_price,
                        soe_upper_soft_limits = soe_limits['soe_high'].tolist() if self._consider_dynamic_soe_upper_limit
                        else None,
                        soe_lower_soft_limits = soe_limits['soe_low'].tolist() if self._consider_dynamic_soe_lower_limit
                        else None)
            
            return solar_kW_to_battery, charge_kW
        except:
            print("Error in act SreshtGyan")

    def load_historical(self, external_states):
        pass

    def _update_market_info(self, market_observation:dict)->dict:
        
        if isinstance(market_observation, pd.Series):
            market_observation = market_observation.to_dict()
        if "interval_start_time" not in market_observation:
            logger.warning('interval_start_time not specified in market inforamtion')
            if self._last_market_info is None:
                market_observation["interval_start_time"] = floor_to_nearest_5_minutes(
                    self._simulation_start_utc)
            else:
                market_observation["interval_start_time"] = self._last_market_info["interval_start_time"] + timedelta(minutes=5)
            print("setting price interval start time as {}".format(market_observation["interval_start_time"]))
        self._last_market_info = market_observation
        return market_observation
