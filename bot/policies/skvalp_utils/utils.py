import pytz
from datetime import datetime

def floor_to_nearest_5_minutes(dt:datetime)->datetime:
    minute = dt.minute
    minute_floor = minute - (minute % 5)
    return dt.replace(minute=minute_floor, second=0, microsecond=0)

def current_settlement_period_start_utc() ->datetime:
    return floor_to_nearest_5_minutes(datetime.now(pytz.utc))
