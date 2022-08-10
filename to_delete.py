import datetime as dt
import pandas as pd
from index_model.index import IndexModel
backtest_start = dt.date(year=2020, month=1, day=1)
backtest_end = dt.date(year=2020, month=12, day=31)


start = pd.Timestamp(backtest_start)
end = pd.Timestamp(backtest_end)
weights = [0.5, 0.25, 0.25]
IndexModel(backtest_start,backtest_end)