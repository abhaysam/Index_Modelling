import datetime as dt

from index_model.index import IndexModel, RiskReturn

if __name__ == "__main__":
    backtest_start = dt.date(year=2020, month=1, day=1)
    backtest_end = dt.date(year=2020, month=12, day=31)
    index = IndexModel(backtest_start, backtest_end,[0.5, 0.25, 0.25])

    index_level = index.index_price.Index_Level
    index.plot_prices()
    perforamnce = RiskReturn(backtest_start, backtest_end)
    perforamnce.annualized_perf()
    index.export_values("export.csv")
