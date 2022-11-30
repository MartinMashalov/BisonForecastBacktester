"""main runner file for backtesting -> assign settings in BacktestSettings model"""

from main import main as daily_backtest
from main import main_intraday as intraday_backtest
from main import main_test
import csv
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
from pydantic import BaseModel
from typing import Any, Iterable
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from functools import partial
import requests

class BacktestSettings(BaseModel):
    """backtesting settings global variable collection"""
    start_date: str = '2022-01-21'
    end_date: str = '2022-11-04'
    threads_count: int = 10
    exchange: str = 'NYSE'
    ticker: str = '^GSPC'
    rsi_mode: bool = False
    forecast_sarimax_steps: int = 30
    backtest_mode: Any = intraday_backtest

class Runner:
    """backtesting functionality with running methods"""

    def __init__(self, testing_instances: list, threads: int, settings):
        """variable initialization"""
        self.testing_instances = [str(i).replace(' 00:00:00', '') for i in testing_instances]
        self.threads = threads
        self.settings = settings
        self.backtesting_func = self.settings.backtest_mode

    @staticmethod
    def _custom_csv_writer(file_name: str, col_names: Iterable, data: Iterable) -> None:
        """custom function for writing backtesting results to csv file"""
        with open(file_name, 'w', encoding='UTF8') as f:
            # write object and create header
            writer = csv.writer(f)
            writer.writerow(col_names)

            # loop through data collection and record each line
            for row in data:
                writer.writerow(row)

    def _download_polygon(self, test_instance: str):
        """download data from polygon backend"""

        # subtract two months from test instance date
        date_format = '%Y-%m-%d'
        dtObj = datetime.strptime(test_instance, date_format)
        past_date = dtObj - relativedelta(months=3)
        past_date = past_date.strftime(date_format)

        # find starting point
        request_url: str = f'https://api.polygon.io/v2/aggs/ticker/SPY/range/30/minute/{past_date}/{test_instance}?' \
                           f'adjusted=true&sort=asc&limit=50000&apiKey=eTD2a0gOvakkPjBpyYBqRgiWY9CLJ0ot'

        # web download
        data = requests.get(request_url).json()
        container: list = []
        for entry in data['results']:
            container.append([
                datetime.fromtimestamp(entry['t'] / 1000), entry['o'], entry['h'], entry['l'], entry['c'], entry['v']
            ])

        # define dataframe
        df = pd.DataFrame(container, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        return df

    def _download_backtesting_data(self, intraday=False):
        """download all the backtesting data from yfinance"""
        if intraday:
            for instance in tqdm(self.testing_instances):
                self._download_polygon(instance).to_csv(f"intra_data/SPX_{instance}.csv")
        else:
            for instance in tqdm(self.testing_instances):
                df = yf.Ticker(self.settings.ticker).history(interval='1d', start='2019-01-01', end=instance)
                df.to_csv(f"data/SPX_{instance}.csv")

    def _input_arr_partition(self, arr=None) -> Any: # generator
        """split the input array of days into the number of threads available"""

        # handle array input to splitting function
        if arr is None:
            arr = self.testing_instances

        # compute number of partitions necessary to create groups of length *threads*
        partition_count: int = len(arr)//self.threads

        # compute partition
        k, m = divmod(len(arr), partition_count)
        return (arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(partition_count))

    def _assign_threads_run(self):
        """assign threads and execute them"""

        # define results container
        results_container: list = []

        # define partitioned inputs
        test_days_split = list(self._input_arr_partition())
        rsi_mode_split = list(self._input_arr_partition([self.settings.rsi_mode for _ in range(
            len(self.testing_instances))]))
        steps_split = list(self._input_arr_partition([self.settings.forecast_sarimax_steps for _ in range(
            len(self.testing_instances))]))
        ticker_split = list(self._input_arr_partition([self.settings.ticker for _ in range(
            len(self.testing_instances))]))

        # execute loop on all items assigning threads in the process
        for instance_group, rsi_mode_group, steps_group, ticker_group in tqdm(zip(test_days_split,
                                                                            rsi_mode_split, steps_split, ticker_split)):
            # create context manager for threading application
            with PoolExecutor() as executor:
                # start generator for multithreading pool
                results_gen = executor.map(self.backtesting_func, instance_group, rsi_mode_group, steps_group,
                                           ticker_group)
                # add results to container for future processing
                for result in results_gen:
                    results_container.append(result)

        return results_container

    def facade(self, file_name: str) -> None:
        """central facade method for running entire class functionality externally"""

        # fetch backtesting results
        backtester_results = self._assign_threads_run()

        # write ACCURACY results to csv file
        accuracy_file_name, direction_file_name = file_name+'_ACCURACY.csv', file_name+"_DIRECTION.csv"
        files: list = [accuracy_file_name, direction_file_name]

        # get accuracies and directions to write
        accuracies, directions = [[acc[2] for acc in i.values()] for i in backtester_results], \
                                 [[dir[0] for dir in i.values()] for i in backtester_results]
        data_to_write: list = [accuracies, directions]
        col_names: list =  [[day_idx[0] for day_idx in i.items()] for i in backtester_results][0]

        # write to csv file
        for custom_file_name, data in zip(files, data_to_write):
            self._custom_csv_writer(custom_file_name, col_names, data)

# declare global variable container instance
backtest_vars = BacktestSettings()

# calendar market days
nyse: Any = mcal.get_calendar(backtest_vars.exchange)
test_days = list(nyse.schedule(start_date=backtest_vars.start_date, end_date=backtest_vars.end_date).index)

# declare runner class instance
runner = Runner(test_days, backtest_vars.threads_count, backtest_vars)
#runner.facade("SPX")
runner._download_backtesting_data(intraday=True)