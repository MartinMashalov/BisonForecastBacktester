"""main bison backtester functionality"""

import pandas as pd
import numpy as np
import ta
import yfinance as yf
import pandas_ta as pd_ta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
import warnings
from tqdm import tqdm
from time import sleep
warnings.filterwarnings("ignore")

# global variable(s)
start = '2019-01-01'

def main_test(end: str, rsi_mode: bool = False, steps: int = 30, ticker: str = '^GSPC'):
    return {'Day1': (0, 1, 0), 'Day2': (0, 1, 0), 'Day3': (0, 1, 0), 'Day4': (0, 1, 0), 'Day5': (0, 1, 0),
            'Day6': (0, 1, 0), 'Day7': (0, 1, 0), 'Day8': (0, 1, 0), 'Day9': (0, 1, 0), 'Day10': (0, 1, 0),
            'Day11': (0, 1, 0), 'Day12': (0, 1, 0), 'Day13': (0, 1, 0), 'Day14': (0, 1, 0), 'Day15': (0, 0, 1),
            'Day16': (0, 0, 1), 'Day17': (0, 0, 1), 'Day18': (0, 0, 1), 'Day19': (0, 0, 1), 'Day20': (0, 0, 1),
            'Day21': (0, 0, 1), 'Day22': (0, 0, 1), 'Day23': (0, 0, 1), 'Day24': (0, 0, 1), 'Day25': (0, 0, 1),
            'Day26': (0, 0, 1), 'Day27': (0, 0, 1), 'Day28': (0, 0, 1), 'Day29': (0, 0, 1)}

# change end date here
def main(end: str, rsi_mode: bool = False, steps: int = 30, ticker: str = '^GSPC') -> dict:
    """main function to serve all forecasting purposes"""
    df = pd.read_csv(f'data/SPX_{end}.csv')

    # rename columns
    df.rename(columns={'Open': 'open_price',
                       'High': 'highest_price',
                       'Close': 'close_price',
                       'Low': 'lowest_price',
                       'Volume': 'volume'}, inplace=True)
    df.index = df['Date']
    df['local_time'] = df.index
    #df['time'] = df.local_time.values.astype(np.int64) // 10 ** 9
    df['local_time'] = df['local_time'].apply(lambda x: str(x).split(' ')[0] + ' 22:30:00')
    df['local_time'] = pd.to_datetime(df['local_time'])
    df.reset_index(inplace=True, drop=True)
    df.drop('Date', axis=1, inplace=True)
    df['stock'] = ['SPX' for _ in range(df.shape[0])]
    df = df[['stock', 'local_time', 'open_price', 'close_price', 'highest_price', 'lowest_price', 'volume']]

    if rsi_mode:
        length = steps
        df['close_price'] = pd_ta.rsi(df['close_price'], length=length)
        df = df.iloc[length:]

    # create walk-forward backtesting setup
    results_df = df[-steps:]
    df = df[:-steps]

    # create gain column
    def label_gain(data, closed, opened):
        df['gain'] = ''
        df.loc[(closed > opened), 'gain'] = 1
        df.loc[(closed < opened), 'gain'] = -1
        df.loc[(closed == opened), 'gain'] = 0

        return df

    df = label_gain(df, df['close_price'], df['open_price'])

    # create reversal column
    df['ema_3'] = ta.trend.ema_indicator(close=df['close_price'], window=3, fillna=True)
    for i in range(1, 11):
        df['pct_change_' + str(i)] = df['close_price'].pct_change(-i) * 100

    signal = ''  # Create Empty Signal for buy and sell toggling
    df['reversal'] = 0

    for index in range(len(df)):
        try:
            if df['stock'][index] == df['stock'][index + 2]:
                if signal == "buy":  # If toggle is at buy, no more buy signals until first sell signal (2)
                    if (df['gain'][index] == -1) & (df['gain'][index - 1] == 1):
                        if (df['pct_change_10'][index - 1] >= 3.75) or (df['pct_change_8'][index - 1] >= 3.75) or (
                                df['pct_change_1'][
                                    index - 1] >= 3.75):  # or (df['pct_change_6'][index-1] >= 0.363.75)
                            if df['ema_3'][index + 2] > df['close_price'][index + 2]:
                                df['reversal'][index - 1] = 2
                                signal = 'sell'  # Change signal to Sell

                elif signal == 'sell':  # If toggle is at sell, no more sell signals until first buy signal(1)
                    if (df['gain'][index] == 1) & (df['gain'][index - 1] == -1):
                        if (df['pct_change_10'][index - 1] <= -3.75) or (df['pct_change_8'][index - 1] <= -3.75) or (
                                df['pct_change_1'][
                                    index - 1] <= -3.75):  # or (df['pct_change_6'][index-1] <= -0.363.75) or
                            if (df['pct_change_10'][index] >= -100) or (df['pct_change_9'][index] >= -100) or (
                                    df['pct_change_8'][index] >= -100) or (df['pct_change_7'][index] >= -100) or (
                                    df['pct_change_6'][index] >= -100):
                                if df['ema_3'][index + 2] < df['close_price'][index + 2]:
                                    df['reversal'][index - 1] = 1
                                    signal = 'buy'  # Change signal to Buy

                else:  # At the start where there is no signals yet
                    if (df['gain'][index] == -1) & (df['gain'][index - 1] == 1):
                        if (df['pct_change_10'][index - 1] >= 3.75) or (df['pct_change_8'][index - 1] >= 3.75) or (
                                df['pct_change_1'][
                                    index - 1] >= 3.75):  # or (df['pct_change_6'][index-1] >= 0.363.75) or
                            if df['ema_3'][index + 2] > df['close_price'][index + 2]:
                                df['reversal'][index - 1] = 2
                                signal = 'sell'  # Change signal to Sell
                    elif (df['gain'][index] == 1) & (df['gain'][index - 1] == -1):
                        if (df['pct_change_10'][index - 1] <= -3.75) or (df['pct_change_8'][index - 1] <= -3.75) or (
                                df['pct_change_1'][
                                    index - 1] <= -3.75):  # or (df['pct_change_6'][index-1] <= -0.363.75) or
                            if (df['pct_change_10'][index] >= -100) or (df['pct_change_9'][index] >= -100) or (
                                    df['pct_change_8'][index] >= -100) or (df['pct_change_7'][index] >= -100) or (
                                    df['pct_change_6'][index] >= -100):
                                if df['ema_3'][index + 2] < df['close_price'][index + 2]:
                                    df['reversal'][index - 1] = 1
                                    signal = 'buy'  # Change signal to Buy
                    else:
                        df['reversal'][index - 1] = 0

            else:
                df['reversal'][index - 1] = 0
        except:
            pass

    df.set_index('local_time', drop=True, inplace=True)

    # split datasets
    aapl = df[df['stock'] == "SPX"]
    aapl_sarima = aapl[['close_price']]
    train_data, test_data = aapl_sarima[0:int(len(aapl_sarima) - 100)], aapl_sarima[int(len(aapl_sarima) - 100):]

    # data type management
    train_ar = train_data.values
    test_ar = test_data.values
    history = [x for x in train_ar]
    predictions_aapl = list()

    # run SARIMA
    for t in tqdm(range(len(test_ar))):
        if (t == len(test_ar) - 1):
            model = SARIMAX(history, order=(3, 0, 3), seasonal_order=(3, 1, 0, 8))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast(steps=steps)
            for i in output:
                yhat = i
                predictions_aapl.append(yhat)
            obs = test_ar[t]
            history.append(obs)
        else:
            model = SARIMAX(history, order=(3, 0, 3), seasonal_order=(3, 1, 0, 8))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions_aapl.append(yhat)
            obs = test_ar[t]
            history.append(obs)

    # create result storage containers
    forecast_arr = predictions_aapl[len(test_data):]

    # linear regression and smoothing with reference point choosing
    y = [[i] for i in forecast_arr]
    X = [[i] for i in range(len(forecast_arr))]
    model = LinearRegression()
    model.fit(X, y)
    forecast_lreg = model.predict(X)
    dir_reference = float(test_data[['close_price']].tail(1).iloc[0])
    result_storage: dict = {}

    # calculate direction and correction from each day forward in range of steps variable
    for i, (forecast, actual) in enumerate(zip(forecast_lreg, results_df['close_price'][:steps - 1].values)):
        # find actual trend
        if actual > dir_reference:
            actual_trend = 1
        else:
            actual_trend = 0

        # find forecasted trend
        if forecast > dir_reference:
            forecast_trend = 1
        else:
            forecast_trend = 0

            # check correctness
        if forecast_trend == actual_trend:
            correct = 1
        else:
            correct = 0
        result_storage[f'Day{i + 1}'] = (forecast_trend, actual_trend, correct)

    return result_storage

def main_intraday(end: str, rsi_mode: bool = False, steps: int = 30, ticker: str = '^GSPC') -> dict:
    """intraday backtesting functionality on 30m timeframe"""

    # read csv file
    df = pd.read_csv(f'intra_data/SPX_{end}.csv')

    # filter out after market and before market data
    df['time'] = df['time'].apply(lambda x: str(x).split(' ')[1])
    accepted_times: list = ['09:30:00', '10:00:00', '10:30:00', '11:00:00', '11:30:00', '12:00:00',
                            '12:30:00', '13:00:00', '13:30:00', '14:00:00', '14:30:00', '15:00:00',
                            '15:30:00', '16:00:00']
    df = df[df['time'].isin(accepted_times)]

    # rename dataframe columns
    df.rename(columns={'open': 'open_price',
                       'high': 'highest_price',
                       'close': 'close_price',
                       'low': 'lowest_price',
                       'volume': 'volume'}, inplace=True)

    # filter dataframe based on start and end times
    df['day'] = df['time'].apply(lambda x: str(x).split(' ')[0])
    df.index = df['day']
    df.index = df['time']

    # create walk-forward backtesting setup
    results_df = df[-steps:]
    df = df[:-steps]

    # create gain column
    def label_gain(data, closed, opened):
        df['gain'] = ''
        df.loc[(closed > opened), 'gain'] = 1
        df.loc[(closed < opened), 'gain'] = -1
        df.loc[(closed == opened), 'gain'] = 0

        return df

    df = label_gain(df, df['close_price'], df['open_price'])

    # create reversal column
    df['ema_3'] = ta.trend.ema_indicator(close=df['close_price'], window=3, fillna=True)
    for i in range(1, 11):
        df['pct_change_' + str(i)] = df['close_price'].pct_change(-i) * 100

    signal = ''  # Create Empty Signal for buy and sell toggling
    df['reversal'] = 0

    for index in range(len(df)):
        try:
            if df['stock'][index] == df['stock'][index + 2]:
                if signal == "buy":  # If toggle is at buy, no more buy signals until first sell signal (2)
                    if (df['gain'][index] == -1) & (df['gain'][index - 1] == 1):
                        if (df['pct_change_10'][index - 1] >= 3.75) or (df['pct_change_8'][index - 1] >= 3.75) or (
                                df['pct_change_1'][
                                    index - 1] >= 3.75):  # or (df['pct_change_6'][index-1] >= 0.363.75)
                            if df['ema_3'][index + 2] > df['close_price'][index + 2]:
                                df['reversal'][index - 1] = 2
                                signal = 'sell'  # Change signal to Sell

                elif signal == 'sell':  # If toggle is at sell, no more sell signals until first buy signal(1)
                    if (df['gain'][index] == 1) & (df['gain'][index - 1] == -1):
                        if (df['pct_change_10'][index - 1] <= -3.75) or (df['pct_change_8'][index - 1] <= -3.75) or (
                                df['pct_change_1'][
                                    index - 1] <= -3.75):  # or (df['pct_change_6'][index-1] <= -0.363.75) or
                            if (df['pct_change_10'][index] >= -100) or (df['pct_change_9'][index] >= -100) or (
                                    df['pct_change_8'][index] >= -100) or (df['pct_change_7'][index] >= -100) or (
                                    df['pct_change_6'][index] >= -100):
                                if df['ema_3'][index + 2] < df['close_price'][index + 2]:
                                    df['reversal'][index - 1] = 1
                                    signal = 'buy'  # Change signal to Buy

                else:  # At the start where there is no signals yet
                    if (df['gain'][index] == -1) & (df['gain'][index - 1] == 1):
                        if (df['pct_change_10'][index - 1] >= 3.75) or (df['pct_change_8'][index - 1] >= 3.75) or (
                                df['pct_change_1'][
                                    index - 1] >= 3.75):  # or (df['pct_change_6'][index-1] >= 0.363.75) or
                            if df['ema_3'][index + 2] > df['close_price'][index + 2]:
                                df['reversal'][index - 1] = 2
                                signal = 'sell'  # Change signal to Sell
                    elif (df['gain'][index] == 1) & (df['gain'][index - 1] == -1):
                        if (df['pct_change_10'][index - 1] <= -3.75) or (df['pct_change_8'][index - 1] <= -3.75) or (
                                df['pct_change_1'][
                                    index - 1] <= -3.75):  # or (df['pct_change_6'][index-1] <= -0.363.75) or
                            if (df['pct_change_10'][index] >= -100) or (df['pct_change_9'][index] >= -100) or (
                                    df['pct_change_8'][index] >= -100) or (df['pct_change_7'][index] >= -100) or (
                                    df['pct_change_6'][index] >= -100):
                                if df['ema_3'][index + 2] < df['close_price'][index + 2]:
                                    df['reversal'][index - 1] = 1
                                    signal = 'buy'  # Change signal to Buy
                    else:
                        df['reversal'][index - 1] = 0

            else:
                df['reversal'][index - 1] = 0
        except:
            pass

    df.set_index('time', drop=True, inplace=True)
    if rsi_mode:
        length = steps
        df['close_price'] = pd_ta.rsi(df['close_price'], length=length)
        df = df.iloc[length:]

    # split datasets
    aapl = df
    aapl_sarima = aapl[['close_price']]
    train_data, test_data = aapl_sarima[0: int(len(aapl_sarima) - 100)], aapl_sarima[int(len(aapl_sarima) - 100): ]

    # data type management
    train_ar = train_data.values
    test_ar = test_data.values
    history = [x for x in train_ar]
    predictions_aapl = list()

    # run SARIMA
    for t in tqdm(range(len(test_ar))):
        if (t == len(test_ar) - 1):
            model = SARIMAX(history, order=(3, 0, 3), seasonal_order=(3, 1, 0, 8))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast(steps=steps)
            for i in output:
                yhat = i
                predictions_aapl.append(yhat)
            obs = test_ar[t]
            history.append(obs)
        else:
            model = SARIMAX(history, order=(3, 0, 3), seasonal_order=(3, 1, 0, 8))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions_aapl.append(yhat)
            obs = test_ar[t]
            history.append(obs)

    # create result storage containers
    forecast_arr = predictions_aapl[len(test_data):]

    # linear regression and smoothing with reference point choosing
    y = [[i] for i in forecast_arr]
    X = [[i] for i in range(len(forecast_arr))]
    model = LinearRegression()
    model.fit(X, y)
    forecast_lreg = model.predict(X)
    dir_reference = float(test_data[['close_price']].tail(1).iloc[0])
    result_storage: dict = {}

    # calculate direction and correction from each day forward in range of steps variable
    for i, (forecast, actual) in enumerate(zip(forecast_lreg, results_df['close_price'][:steps - 1].values)):
        # find actual trend
        if actual > dir_reference:
            actual_trend = 1
        else:
            actual_trend = 0

        # find forecasted trend
        if forecast > dir_reference:
            forecast_trend = 1
        else:
            forecast_trend = 0

            # check correctness
        if forecast_trend == actual_trend:
            correct = 1
        else:
            correct = 0
        result_storage[f'Day{i + 1}'] = (forecast_trend, actual_trend, correct)

    return result_storage

#if __name__ == '__main__':
    #print(main_intraday('2022-11-02', rsi_mode=False, steps=30))