import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.interactive(False)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sparse
from sklearn.utils import shuffle
import random
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.ensemble import VotingRegressor

# resampling
from scipy.interpolate import CubicSpline

# time conversion
from datetime import datetime
import pytz

# solar irradiance estimation
from solarpy import irradiance_on_plane

# boostrap resampling
from sklearn.utils import resample

from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import time
import warnings


def load_files():
    event = pd.read_csv("reduced_event.csv", low_memory=False)
    swp = pd.read_csv("reduced_swp.csv", low_memory=False)

    return event, swp


def extract_pool(event, swp_id):
    return event.loc[event['swimming_pool_id'] == swp_id]


def plot_pool(event, swp_id):
    pool_data = extract_pool(event, swp_id)
    times = pd.Series((pd.DatetimeIndex(pool_data['created']).asi8 / 10 ** 9).astype(np.int))

    n_times_ticks = 1000
    time_ticks_timestamp = []
    time_ticks_date = []
    delta = (times[len(times) - 1] - times[0]) / n_times_ticks
    for i in range(n_times_ticks):
        time_stamp = times[0] + i * delta
        time_ticks_timestamp.append(time_stamp)
        date = datetime.fromtimestamp(time_stamp)
        time_ticks_date.append("{}/{}/{}:{}H".format(date.day, date.month, date.year, date.hour))

    print(time_ticks_date)

    measures = pool_data.data_temperature
    plt.plot(times, measures)
    plt.xticks(time_ticks_timestamp, time_ticks_date)
    plt.show()

    from pmdarima.arima import auto_arima

    model = auto_arima(measures, start_p=1, start_q=1,
                       max_p=3, max_q=3, m=20,
                       start_P=0, seasonal=True,
                       d=1, D=1, trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)
    future_forecast = model.predict(n_periods=10)
    plt.plot(future_forecast)
    plt.show()


"""
    Split dataset into training set and test set. The split is made on the pools. Hence, the train set
    and test set contain different pools time series and features.

    Inputs:
        - X, y: dataset
        - train ratio: proportion in train set
        - random_state: reproducibility seed
    Returns
        - X_train, y_train, X_test, y_test: dataframes

    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def temperature_build_db_train_test(X, y, train_ratio=0.9, random_state=0):
    swp_ids = list(pd.unique(X["swimming_pool_id"]))
    random.Random(random_state).shuffle(swp_ids)

    n_pools = len(swp_ids)
    n_lines_train_set = int(np.ceil(n_pools * train_ratio))
    swp_ids_train = swp_ids[0:n_lines_train_set]
    swp_ids_test = swp_ids[n_lines_train_set:-1]

    X_train = X[X["swimming_pool_id"].isin(swp_ids_train)]
    y_train = y.loc[X["swimming_pool_id"].isin(swp_ids_train)]

    X_test = X[X["swimming_pool_id"].isin(swp_ids_test)]
    y_test = y.loc[X["swimming_pool_id"].isin(swp_ids_test)]

    return X_train, y_train, X_test, y_test


"""
    Build the dataset for model-based and RNN approaches
    Slices time series and add categorical features

    Inputs :
        - event: event dataframe 
        - swp: swp dataframe
        - time horizon (int) : number of past time observations to capture
        - time_delay (int): delay of the target w.r.t the last observation capture in explanartory variables
        - multiple_target (boolean): whether the dataset is for multiple prediction or single point prediction
                If False: only one point at time_delay in the future could be predicted
                If True: all the points from t+1 up to t+time_delay could be predicted

    Returns:
        - X : panda dataframe whose rows are samples identified by swp id and time
        - y : target value to predict (temperature)

    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def temperature_build_db(event, swp, time_horizon, time_delay, multiple_target=False):
    def swp_del_miss(swp):
        n_rows = swp.shape[0]
        # eliminate pools for which
        #  - type is missing
        #  - location is missing
        #  - kind is missing
        # -  volume capacity is missing
        # -  sanitizer_process
        # - equipment_heatings
        # - equipment_protections
        vars = ['type', 'location', 'kind', 'volume_capacity', 'sanitizer_process', 'equipment_protections',
                'equipment_heatings']
        to_keep = np.full((n_rows), True)
        for var in vars:
            to_keep = to_keep & np.logical_not(swp[var].isna())

        return swp[to_keep]

    # eliminate non-necessary swp variables
    def swp_var_of_interest(swp):
        vars_to_keep = set(['swimming_pool_id', 'type', 'location', 'kind', 'volume_capacity', 'sanitizer_process',
                            'equipment_protections', 'equipment_heatings'])
        vars_to_drop = set(swp.columns) - vars_to_keep
        swp = swp.drop(columns=vars_to_drop)
        return swp

    # joint between swp pools and event pools : keep only pools in both
    def joint_over_swp_id(event, swp):
        swp_id_to_keep = swp['swimming_pool_id']
        return event.loc[event['swimming_pool_id'].isin(swp_id_to_keep)]

    def event_del_miss(event):
        # eliminate pools for which
        #  - data_temperature is missing
        #  - weather_temp is missing
        #  - weather_humidity is missing
        # -  weather_pressure capacity is missing
        n_rows = event.shape[0]
        vars = ["swimming_pool_id", "data_temperature", "weather_temp", "weather_humidity", "weather_pressure"]
        to_keep = np.full((n_rows), True)
        for var in vars:
            to_keep = to_keep & np.logical_not(event[var].isna())

        return event[to_keep]

    def event_var_of_interest(event):
        vars_to_keep = set(
            ["swimming_pool_id", "created", "data_temperature", "weather_temp", "weather_humidity", "weather_pressure"])
        vars_to_drop = set(event.columns) - vars_to_keep
        event = event.drop(columns=vars_to_drop)
        return event

    # del pools for which there is not enough measurements
    def del_too_few_obs(event, swp, min_obs):
        swp_ids = swp['swimming_pool_id']

        # to improve
        """for id in swp_ids:
            measures = event.loc[event['swimming_pool_id'] == id]
            if len(measures) < min_obs:
                index = event[event['swimming_pool_id'] == id].index
                event.drop(index, inplace=True)"""

        counts = event['swimming_pool_id'].value_counts()
        event_ids_to_keep = counts[counts > min_obs].index

        swp_id_swp = swp["swimming_pool_id"]
        intersection = set(swp_id_swp).intersection(set(event_ids_to_keep))

        event = event.loc[event['swimming_pool_id'].isin(intersection)]
        swp = swp.loc[swp['swimming_pool_id'].isin(intersection)]

        event.reset_index(inplace=True)
        swp.reset_index(inplace=True)

        return event, swp

    event = event.copy()

    # swp = swp_del_miss(swp)
    swp = swp_var_of_interest(swp)
    event = joint_over_swp_id(event, swp)
    event = event_del_miss(event)
    event = event_var_of_interest(event)
    event, swp = del_too_few_obs(event, swp, min_obs=50)

    # datatime to timestamp
    event['created'] = pd.Series((pd.DatetimeIndex(event['created']).asi8 / 10 ** 9).astype(np.int))

    def build_X_y(event, swp, time_horizon, time_delay, multiple_target=False):
        swp_ids = swp['swimming_pool_id']
        n_pools = len(swp_ids)  # number of diff. pools

        n_measures = event.shape[0]

        swp_explanatory_variables = ['type', 'location', 'kind', 'sanitizer_process', 'equipment_protections',
                                     'equipment_heatings', 'volume_capacity']
        event_explanatory_variables = ["weather_temp", "weather_humidity", "weather_pressure"]

        categorical_variables = ['type', 'location', 'kind', 'sanitizer_process']

        # columns of the final dataset
        def create_columns():
            columns = []
            columns += ["swimming_pool_id"]
            columns += ["timestamp"]
            columns += swp_explanatory_variables
            columns += ['day_year', 'seconds_day']

            for i in range(time_delay + 1):  # solar irradiance from t to t+time_delay (included)
                if i == 0:
                    columns.append('solar_irradiance' + " t")
                else:
                    columns.append('solar_irradiance' + " t+" + str(i))

            for i in range(time_horizon):
                if i == 0:
                    columns.append("temp. t")
                else:
                    columns.append("temp. t-" + str(i))

            for var in event_explanatory_variables:
                for i in range(time_delay + 1):
                    if i == 0:
                        columns.append(var + " t")
                    else:
                        columns.append(var + " t+" + str(i))

            return columns

        columns = create_columns()
        print("Columns: {}".format(columns))

        # number of final samples in the data sset
        # n_measures: total number of measurements
        # time_horizon*n_pools: time_horizon size window is needed to begin (on each time series)
        # time_delay*n_pools: time_delay size window is needed to end  (on each time series)
        final_n_rows = n_measures - time_horizon * n_pools - time_delay * n_pools  # wait for the time_horizon first measures
        X = np.full((final_n_rows, len(columns)), None)

        if multiple_target:
            y = np.zeros((final_n_rows, time_delay))
        else:
            y = np.zeros((final_n_rows, 1))

        n_lines_filled = 0
        n_id = 0
        for id in swp_ids:
            print(n_id)
            n_id += 1
            swp_measures = event.loc[event['swimming_pool_id'] == id]
            swp_data = swp.loc[swp['swimming_pool_id'] == id].iloc[0]

            timestamps = swp_measures['created']
            timestamps_diffs = timestamps.diff()[1:]  # time delta between two an observation and the next

            """
                Resample a time series: the time diff. between two observations is irregular
                    Effect: apply cublic spline interpolation on one time series. The time series is sliced into
                    valid sub-time-series on which a cublic spline model is fitted and then sampled.
            """

            def resample():
                resampling_T = 4320  # 4320 sec ; 1.2 H ; is the device normal sampling period
                n_max_new_samples = int(np.ceil((timestamps.iloc[-1] - timestamps.iloc[0]) / resampling_T))

                resampled_measures = pd.DataFrame(
                    data=np.empty((n_max_new_samples, 2 + len(event_explanatory_variables))),
                    columns=['resampled_timestamp', 'data_temperature'] + event_explanatory_variables)

                """ 
                    Cut the time series into valid pieces
                    Output: valid ranges
                """
                # visit the time series
                # cut the series into not too far apart episodes
                time_series_ranges = []
                range_start = 0  # index start of the new episode
                for measure_index in range(0, swp_measures.shape[0] - 1):  # loop through timestamps_diffs

                    # there is a cutting point
                    # either too time points are too distant or too close
                    if timestamps_diffs.iloc[measure_index] > 10 * 3600 or timestamps_diffs.iloc[
                        measure_index] < 0.05 * 3600:
                        index_range = (range_start, measure_index)

                        if range_start != measure_index:  # don't add undefined ranges
                            time_series_ranges.append(index_range)

                        range_start = measure_index + 1

                """
                    Loop through the valid pieces. Create a model of each variable of interest in each piece.
                    Output: models per valid piece
                """
                # list of lists ; each list for a range
                # in each sub-list : models for each variable
                interpolation_models_X = []

                # measure to resample
                measures_array = swp_measures[['data_temperature'] + event_explanatory_variables].to_numpy()

                for time_range in time_series_ranges:
                    start = time_range[0]  # start of the valid piece
                    end = time_range[1]  # end of the valid piece (INCLUDED)

                    series_model = []

                    for index_var in range(measures_array.shape[1]):  # loop through var of interest
                        series_model.append(
                            CubicSpline(timestamps[start:end + 1], measures_array[start:end + 1, index_var]))

                    interpolation_models_X.append(((start, end), series_model))

                """
                    Loop through the models of each valid piece and sample at regular time interval.
                    Output : re-sampled time series
                """
                resampled_index = 0  # number of lines re-sampled
                for time_range in interpolation_models_X:
                    index_range = time_range[0]
                    start = index_range[0]
                    end = index_range[1]
                    models = time_range[1]

                    times = np.arange(timestamps.iloc[start], timestamps.iloc[end], resampling_T)
                    resampled_measures.iloc[resampled_index:resampled_index + len(times), 0] = times
                    var = 1
                    for model in models:
                        interpolation = model(times)
                        resampled_measures.iloc[resampled_index:resampled_index + len(times), var] = interpolation

                        var += 1

                    resampled_index += len(times)

                return resampled_measures.iloc[0:resampled_index, :]

            resampled_swp_measures = resample()
            resampled_timestamps_diffs = resampled_swp_measures['resampled_timestamp'].diff()[1:]

            # some time windows encounter great shift in the time difference between
            # two measures ; this makes the approximation incoherent if used as it is
            # everything from the time horizon past measures to the predictions on weather and the target
            # temperature must be time coherent
            def is_valid_time_window(measure_index):
                # looks for anomalies
                for i in range(-time_horizon, time_delay):
                    if resampled_timestamps_diffs.iloc[measure_index + i] > 10 * 3600:
                        return False

                return True

            """
                Loop on the time series from time_horizon to len(time_series) - time_delay and fill the datasets X y
            """
            for measure_index in range(time_horizon, len(resampled_swp_measures) - time_delay):
                if is_valid_time_window(measure_index) is False:
                    continue

                print(n_lines_filled)

                if multiple_target:
                    for j in range(time_delay):
                        y[n_lines_filled, j] = resampled_swp_measures['data_temperature'].iloc[measure_index + j + 1]
                else:
                    y[n_lines_filled, 0] = resampled_swp_measures['data_temperature'].iloc[measure_index + time_delay]

                # swimming_pool_id
                X[n_lines_filled, 0] = id

                # resampled timestamp
                X[n_lines_filled, 1] = resampled_swp_measures['resampled_timestamp'].iloc[measure_index + time_delay]

                # variables about swp : fixed pool features
                i_var = 2  # start
                for var in swp_explanatory_variables:
                    X[n_lines_filled, i_var] = swp_data[var]
                    i_var += 1

                """ Create time and irradiance features"""
                # extract Italy time
                date = datetime.fromtimestamp(resampled_swp_measures['resampled_timestamp'].iloc[measure_index],
                                              tz=pytz.timezone("Europe/Rome"))
                day_year = date.day
                sec_in_day = date.hour * 3600 + date.minute * 60 + date.second
                # day in the year (of the measurement)
                X[n_lines_filled, i_var] = day_year
                i_var += 1
                # seconds in the day (of the measurement)
                X[n_lines_filled, i_var] = sec_in_day
                i_var += 1

                # Add solar irradiance from t to t+time_delay (included)
                vnorm = np.array([0, 0, -1])  # plane pointing zenith
                h = 0  # sea-level
                lat = 42  # latitude of Roma (middle Italy, very approximative)
                for i in range(time_delay + 1):
                    date = datetime.fromtimestamp(resampled_swp_measures['resampled_timestamp'].iloc[measure_index + i])
                    X[n_lines_filled, i_var] = irradiance_on_plane(vnorm, h, date, lat)
                    i_var += 1

                # pool temperature from t to t-(time_horizon-1)
                for i_temp in range(time_horizon):
                    X[n_lines_filled, i_var] = resampled_swp_measures['data_temperature'].iloc[measure_index - i_temp]
                    i_var += 1

                # every other explanatory variables
                for var in event_explanatory_variables:
                    for i in range(time_delay + 1):
                        X[n_lines_filled, i_var] = resampled_swp_measures[var].iloc[measure_index + i]
                        i_var += 1

                n_lines_filled += 1

        X_df = pd.DataFrame(data=X[0:n_lines_filled], columns=columns)

        if multiple_target:
            y_columns = []
            for j in range(0, time_delay):
                y_columns.append('temp. t+' + str(j + 1))
        else:
            y_columns = ['temp. t+' + str(time_delay)]

        y_df = pd.DataFrame(data=y[0:n_lines_filled], columns=y_columns)

        return X_df, y_df

    X, y = build_X_y(event, swp, time_horizon=time_horizon, time_delay=time_delay, multiple_target=multiple_target)

    # eliminate pools with too few valid obs
    counts = X['swimming_pool_id'].value_counts()
    swp_id_to_keep = counts[counts > 25].index
    samples_to_keep = X['swimming_pool_id'].isin(swp_id_to_keep).index

    X = X.loc[samples_to_keep]
    y = y.loc[samples_to_keep]

    return X, y


"""
    Plot a temperature time series along with prediction and prediction interval

    Inputs:
        - ret_dict : return dictionnary by regression method
        - time_delays : times to plots ; allows to compare prediction of same point in time but performed
                        at different moments
        - colors : list of colors for each prediction time_delays
        - index_pool_test : if > 0: select this pool from the set of unique pool ids
                            if < 0: sort the pools by number of measurements and take the -index_pool_test th one
        - period : time between measurements in normal mode/resampled mode

    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def plot_predicted_temperatures(ret_dict, time_delays, colors, index_pool_test=0, period=4320):
    if "X_test_normalized" in ret_dict:
        X_test = ret_dict['X_test_normalized']
    else:
        X_test = ret_dict['X_test']

    swp_ids = np.unique(X_test[:, 0])

    if index_pool_test < 0:
        selected_id = pd.Series(X_test[:, 0]).value_counts().index[-index_pool_test]
        print(pd.Series(X_test[:, 0]).value_counts().index[-index_pool_test])
    else:
        selected_id = swp_ids[index_pool_test]

    X_test_id = X_test[:, 0] == selected_id

    # times of observation
    times_truth = X_test[X_test_id, 1]
    datetimes_truth = pd.to_datetime(times_truth, unit='s')
    y_truth = ret_dict['y_truth'][X_test_id]

    for i, time_delay in enumerate(time_delays):
        key = 't+' + str(time_delay)
        y_pred_low = ret_dict[key]['y_pred_low']
        y_pred_high = ret_dict[key]['y_pred_high']

        y_pred_low_id = y_pred_low[X_test_id]
        y_pred_high_id = y_pred_high[X_test_id]
        y_pred_mean_id = (y_pred_low_id + y_pred_high_id) / 2
        times_pred = X_test[X_test_id, 1] + time_delay * period  # lag

        """
            Slice the times series so that missing values are not shown. If the difference between two instants
            is larger than period, there is a cut: no interpolation in between.

            Inputs :
                - times: timestamps
                - y_pred_low, y_pred_high, y_pred_mean: times series
                - period: normal sampling period

            Returns:
                The sliced time series

            Author: Baptiste Debes ; b.debes@student.uliege.be
        """

        def slice_time_series(times, y_pred_low, y_pred_high, y_pred_mean, period=period):
            times_buffer = []
            y_pred_lows = []
            y_pred_highs = []
            y_pred_means = []

            diffs = np.diff(times)
            last = 0
            for i, diff in enumerate(diffs):
                if diff > period * 1.01:  # *1.01 just to be safe
                    times_buffer.append(times[last:i + 1])
                    y_pred_lows.append(y_pred_low[last:i + 1])
                    y_pred_highs.append(y_pred_high[last:i + 1])
                    y_pred_means.append(y_pred_mean[last:i + 1])
                    last = i + 1

            # add remaining
            times_buffer.append(times[last:])
            y_pred_lows.append(y_pred_low[last:])
            y_pred_highs.append(y_pred_high[last:])
            y_pred_means.append(y_pred_mean[last:])

            return times_buffer, y_pred_lows, y_pred_highs, y_pred_means

        times_sliced, y_pred_lows_id_sliced, y_pred_highs_id_sliced, y_pred_mean_id_sliced = \
            slice_time_series(times_pred, y_pred_low_id, y_pred_high_id, y_pred_mean_id, period)

        # datetimes_sliced = [datetime.fromtimestamp(ts, tz=pytz.timezone("Europe/Rome")) for ts in times_sliced]
        datetimes_sliced = [pd.to_datetime(ts, unit='s') for ts in times_sliced]

        color = colors[i]  # np.random.rand(3,)

        # plot all the slices
        for j in range(len(datetimes_sliced)):
            if j == len(datetimes_sliced) - 1:
                plt.fill_between(list(datetimes_sliced[j]), list(y_pred_lows_id_sliced[j]),
                                 list(y_pred_highs_id_sliced[j]), color=color,
                                 alpha=0.3, interpolate=False, step='pre',
                                 label="Quantiles space of t+{}".format(time_delay))
            else:
                plt.fill_between(list(datetimes_sliced[j]), list(y_pred_lows_id_sliced[j]),
                                 list(y_pred_highs_id_sliced[j]), color=color,
                                 alpha=0.3, interpolate=False, step='pre')

        # plot all the slices
        for j in range(len(datetimes_sliced)):
            if j == len(datetimes_sliced) - 1:
                plt.step(datetimes_sliced[j], y_pred_mean_id_sliced[j], color=color,
                         label="Main predictor of t+{}".format(time_delay))
            else:
                plt.step(datetimes_sliced[j], y_pred_mean_id_sliced[j], color=color)

    plt.scatter(datetimes_truth, y_truth, s=15, c='red', label="Measurements")
    plt.legend(fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel("Datetime", fontsize=15)
    plt.title("Temperature predictions with 0.05-0.95 quantiles", fontsize=22)
    plt.ylabel("Temperature in ° C", fontsize=15)
    plt.show()


"""
    Benchmark implemented models. Produces bar plots with uncertainty bar on all their metrics.
    Perfoms cross-validation to obtain means and std.

    Inputs:
        - X, y: datasets
        - time_delay: horizon of max prediction (limited by y)
        - n_estimators: number of experiments per estimator
        - alpha: parameter for the quantiles alpha and 1-alpha
        - unpack: buffers from previous computation if None will compute everything from scratch 


    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def benchmark_methods(X, y, time_delay=20, n_estimators=10, alpha=0.05, unpack=None):
    if unpack is not None:
        baseline_ret_buffer = unpack['baseline']
        linear_ret_buffer = unpack['linear']
        lgbm_ret_buffer = unpack['lgbm']
        mlp_ret_buffer = unpack['mlp']
        gluonts_ret_buffer = unpack['gluonts']
    else:
        baseline_ret_buffer = []
        linear_ret_buffer = []
        lgbm_ret_buffer = []

        for i in range(n_estimators):
            print("baseline {}".format(i + 1))
            baseline_ret = baseline_regression(X, y, time_delay=time_delay, alpha=alpha, verbose=0,
                                               multiple_target=True, random_state=i, low_memory=True)
            baseline_ret_buffer.append(baseline_ret)

        for i in range(n_estimators):
            print("linear {}".format(i + 1))
            linear_ret = linear_regression(X, y, time_delay=time_delay, alpha=alpha, verbose=0, multiple_target=True,
                                           random_state=i, low_memory=True)
            linear_ret_buffer.append(linear_ret)

        for i in range(n_estimators):
            print("lgbm {}".format(i + 1))
            lgbm_ret = lightGBM_regression(X, y, time_delay=time_delay, alpha=alpha, kind="temp", verbose=1,
                                           multiple_target=True, random_state=i, low_memory=True)
            lgbm_ret_buffer.append(lgbm_ret)

    def mean_std(ret_buffer, metric_name, time_delay, n_estimators):
        ret = {}

        for j in range(time_delay):
            key = 't+' + str(j + 1)
            ret[key] = []

            for i in range(n_estimators):
                ret[key].append(ret_buffer[i][key][metric_name])

            mean = np.mean(ret[key])
            std = np.std(ret[key])
            ret[key] = {'mean': mean, 'std': std}

        return ret

    def bar_plot_compare(ret_buffers, names, metric_name, time_delay, n_estimators, ylabel, title):
        metrics = []

        for ret_buffer in ret_buffers:
            metrics.append(mean_std(ret_buffer=ret_buffer,
                                    metric_name=metric_name,
                                    time_delay=time_delay,
                                    n_estimators=n_estimators))

        index = list(metrics[0].keys())

        values = []
        # iterate through time delays
        for i in range(len(index)):
            key = index[i]
            buffer = []
            for metric in metrics:
                buffer.append(metric[key]['mean'])

            for metric in metrics:
                buffer.append(metric[key]['std'])

            values.append(buffer)

        std_names = []
        for name in names:
            std_names.append(name + "_std")
        columns = names + std_names

        df = pd.DataFrame(values,
                          columns=columns,
                          index=index)

        df[names].plot(kind='bar', yerr=df[std_names].values.T, alpha=0.5, error_kw=dict(ecolor='k', lw=1))
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel("Time target", fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.title(title, fontsize=25)
        plt.legend(fontsize=16, loc='upper right')
        plt.show()

    names = [
        'Baseline',
        'Linear',
        'LGBM',
        'RNN-LSTM',
        'RNN-GluonTS'
    ]
    ret_buffers = [
        baseline_ret_buffer,
        linear_ret_buffer,
        lgbm_ret_buffer,
        mlp_ret_buffer,
        gluonts_ret_buffer
    ]

    ret_buffers = [ret_buffers[4]]
    names = [names[4]]

    bar_plot_compare(ret_buffers=ret_buffers,
                     names=names,
                     metric_name='mean_absolute_error',
                     time_delay=time_delay,
                     n_estimators=n_estimators,
                     ylabel="Mean absolute error",
                     title="Mean absolute error vs time target")

    bar_plot_compare(ret_buffers=ret_buffers,
                     names=names,
                     metric_name='std_absolute_error',
                     time_delay=time_delay,
                     n_estimators=n_estimators,
                     ylabel="Standard deviation of absolute error",
                     title="Standard deviation of absolute error vs time target")

    bar_plot_compare(ret_buffers=ret_buffers,
                     names=names,
                     metric_name='quantile_loss',
                     time_delay=time_delay,
                     n_estimators=n_estimators,
                     ylabel="Quantile loss",
                     title="Quantile loss vs time target")

    bar_plot_compare(ret_buffers=ret_buffers,
                     names=names,
                     metric_name='coverage',
                     time_delay=time_delay,
                     n_estimators=n_estimators,
                     ylabel="Coverage",
                     title="Coverage vs time target")

    bar_plot_compare(ret_buffers=ret_buffers,
                     names=names,
                     metric_name='average_quantile_span',
                     time_delay=time_delay,
                     n_estimators=n_estimators,
                     ylabel="Average quantile span",
                     title="Average quantile span vs time target")

    bar_plot_compare(ret_buffers=ret_buffers,
                     names=names,
                     metric_name='time_train_predict',
                     time_delay=time_delay,
                     n_estimators=n_estimators,
                     ylabel="Computation time train and prediction",
                     title="Computation time for training and prediction vs time target")

    # plot feature importances
    LGBM_feature_importances(lgbm_ret_buffer, [1, 20, 40])

    return {'baseline': baseline_ret_buffer, 'linear': linear_ret_buffer, 'lgbm': lgbm_ret_buffer}


def save_training_data(filename, X, y):
    # np.save(filename+"_X", X)
    # np.save(filename+"_y", y)
    X.to_pickle(filename + "_X")
    y.to_pickle(filename + "_y")


def read_training_data(filename):
    # return np.load(filename+"_X.npy", allow_pickle=True), np.load(filename+"_y.npy", allow_pickle=True)
    return pd.read_pickle(filename + "_X"), pd.read_pickle(filename + "_y")


"""
    Simple forecasting baseline. Predict temperature at any point in the future as [(1-beta)*temp_t;(1+beta)*temp_t]

    Inputs:
        - X: explanatory variables
        - y: explained variables
        - alpha: prediction interval parameter, not used for computation ; just used for metric comparison
        - train_index: index of the samples to be used for training ; if None: index is selected randomly
        - test_index : index of the samples to be used for testing ; if None: index is selected randomly
        - kind: only "temp" is implemented
        - multiple_target: whether the prediction is single point in time or every point in time from t+1 to t+time_delay
        - verbose: 0 = silent, 1 = accuracy printed progressively
        - low_memory: whether to save X and y or just the accuracy measurements and what is necessary for plotting
    Returns
        results dictionary

    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def baseline_regression(X, y, time_delay, alpha, train_index=None, test_index=None, kind="temp",
                        multiple_target=False, verbose=0, random_state=0, low_memory=False):
    random.seed(random_state)

    # if no index is provided: generate partition randomly
    if train_index is None and test_index is None:
        X_train, y_train, X_test, y_test = temperature_build_db_train_test(X, y, train_ratio=0.8,
                                                                           random_state=random_state)
    else:  # index is provided
        X_train = X.loc[train_index, :]
        y_train = y.loc[train_index, :]
        X_test = X.loc[test_index, :]
        y_test = y.loc[test_index, :]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    ret = dict()
    ret['alpha'] = alpha
    if low_memory is False:
        # save
        ret['X_train'] = X_train
        ret['X_test'] = X_test
        ret['y_train'] = y_train
        ret['y_test'] = y_test
        ret['y_truth'] = X_test[:, list(X.columns).index('temp. t')]  # the original time series

    # wrapper for the baseline regressor
    class baseline:
        def __init__(self, ratio):
            self._ratio = ratio

        # only provide the last measurement
        def predict(self, X):
            return X * self._ratio

    # multiple point prediction
    if multiple_target:

        # loop through every target point from t+1 to t+time_delay (included)
        for i in range(time_delay):

            time_delay_i = i + 1
            key = 't+' + str(time_delay_i)
            ret[key] = {}

            y_test_i = y_test[:, i]

            if low_memory is False:
                ret[key]['y_test_original'] = y_test_i

            # get explanatory variables for this point in time
            _, var_to_use_y = regression_var_to_use(time_delay=time_delay_i,
                                                    column_names_x=list(X.columns),
                                                    column_names_y=list(y.columns),
                                                    multiple_target=True)

            var_to_use_x = list(X.columns).index('temp. t')  # only variable needed
            var_to_use_y = var_to_use_y[-1]  # wants to predict only the last one

            ret[key]["var_to_use_x"] = list(np.array(X.columns)[var_to_use_x])
            ret[key]["var_to_use_y"] = list(np.array(y.columns)[var_to_use_y])

            # time measurement
            start_time = time.clock()

            low_regressor = baseline(0.95)

            if low_memory is False:
                ret[key]['train_set_low_regressor'] = low_regressor

            high_regressor = baseline(1.05)

            if low_memory is False:
                ret[key]['train_set_high_regressor'] = high_regressor

            y_pred_low = low_regressor.predict(X_test[:, var_to_use_x])
            y_pred_high = high_regressor.predict(X_test[:, var_to_use_x])

            ret[key]['time_train_predict'] = time.clock() - start_time

            # save results
            if low_memory is False:
                ret[key]['y_pred_low'] = y_pred_low
                ret[key]['y_pred_high'] = y_pred_high

            # save metrics
            ret[key]['mean_absolute_error'] = mean_absolute_error(y_pred=(y_pred_low + y_pred_high) / 2,
                                                                  y_true=y_test_i)
            ret[key]['std_absolute_error'] = std_absolute_error(y_true=y_test_i, y_pred=(y_pred_low + y_pred_high) / 2)
            ret[key]['quantile_loss'] = full_quantile_loss(y_test_i, y_pred_low, y_pred_high, alpha=alpha)
            ret[key]['coverage'] = coverage(y_test_i, y_pred_low, y_pred_high)
            ret[key]['average_quantile_span'] = average_quantile_span(y_pred_low, y_pred_high)

            if verbose > 0:
                print("Time delay {}".format(time_delay_i))
                print("Time spent for training and prediction {:.2f}".format(ret[key]['time_train_predict']))
                print("Mean absolute error {}".format(ret[key]['mean_absolute_error']))
                print("Std absolute error {}".format(ret[key]['std_absolute_error']))
                print("Quantile loss {}".format(ret[key]['quantile_loss']))
                print("Coverage {}".format(ret[key]['coverage']))
                print("Average quantile span {}".format(ret[key]['average_quantile_span']))


    # single point prediction
    else:
        key = 't+' + str(time_delay)
        ret[key] = {}

        y_test_i = y_test[:, time_delay - 1]

        if low_memory is False:
            ret[key]['y_test_original'] = y_test_i

        # get explanatory variables for this point in time
        _, var_to_use_y = regression_var_to_use(time_delay=time_delay,
                                                column_names_x=list(X.columns),
                                                column_names_y=list(y.columns),
                                                multiple_target=False)

        var_to_use_x = list(X.columns).index('temp. t')

        ret[key]["var_to_use_x"] = list(np.array(X.columns)[var_to_use_x])
        ret[key]["var_to_use_y"] = list(np.array(y.columns)[var_to_use_y])

        # time measurement
        start_time = time.clock()

        lgb_l = baseline(0.95)

        if low_memory is False:
            ret[key]['train_set_low_regressor'] = lgb_l

        lgb_h = baseline(1.05)

        if low_memory is False:
            ret[key]['train_set_high_regressor'] = lgb_h

        y_pred_low = lgb_l.predict(X_test[:, var_to_use_x])
        y_pred_high = lgb_h.predict(X_test[:, var_to_use_x])

        ret[key]['time_train_predict'] = time.clock() - start_time

        # save results
        if low_memory is False:
            ret[key]['y_pred_low'] = y_pred_low
            ret[key]['y_pred_high'] = y_pred_high

        # save metrics
        ret[key]['mean_absolute_error'] = mean_absolute_error(y_pred=(y_pred_low + y_pred_high) / 2, y_true=y_test_i)
        ret[key]['std_absolute_error'] = std_absolute_error(y_true=y_test_i, y_pred=(y_pred_low + y_pred_high) / 2)
        ret[key]['quantile_loss'] = full_quantile_loss(y_test_i, y_pred_low, y_pred_high, alpha=alpha)
        ret[key]['coverage'] = coverage(y_test_i, y_pred_low, y_pred_high)
        ret[key]['average_quantile_span'] = average_quantile_span(y_pred_low, y_pred_high)

        if verbose > 0:
            print("Time delay {}".format(time_delay))
            print("Time spent for training and prediction {:.2f}".format(ret[key]['time_train_predict']))
            print("Mean absolute error {}".format(ret[key]['mean_absolute_error']))
            print("Std absolute error {}".format(ret[key]['std_absolute_error']))
            print("Quantile loss {}".format(ret[key]['quantile_loss']))
            print("Coverage {}".format(ret[key]['coverage']))
            print("Average quantile span {}".format(ret[key]['average_quantile_span']))

    return ret


"""
    Uses ordinary least squares from statsmodel. A gaussian gives the quantiles.

    Inputs:
        - X: explanatory variables
        - y: explained variables
        - alpha: prediction interval parameter, prediction interval will be from alpha to 1-alpha
        - train_index: index of the samples to be used for training ; if None: index is selected randomly
        - test_index : index of the samples to be used for testing ; if None: index is selected randomly
        - kind: only "temp" is implemented
        - multiple_target: whether the prediction is single point in time or every point in time from t+1 to t+time_delay
        - verbose: 0 = silent, 1 = accuracy printed progressively, 2 = training information (TO IMPLEMENT)

    Returns
        results dictionary

    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def linear_regression(X, y, time_delay, alpha=0.05, train_index=None, test_index=None, kind="temp",
                      multiple_target=False, verbose=0, random_state=0, low_memory=False):
    random.seed(random_state)

    X = X.copy()

    if kind == "temp":
        quant_tresh = 8
    else:
        quant_tresh = 6

    column_names_x = X.columns
    column_names_y = y.columns

    from sklearn.preprocessing import StandardScaler

    # scale explanatory variables
    x_scaler = StandardScaler()
    scaled_X = x_scaler.fit_transform(X.values[:, quant_tresh:])
    scaled_X = np.hstack([X.iloc[:, 0:quant_tresh].values, scaled_X])

    X = pd.DataFrame(scaled_X, columns=column_names_x)

    # scale explained variables
    y_scaled_buffer = []
    y_scalers = []
    for i in range(len(y.columns)):
        y_scaler = StandardScaler()
        scaled_y = y_scaler.fit_transform(y.values[:, i].reshape(-1, 1))
        y_scaled_buffer.append(pd.DataFrame(scaled_y, columns=[column_names_y[i]]))
        y_scalers.append(y_scaler)

    y = pd.concat(y_scaled_buffer, axis=1, sort=False)

    # use regex to graph digit

    # if no index is provided: generate partition randomly
    if train_index is None and test_index is None:
        X_train, y_train, X_test, y_test = temperature_build_db_train_test(X, y, train_ratio=0.8,
                                                                           random_state=random_state)
    else:  # index is provided
        X_train = X.loc[train_index, :]
        y_train = y.loc[train_index, :]
        X_test = X.loc[test_index, :]
        y_test = y.loc[test_index, :]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    ret = dict()
    ret['alpha'] = alpha

    if low_memory is False:
        # save
        ret['X_train_normalized'] = X_train
        ret['X_test_normalized'] = X_test
        ret['y_train_normalized'] = y_train
        ret['y_test_normalized'] = y_test

    inverse_scaled_X_test = x_scaler.inverse_transform(X_test[:, quant_tresh:])
    inverse_scaled_X_test = np.hstack([X_test[:, 0:quant_tresh], inverse_scaled_X_test])

    if low_memory is False:
        ret['y_truth'] = inverse_scaled_X_test[:, list(X.columns).index('temp. t')]  # the original time series

    """
        Wrapper around the linear regression models
        Inputs:
            - model: the linear model object
            - mode: either statsmodel_OLS to use statsmodel OLS model or anything else
                    if not stats model OLS the std of the residuals is estimated empirically
    """

    class LinearWrapper:

        def __init__(self, mode="statsmodel_OLS", random_state=0):
            self._mode = mode
            self._model = None
            self._residuals_scale = None
            self._random_state = random_state

        def fit(self, X, y):
            if self._mode == "statsmodel_OLS":
                from statsmodels.regression.linear_model import OLS
                self._model = OLS(y, X)
                self._model = self._model.fit()
                self._residuals_scale = np.sqrt(self._model.scale)
            elif self._mode == "sklearn_regularized":
                from sklearn.linear_model import Ridge
                self._model = Ridge(alpha=1, random_state=self._random_state)
                self._model = self._model.fit(X, y)
                # computation made on training data: might be over-optmistic
                self._residuals_scale = np.std(self._model.predict(X) - y)

            return self

        def get_coefficients(self):
            if self._mode == "statsmodel_OLS":
                return self._model.params
            elif self._mode == "sklearn_regularized":
                return self._model.coef_

        def predict(self, X, alpha):
            from scipy.stats import norm
            mean_pred = self._model.predict(X)
            return mean_pred + norm.ppf(alpha) * self._residuals_scale

    # multiple point prediction
    if multiple_target:

        # loop through every target point from t+1 to t+time_delay (included)
        for i in range(time_delay):

            time_delay_i = i + 1
            key = 't+' + str(time_delay_i)
            ret[key] = {}

            # standardize BACK
            y_test_i = y_scalers[i].inverse_transform(y_test[:, i])

            if low_memory is False:
                ret[key]['y_test_original'] = y_test_i

            # get explanatory variables for this point in time
            var_to_use_x, var_to_use_y = regression_var_to_use(time_delay=time_delay_i,
                                                               column_names_x=list(column_names_x),
                                                               column_names_y=list(column_names_y),
                                                               multiple_target=True)

            var_to_use_y = var_to_use_y[-1]  # wants to predict only the last one

            ret[key]["var_to_use_x"] = list(np.array(column_names_x)[var_to_use_x])
            ret[key]["var_to_use_y"] = list(np.array(column_names_y)[var_to_use_y])

            # time measurement
            start_time = time.clock()

            model = LinearWrapper(mode="statsmodel_OLS", random_state=random_state)
            model = model.fit(X_train[:, var_to_use_x].astype(float), y_train[:, var_to_use_y].astype(float))

            if low_memory is False:
                ret[key]['train_set_low_regressor'] = model
                ret[key]['train_set_high_regressor'] = model

            y_pred_low = model.predict(X_test[:, var_to_use_x].astype(float), alpha=alpha)
            y_pred_high = model.predict(X_test[:, var_to_use_x].astype(float), alpha=1 - alpha)

            y_pred_low = y_scalers[i].inverse_transform(y_pred_low)
            y_pred_high = y_scalers[i].inverse_transform(y_pred_high)

            ret[key]['time_train_predict'] = time.clock() - start_time

            # save results
            if low_memory is False:
                ret[key]['y_pred_low'] = y_pred_low.reshape((-1))
                ret[key]['y_pred_high'] = y_pred_high.reshape((-1))

            # save metrics
            ret[key]['mean_absolute_error'] = mean_absolute_error(y_pred=(y_pred_low + y_pred_high) / 2,
                                                                  y_true=y_test_i)
            ret[key]['std_absolute_error'] = std_absolute_error(y_true=y_test_i, y_pred=(y_pred_low + y_pred_high) / 2)
            ret[key]['quantile_loss'] = full_quantile_loss(y_test_i, y_pred_low, y_pred_high, alpha=alpha)
            ret[key]['coverage'] = coverage(y_test_i, y_pred_low, y_pred_high)
            ret[key]['average_quantile_span'] = average_quantile_span(y_pred_low, y_pred_high)

            if verbose > 0:
                print("Time delay {}".format(time_delay_i))
                print("Time spent for training and prediction {:.2f}".format(ret[key]['time_train_predict']))
                print("Mean absolute error {}".format(ret[key]['mean_absolute_error']))
                print("Std absolute error {}".format(ret[key]['std_absolute_error']))
                print("Quantile loss {}".format(ret[key]['quantile_loss']))
                print("Coverage {}".format(ret[key]['coverage']))
                print("Average quantile span {}".format(ret[key]['average_quantile_span']))


    # single point prediction
    else:
        time_delay_i = time_delay
        key = 't+' + str(time_delay_i)
        ret[key] = {}

        # standardize BACK
        y_test_i = y_scalers[i].inverse_transform(y_test[:, i])

        if low_memory is False:
            ret[key]['y_test_original'] = y_test_i

        # get explanatory variables for this point in time
        var_to_use_x, var_to_use_y = regression_var_to_use(time_delay=time_delay_i,
                                                           column_names_x=list(column_names_x),
                                                           column_names_y=list(column_names_y),
                                                           multiple_target=False)

        ret[key]["var_to_use_x"] = list(np.array(column_names_x)[var_to_use_x])
        ret[key]["var_to_use_y"] = list(np.array(column_names_y)[var_to_use_y])

        # time measurement
        start_time = time.clock()

        model = LinearWrapper(mode="statsmodel_OLS", random_state=random_state)
        model = model.fit(X_train[:, var_to_use_x].astype(float), y_train[:, var_to_use_y].astype(float))

        ret[key]['train_set_low_regressor'] = model

        ret[key]['train_set_high_regressor'] = model

        y_pred_low = model.predict(X_test[:, var_to_use_x].astype(float), alpha=alpha)
        y_pred_high = model.predict(X_test[:, var_to_use_x].astype(float), alpha=1 - alpha)

        y_pred_low = y_scalers[i].inverse_transform(y_pred_low)
        y_pred_high = y_scalers[i].inverse_transform(y_pred_high)

        ret[key]['time_train_predict'] = time.clock() - start_time

        # save results
        if low_memory is False:
            ret[key]['y_pred_low'] = y_pred_low.reshape((-1))
            ret[key]['y_pred_high'] = y_pred_high.reshape((-1))

        # save metrics
        ret[key]['mean_absolute_error'] = mean_absolute_error(y_pred=(y_pred_low + y_pred_high) / 2, y_true=y_test_i)
        ret[key]['std_absolute_error'] = std_absolute_error(y_true=y_test_i, y_pred=(y_pred_low + y_pred_high) / 2)
        ret[key]['quantile_loss'] = full_quantile_loss(y_test_i, y_pred_low, y_pred_high, alpha=alpha)
        ret[key]['coverage'] = coverage(y_test_i, y_pred_low, y_pred_high)
        ret[key]['average_quantile_span'] = average_quantile_span(y_pred_low, y_pred_high)

        if verbose > 0:
            print("Time delay {}".format(time_delay))
            print("Time spent for training and prediction {:.2f}".format(ret[key]['time_train_predict']))
            print("Mean absolute error {}".format(ret[key]['mean_absolute_error']))
            print("Std absolute error {}".format(ret[key]['std_absolute_error']))
            print("Quantile loss {}".format(ret[key]['quantile_loss']))
            print("Coverage {}".format(ret[key]['coverage']))
            print("Average quantile span {}".format(ret[key]['average_quantile_span']))

    return ret

    # OLD VERSION
    """from sklearn.preprocessing import StandardScaler
    random_seed = 0
    random.seed(random_seed)
    columns = X.columns

    if kind == "temp":
        quant_treshold = 8
    else:
        quant_treshold = 6

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X.values[:,quant_treshold:], y)
    scaled_X = np.hstack([X.iloc[:,0:quant_treshold].values, scaled_X])
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.values)
    X = pd.DataFrame(scaled_X, columns=columns)
    y = pd.DataFrame(y)

    X_train, y_train, X_test, y_test = temperature_build_db_train_test(X, y, train_ratio=0.9)
    model = OLS(y_train.values[:].astype(float), X_train.values[:, quant_treshold:].astype(float))
    model = model.fit()
    y_pred_low = ols_quantile(model, X_test.values[:, quant_treshold:].astype(float), 0.05)
    y_pred_high = ols_quantile(model, X_test.values[:, quant_treshold:].astype(float), 0.95)

    y_pred_low = y_scaler.inverse_transform(y_pred_low).reshape((-1))
    y_pred_high = y_scaler.inverse_transform(y_pred_high).reshape((-1))
    y_test = y_scaler.inverse_transform(y_test).reshape((-1))

    print("Mean absolute error " + str(mean_absolute_error(y_pred=(y_pred_low+y_pred_high)/2, y_true=y_test)))
    print("Quantile loss {}".format(full_quantile_loss(y_test, y_pred_low, y_pred_high, alpha=0.05)))
    print("Coverage {}".format(coverage(y_test, y_pred_low, y_pred_high)))
    print("Average quantile span {}".format(average_quantile_span(y_pred_low, y_pred_high)))

    regressor=None
    return regressor, columns, X_test.values, y_test, y_pred_low, y_pred_high"""


"""
    Statsmodel OLS is not appropriate to compute linear feature importance. Regularization is not available in our very
    specific case. Hence, we use, as a demonstration, an equivalent tool: f_regression from sthe klearn.

    Compute the scores and plot them.

    Inputs:
        - X, y: dataframes
        - time_delays: list of integers
        - n_estimators: number of cross-validation runs.

    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def linear_feature_importances(X, y, time_delays, n_estimators=10):
    from sklearn.feature_selection import f_regression
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split

    X = X.copy()

    """
        Encode categorical features in dummies with custom column names.
    """

    def one_hot_encode(X, cat_features):
        cat_dummies = pd.get_dummies(X[cat_features], prefix=cat_features, dummy_na=True)
        return cat_dummies

    for time_delay in time_delays:
        var_to_use = []

        for name in list(X.columns):
            if 'temp. t' in name:
                var_to_use += [name]

        for i in range(0, time_delay + 1):
            if i == 0:
                var_to_use += ['solar_irradiance t']
            else:
                var_to_use += ['solar_irradiance t+{}'.format(i)]

        for i in range(0, time_delay + 1):
            if i == 0:
                var_to_use += ['weather_temp t']
            else:
                var_to_use += ['weather_temp t+{}'.format(i)]

        buffer = np.zeros((n_estimators, len(var_to_use)))
        for n in range(n_estimators):
            X_n, _, y_n, _ = train_test_split(X[var_to_use].values, y.iloc[:, time_delay - 1].values, train_size=0.8,
                                              random_state=n)
            scores_i, _ = f_regression(X_n, y_n, center=True)
            buffer[n, :] = scores_i

        scores = np.mean(buffer, axis=0)
        stds = np.std(buffer, axis=0)

        stds /= np.max(scores)
        scores /= np.max(scores)
        print(stds)

        index = []
        index.append("t+{}".format(time_delay))

        columns = var_to_use
        df_scores = pd.DataFrame(np.array(scores).reshape((-1, 1)), index=columns)
        df_stds = pd.DataFrame(np.array(stds).reshape((-1, 1)))

        df_scores.plot(kind='bar', yerr=df_stds.values.T, alpha=0.9, legend=None)
        plt.xticks(fontsize=13, rotation=90)
        plt.yticks(fontsize=13)
        plt.ylabel("F-test scores", fontsize=15)
        plt.xlabel("Explanatory variables", fontsize=15)
        plt.title("Standardized F-test scores for time t+{}".format(time_delay), fontsize=25)
        plt.subplots_adjust(bottom=0.25)
        plt.show()


"""
    Plot the vif scores of the explanatory variables before and after PCA.

    Inputs:
        - X: explanatory variables

    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def plot_vif_scores(X):
    from statsmodels.regression.linear_model import OLS
    # source
    # https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
    def variance_inflation_factor(exog, exog_idx):
        """
        exog : ndarray, (nobs, k_vars)
            design matrix with all explanatory variables, as for example used in
            regression
        exog_idx : int
            index of the exogenous variable in the columns of exog
        """
        k_vars = exog.shape[1]
        x_i = exog[:, exog_idx]
        mask = np.arange(k_vars) != exog_idx
        x_noti = exog[:, mask]
        r_squared_i = OLS(x_i, x_noti).fit().rsquared
        vif = 1. / (1. - r_squared_i)
        print("{}/{}".format(exog_idx, exog.shape[1]))
        return vif

    from sklearn.preprocessing import StandardScaler

    column_names_x = X.columns
    quant_tresh = 8
    # scale explanatory variables
    x_scaler = StandardScaler()
    scaled_X = x_scaler.fit_transform(X.values[:, quant_tresh:])
    scaled_X = np.hstack([X.iloc[:, 0:quant_tresh].values, scaled_X])
    X = pd.DataFrame(scaled_X, columns=column_names_x)

    vifs = [variance_inflation_factor(X.values[:, quant_tresh:-42].astype(np.float), i) for i in
            range(len(X.columns[quant_tresh:-42]))]
    print(dict(zip(X.columns[quant_tresh:], vifs)))
    vifs = pd.DataFrame(data=vifs, index=X.columns[quant_tresh:-42], columns=[''])
    print(vifs)
    vifs.plot.bar(alpha=0.85, error_kw=dict(ecolor='k'))
    plt.axhline(y=10, color='r', label='Problematic threshold')
    plt.legend(fontsize=15)
    plt.xlabel("Explanatory variables", fontsize=15)
    plt.ylabel("VIF scores", fontsize=15)
    plt.title("VIF scores of the explanatory variables", fontsize=25)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.subplots_adjust(bottom=0.28)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=len(X.columns[quant_tresh:-42]))
    X_pca = pca.fit_transform(X.values[:, quant_tresh:-42].astype(np.float))
    vifs = [variance_inflation_factor(X_pca, i) for i in range(X_pca.shape[1])]
    vifs = pd.DataFrame(data=vifs, index=['PC {}'.format(i) for i in range(X_pca.shape[1])], columns=[''])
    print(vifs)
    vifs.plot.bar(alpha=0.85, error_kw=dict(ecolor='k'))
    plt.axhline(y=10, color='r', label='Problematic threshold')
    plt.legend(fontsize=15)
    plt.xlabel("Principal components", fontsize=15)
    plt.ylabel("VIF scores", fontsize=15)
    plt.title("VIF scores of the explanatory variables", fontsize=25)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.subplots_adjust(bottom=0.28)

    vars = pd.DataFrame(data=pca.explained_variance_, index=['PC {}'.format(i) for i in range(X_pca.shape[1])],
                        columns=[''])
    vars.plot.bar(alpha=0.85, error_kw=dict(ecolor='k'))
    plt.axhline(y=1, color='r', label='Rule of thumb threshold')
    plt.legend(fontsize=15)
    plt.xlabel("Principal components", fontsize=15)
    plt.ylabel("Explained variance", fontsize=15)
    plt.title("Scree plot - explained variance per principal component", fontsize=25)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.subplots_adjust(bottom=0.28)


"""
    Plot cross-validated coefficients of linear regression

    Inputs:
        - X, y : datafames
        - time_delays: list of integers
        - n_estimators: number of cross-validation runs

    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def plot_linear_coefficients(X, y, time_delays, n_estimators=10):
    for time_delay in time_delays:
        buffer = {}
        for n in range(n_estimators):
            ret = linear_regression(X, y, time_delay=time_delay, low_memory=True, random_state=n)
            coefficients = ret['t+' + str(time_delay)]['train_set_low_regressor'].get_coefficients()
            var_names = ret['t+' + str(time_delay)]['var_to_use_x']

            for i, key in enumerate(var_names):
                if key not in buffer:
                    buffer[key] = []

                buffer[key].append(coefficients[i])

        # buffer for dataframe conversion
        buffer_dict = {'mean': {}, 'std': {}}
        for key in buffer.keys():
            buffer_dict['mean'][key] = np.array(buffer[key]).mean()
            buffer_dict['std'][key] = np.array(buffer[key]).std()

        df = pd.DataFrame(buffer_dict)

        plt.figure()
        df['mean'].plot(kind='bar', yerr=df['std'].values.T, alpha=0.5, error_kw=dict(ecolor='k'))

        plt.xticks(fontsize=13, rotation=90)
        plt.yticks(fontsize=13)
        plt.xlabel("Explanatory variables", fontsize=15)
        plt.ylabel("Coefficient value", fontsize=15)
        plt.title("Linear regression coefficients for t+{}".format(time_delay), fontsize=22)
        plt.subplots_adjust(bottom=0.28)
        plt.show()


"""
    From columns names returns which variable to be use for a model with time delay=time_delay

    Inputs:
        - time_delay: time of target prediction
        - column_names: name of the variable (from dataframe)
    Returns:
        - list of integers which are indexes of the variable to be used in column_names

    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def regression_var_to_use(time_delay, column_names_x, column_names_y, multiple_target):
    var_to_use_x = []
    for name in column_names_x:
        if 'temp.' in name:
            var_to_use_x.append(column_names_x.index(name))
        elif 'solar_irradiance' in name:
            if '+' in name:  # solar_irradiance t+i
                buffer = name.replace('solar_irradiance t+', '')
                if int(buffer) <= time_delay:
                    var_to_use_x.append(column_names_x.index(name))
            else:  # solar_irradiance t
                var_to_use_x.append(column_names_x.index(name))
        elif 'weather_temp' in name:
            if '+' in name:  # weather_temp t+i
                buffer = name.replace('weather_temp t+', '')
                if int(buffer) <= time_delay:
                    var_to_use_x.append(column_names_x.index(name))
            else:  # weather_temp t
                var_to_use_x.append(column_names_x.index(name))
        """elif 'weather_humidity' in name:
            if '+' in name: # weather_humidity t+i
                buffer = name.replace('weather_humidity t+', '')
                if int(buffer) <= time_delay:
                    var_to_use_x.append(column_names_x.index(name))
            else: # weather_humidity t
                var_to_use_x.append(column_names_x.index(name))
        elif 'weather_pressure' in name:
            if '+' in name: # weather_pressure t+i
                buffer = name.replace('weather_pressure t+', '')
                if int(buffer) <= time_delay:
                    var_to_use_x.append(column_names_x.index(name))
            else: # weather_pressure t
                var_to_use_x.append(column_names_x.index(name))"""

    var_to_use_y = []
    for name in column_names_y:
        if 'temp. t' in name:
            buffer = name.replace('temp. t+', '')
            if (multiple_target and int(buffer) <= time_delay) or (not (multiple_target) and int(buffer) == time_delay):
                var_to_use_y.append(column_names_y.index(name))

    return var_to_use_x, var_to_use_y


"""
    Uses Extreme Gradient Tree Boosting from the library lightGBM.

    Inputs:
        - X: explanatory variables
        - y: explained variables
        - alpha: prediction interval parameter, prediction interval will be from alpha to 1-alpha
        - train_index: index of the samples to be used for training ; if None: index is selected randomly
        - test_index : index of the samples to be used for testing ; if None: index is selected randomly
        - kind: only "temp" is implemented
        - multiple_target: whether the prediction is single point in time or every point in time from t+1 to t+time_delay
        - verbose: 0 = silent, 1 = accuracy printed progressively, 2 = training information (TO IMPLEMENT)

    Returns
        results dictionary

    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def lightGBM_regression(X, y, time_delay, alpha=0.05, train_index=None, test_index=None, kind="temp",
                        multiple_target=False, verbose=0, random_state=0, low_memory=False):
    training_verbose = False
    if verbose >= 2:
        training_verbose = True

    random.seed(random_state)

    X = X.copy()

    def integer_encode(X, columns):
        X[columns] = X[columns][categorical_columns].astype('category')
        for col in categorical_columns:
            X[col] = X[col].cat.codes

        return X

    if kind == "temp":
        categorical_columns = X.columns[[2, 3, 4, 5, 6, 7]]
        quant_tresh = 8
        final_categorigal_columns = [0, 1, 2, 3, 4, 5]
    else:
        categorical_columns = X.columns[[2, 3, 4, 5]]
        quant_tresh = 6
        final_categorigal_columns = [0, 1, 2, 3]

    column_names_x = X.columns
    column_names_y = y.columns
    from sklearn.preprocessing import StandardScaler

    # integer encode categorical explanatory variables
    X = integer_encode(X, categorical_columns)
    X[categorical_columns] = X[categorical_columns].replace(-1, np.nan)

    # taking account of time_delay
    var_to_use_base = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # if no index is provided: generate paritition randomly
    if train_index is None and test_index is None:
        X_train, y_train, X_test, y_test = temperature_build_db_train_test(X, y, train_ratio=0.8,
                                                                           random_state=random_state)
    else:  # index is provided
        X_train = X.loc[train_index, :]
        y_train = y.loc[train_index, :]
        X_test = X.loc[test_index, :]
        y_test = y.loc[test_index, :]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    import lightgbm as lgb

    ret = dict()
    ret['alpha'] = alpha

    # save

    if low_memory is False:
        ret['X_train_normalized'] = X_train
        ret['X_test_normalized'] = X_test
        ret['y_train_normalized'] = y_train
        ret['y_test_normalized'] = y_test

    ret['y_truth'] = X_test[:, list(X.columns).index('temp. t')]  # the original time series

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # multiple point prediction
        if multiple_target:

            # loop through every target point from t+1 to t+time_delay (included)
            for i in range(time_delay):

                time_delay_i = i + 1
                key = 't+' + str(time_delay_i)
                ret[key] = {}

                y_test_i = y_test[:, i]

                if low_memory is False:
                    ret[key]['y_test_original'] = y_test_i

                # get explanatory variables for this point in time
                var_to_use_x_, var_to_use_y = regression_var_to_use(time_delay=time_delay_i,
                                                                    column_names_x=list(column_names_x),
                                                                    column_names_y=list(column_names_y),
                                                                    multiple_target=True)

                var_to_use_y = var_to_use_y[-1]  # wants to predict only the last one

                var_to_use_x = var_to_use_base + var_to_use_x_
                ret[key]["var_to_use_x"] = list(np.array(column_names_x)[var_to_use_x])
                ret[key]["var_to_use_y"] = list(np.array(column_names_y)[var_to_use_y])

                # time measurement
                start_time = time.clock()
                lgb_l = lgb.LGBMRegressor(objective="quantile", alpha=alpha, importance_type="gain", num_leaves=7,
                                          max_depth=-1, n_estimators=250,
                                          learning_rate=0.1, boosting_type="gbdt")
                lgb_l.fit(X_train[:, var_to_use_x],
                          y=y_train[:, var_to_use_y].ravel(), categorical_feature=final_categorigal_columns,
                          eval_set=[(X_train[:, var_to_use_x], y_train[:, var_to_use_y]),
                                    (X_test[:, var_to_use_x], y_test[:, var_to_use_y])],
                          eval_names=['Train', 'Validation'], verbose=training_verbose)

                if low_memory is False:
                    ret[key]['train_set_low_regressor'] = lgb_l

                lgb_h = lgb.LGBMRegressor(objective="quantile", alpha=1 - alpha, importance_type="gain", num_leaves=7,
                                          max_depth=-1, n_estimators=250,
                                          learning_rate=0.1, boosting_type="gbdt")
                lgb_h.fit(X_train[:, var_to_use_x],
                          y=y_train[:, var_to_use_y].ravel(), categorical_feature=final_categorigal_columns,
                          eval_set=[(X_train[:, var_to_use_x], y_train[:, var_to_use_y]),
                                    (X_test[:, var_to_use_x], y_test[:, var_to_use_y])],
                          eval_names=['Train', 'Validation'], verbose=training_verbose)

                if low_memory is False:
                    ret[key]['train_set_high_regressor'] = lgb_h

                ret[key]['training_loss'] = (np.array(lgb_l.evals_result_['Train']['quantile']) + \
                                             np.array(lgb_h.evals_result_['Train']['quantile'])) / 2

                ret[key]['validation_loss'] = (np.array(lgb_l.evals_result_['Validation']['quantile']) + \
                                               np.array(lgb_h.evals_result_['Validation']['quantile'])) / 2

                # feature importances are averaged over the two regressors
                ret[key]['feature_importances'] = dict(
                    zip(ret[key]["var_to_use_x"], (lgb_l.feature_importances_ + lgb_h.feature_importances_) / 2))

                y_pred_low = lgb_l.predict(X_test[:, var_to_use_x], num_iteration=lgb_l.best_iteration_)
                y_pred_high = lgb_h.predict(X_test[:, var_to_use_x], num_iteration=lgb_h.best_iteration_)

                ret[key]['time_train_predict'] = time.clock() - start_time

                # save results

                if low_memory is False:
                    ret[key]['y_pred_low'] = y_pred_low
                    ret[key]['y_pred_high'] = y_pred_high

                # save metrics
                ret[key]['mean_absolute_error'] = mean_absolute_error(y_pred=(y_pred_low + y_pred_high) / 2,
                                                                      y_true=y_test_i)
                ret[key]['std_absolute_error'] = std_absolute_error(y_true=y_test_i,
                                                                    y_pred=(y_pred_low + y_pred_high) / 2)
                ret[key]['quantile_loss'] = full_quantile_loss(y_test_i, y_pred_low, y_pred_high, alpha=alpha)
                ret[key]['coverage'] = coverage(y_test_i, y_pred_low, y_pred_high)
                ret[key]['average_quantile_span'] = average_quantile_span(y_pred_low, y_pred_high)

                if verbose > 0:
                    print("Time delay {}".format(time_delay_i))
                    print("Time spent for training and prediction {:.2f}".format(ret[key]['time_train_predict']))
                    print("Mean absolute error {}".format(ret[key]['mean_absolute_error']))
                    print("Std absolute error {}".format(ret[key]['std_absolute_error']))
                    print("Quantile loss {}".format(ret[key]['quantile_loss']))
                    print("Coverage {}".format(ret[key]['coverage']))
                    print("Average quantile span {}".format(ret[key]['average_quantile_span']))


        # single point prediction
        else:
            key = 't+' + str(time_delay)
            ret[key] = {}
            # standardize BACK
            y_test_i = y_test[:, time_delay - 1]

            if low_memory is False:
                ret[key]['y_test_original'] = y_test_i

            # get explanatory variables for this point in time
            var_to_use_x_, var_to_use_y = regression_var_to_use(time_delay=time_delay,
                                                                column_names_x=list(column_names_x),
                                                                column_names_y=list(column_names_y),
                                                                multiple_target=False)

            var_to_use_y = var_to_use_y[0]
            var_to_use_x = var_to_use_base + var_to_use_x_
            ret[key]["var_to_use_x"] = list(np.array(column_names_x)[var_to_use_x])
            ret[key]["var_to_use_y"] = list(np.array(column_names_y)[var_to_use_y])
            # time measurement
            start_time = time.clock()
            lgb_l = lgb.LGBMRegressor(objective="quantile", alpha=alpha, importance_type="gain", num_leaves=7,
                                      max_depth=-1, n_estimators=250,
                                      learning_rate=0.25, boosting_type="goss")

            lgb_l.fit(X_train[:, var_to_use_x],
                      y=y_train[:, var_to_use_y].ravel(), categorical_feature=final_categorigal_columns,
                      eval_set=[(X_train[:, var_to_use_x], y_train[:, var_to_use_y]),
                                (X_test[:, var_to_use_x], y_test[:, var_to_use_y])],
                      eval_names=['Train', 'Validation'], verbose=training_verbose)

            if low_memory is False:
                ret[key]['train_set_low_regressor'] = lgb_l

            lgb_h = lgb.LGBMRegressor(objective="quantile", alpha=1 - alpha, importance_type="gain", num_leaves=7,
                                      max_depth=-1, n_estimators=250,
                                      learning_rate=0.25, boosting_type="goss")
            lgb_h.fit(X_train[:, var_to_use_x],
                      y=y_train[:, var_to_use_y].ravel(), categorical_feature=final_categorigal_columns,
                      eval_set=[(X_train[:, var_to_use_x], y_train[:, var_to_use_y]),
                                (X_test[:, var_to_use_x], y_test[:, var_to_use_y])],
                      eval_names=['Train', 'Validation'], verbose=training_verbose)

            if low_memory is False:
                ret[key]['train_set_high_regressor'] = lgb_h

            ret[key]['training_loss'] = (np.array(lgb_l.evals_result_['Train']['quantile']) + \
                                         np.array(lgb_h.evals_result_['Train']['quantile'])) / 2

            ret[key]['validation_loss'] = (np.array(lgb_l.evals_result_['Validation']['quantile']) + \
                                           np.array(lgb_h.evals_result_['Validation']['quantile'])) / 2

            # feature importances are averaged over the two regressors
            ret[key]['feature_importances'] = dict(
                zip(ret[key]["var_to_use_x"], (lgb_l.feature_importances_ + lgb_h.feature_importances_) / 2))

            y_pred_low = lgb_l.predict(X_test[:, var_to_use_x], num_iteration=lgb_l.best_iteration_)
            y_pred_high = lgb_h.predict(X_test[:, var_to_use_x], num_iteration=lgb_h.best_iteration_)

            ret[key]['time_train_predict'] = time.clock() - start_time

            # save results
            if low_memory is False:
                ret[key]['y_pred_low'] = y_pred_low
                ret[key]['y_pred_high'] = y_pred_high

            # save metrics
            ret[key]['mean_absolute_error'] = mean_absolute_error(y_pred=(y_pred_low + y_pred_high) / 2,
                                                                  y_true=y_test_i)
            ret[key]['std_absolute_error'] = std_absolute_error(y_true=y_test_i, y_pred=(y_pred_low + y_pred_high) / 2)
            ret[key]['quantile_loss'] = full_quantile_loss(y_test_i, y_pred_low, y_pred_high, alpha=alpha)
            ret[key]['coverage'] = coverage(y_test_i, y_pred_low, y_pred_high)
            ret[key]['average_quantile_span'] = average_quantile_span(y_pred_low, y_pred_high)

            if verbose > 0:
                print("Time delay {}".format(time_delay))
                print("Time spent for training and prediction {:.2f}".format(ret[key]['time_train_predict']))
                print("Mean absolute error {}".format(ret[key]['mean_absolute_error']))
                print("Std absolute error {}".format(ret[key]['std_absolute_error']))
                print("Quantile loss {}".format(ret[key]['quantile_loss']))
                print("Coverage {}".format(ret[key]['coverage']))
                print("Average quantile span {}".format(ret[key]['average_quantile_span']))

    return ret


"""
    Given several runs of lgbm. Plot the variable importances with std at different time delays.

    Inputs:
        - lgbm_ret_buffer: list of lgbm ret
        - time delays: list of integers


    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def LGBM_feature_importances(lgbm_ret_buffer, time_delays):
    for time_delay in time_delays:
        key = 't+' + str(time_delay)
        scores_buffer = {}
        for lgbm_ret in lgbm_ret_buffer:
            scores = lgbm_ret[key]['feature_importances']
            for var in scores:
                if var not in scores_buffer:
                    scores_buffer[var] = []
                score = scores[var]
                scores_buffer[var].append(score)

        # buffer for dataframe conversion
        buffer_dict = {'mean': {}, 'std': {}}
        max_buffer = -1.0
        for var in scores_buffer:
            buffer_dict['mean'][var] = np.array(scores_buffer[var]).mean()
            max_buffer = max(max_buffer, buffer_dict['mean'][var])

            buffer_dict['std'][var] = np.array(scores_buffer[var]).std()

        df = pd.DataFrame(buffer_dict)
        df /= max_buffer

        plt.figure()
        df['mean'].plot(kind='bar', yerr=df['std'].values.T, alpha=0.5, error_kw=dict(ecolor='k'))

        plt.xticks(fontsize=11, rotation=90)
        plt.yticks(fontsize=11)
        plt.ylabel("Standardized LGBM feature importance score", fontsize=20)
        plt.xlabel("Explanatory variables", fontsize=20)
        plt.title("Standardized LGBM feature importance scores for time t+{}".format(time_delay), fontsize=25)
        plt.subplots_adjust(bottom=0.28)
        plt.show()


"""
    Compute more permutation importance for given time delays
    Source : https://scikit-learn.org/stable/auto_examples/inspection/
    plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
    
    Inputs:
        - X,y : dataframes
        - time_delays: list of integers ; the time delays to compute feature importance on
        - train_ratio: ratio of the training set
"""
def features_permutation_importance(X, y, time_delays, train_ratio, random_state=0):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.inspection import permutation_importance

    ret = {}
    # drop two first columns
    X = X.iloc[:, 2:]

    # qualitative features are the non-categorical features
    categorical_features = [
        'type',
        'location',
        'kind',
        'sanitizer_process',
        'equipment_protections',
        'equipment_heatings'
    ]

    categorical_indexes = []
    quantitative_indexes = []

    for i, name in enumerate(list(X.columns)):
        if name in categorical_features:
            categorical_indexes.append(i)
        else:
            quantitative_indexes.append(i)

    """
        Encode categorical features in dummies with custom column names.
    """

    def one_hot_encode(X, cat_features):
        cat_dummies = pd.get_dummies(X[cat_features], prefix=cat_features, dummy_na=True)
        return cat_dummies

    dummies = one_hot_encode(X, categorical_features)
    # merge back
    X = pd.DataFrame(data=np.hstack([dummies.values, X.iloc[:, quantitative_indexes]]),
                     columns=list(dummies.columns) + list(X.columns[quantitative_indexes]))

    max_features = len(list(X.columns))
    for time_delay in time_delays:
        key = 't+{}'.format(time_delay)
        ret[key] = {}
        ret[key]['time_fit'] = []
        ret[key]['time_permute'] = []
        ret[key]['EXT_feature_importances'] = []
        ret[key]['permutation_feature_importance_mean'] = []
        ret[key]['permutation_feature_importance_std'] = []
        ret[key]['train_accuracy'] = []
        ret[key]['test_accuracy'] = []
        ret[key]['MAE'] = []


        y_target = y.iloc[:, time_delay - 1]
        X_train, X_test, y_train, y_test = train_test_split(X.values, y_target.values, shuffle=True,
                                                            train_size=train_ratio, random_state=random_state)


        start_time = time.clock()
        model = ExtraTreesRegressor(n_estimators=1000, criterion="mae", max_depth=5, max_features=max_features,
                                    random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        time_measurement = time.clock() - start_time

        ret[key]['EXT_feature_importances'].append(model.feature_importances_)
        ret[key]['train_accuracy'].append(model.score(X_train, y_train))
        ret[key]['test_accuracy'].append(model.score(X_test, y_test))
        ret[key]['MAE'].append(mean_absolute_error(y_test, model.predict(X_test)))


        print("EXT train accuracy: {:.3f}".format(ret[key]['train_accuracy'][0]))
        print("EXT test accuracy: {:.3f}".format(ret[key]['test_accuracy'][0]))
        print("EXT test MAE: {:.3f}".format(ret[key]['MAE'][0]))
        print("Time training: {:.3f}".format(time_measurement))


        start_time = time.clock()
        result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                        random_state=random_state, n_jobs=-1)
        time_measurement = time.clock() - start_time
        ret[key]['time_permute'].append(time_measurement)
        print("Time training: {:.3f}".format(time_measurement))

        ret[key]['permutation_feature_importance_mean'] = result.importances_mean
        ret[key]['permutation_feature_importance_std'] = result.importances_std

        df = pd.DataFrame(data=np.transpose(np.vstack([result.importances_mean, result.importances_std])),
                          columns=['mean', 'std'],
                          index= X.columns)
        df['mean'].plot(kind='bar', yerr=df['std'].values.T, alpha=0.5, error_kw=dict(ecolor='k', lw=1))
        plt.title("Permutation importance from Extra Trees Regressor", fontsize=25)
        plt.xlabel("Explanatory variables", fontsize=18)
        plt.ylabel("Permutation importance score", fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.show()

        return ret


"""
    Compute more permutation importance for given time delays
    Source : https://scikit-learn.org/stable/auto_examples/inspection/
    plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py

    Inputs:
        - X,y : dataframes
        - time_delays: list of integers ; the time delays to compute feature importance on
        - train_ratio: ratio of the training set
"""


def extra__importance(X, y, time_delays, train_ratio, random_state=0):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.inspection import permutation_importance

    ret = {}
    # drop two first columns
    X = X.iloc[:, 2:]

    # qualitative features are the non-categorical features
    categorical_features = [
        'type',
        'location',
        'kind',
        'sanitizer_process',
        'equipment_protections',
        'equipment_heatings'
    ]

    categorical_indexes = []
    quantitative_indexes = []

    for i, name in enumerate(list(X.columns)):
        if name in categorical_features:
            categorical_indexes.append(i)
        else:
            quantitative_indexes.append(i)

    """
        Encode categorical features in dummies with custom column names.
    """

    def one_hot_encode(X, cat_features):
        cat_dummies = pd.get_dummies(X[cat_features], prefix=cat_features, dummy_na=True)
        return cat_dummies

    dummies = one_hot_encode(X, categorical_features)
    # merge back
    X = pd.DataFrame(data=np.hstack([dummies.values, X.iloc[:, quantitative_indexes]]),
                     columns=list(dummies.columns) + list(X.columns[quantitative_indexes]))

    max_features = len(list(X.columns))
    for time_delay in time_delays:
        key = 't+{}'.format(time_delay)
        ret[key] = {}
        ret[key]['time_fit'] = []
        ret[key]['time_permute'] = []
        ret[key]['EXT_feature_importances'] = []
        ret[key]['permutation_feature_importance_mean'] = []
        ret[key]['permutation_feature_importance_std'] = []
        ret[key]['train_accuracy'] = []
        ret[key]['test_accuracy'] = []
        ret[key]['MAE'] = []

        y_target = y.iloc[:, time_delay - 1]
        X_train, X_test, y_train, y_test = train_test_split(X.values, y_target.values, shuffle=True,
                                                            train_size=train_ratio, random_state=random_state)

        start_time = time.clock()
        model = ExtraTreesRegressor(n_estimators=1000, criterion="mae", max_depth=5, max_features=max_features,
                                    random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        time_measurement = time.clock() - start_time

        ret[key]['EXT_feature_importances'].append(model.feature_importances_)
        ret[key]['train_accuracy'].append(model.score(X_train, y_train))
        ret[key]['test_accuracy'].append(model.score(X_test, y_test))
        ret[key]['MAE'].append(mean_absolute_error(y_test, model.predict(X_test)))

        print("EXT train accuracy: {:.3f}".format(ret[key]['train_accuracy'][0]))
        print("EXT test accuracy: {:.3f}".format(ret[key]['test_accuracy'][0]))
        print("EXT test MAE: {:.3f}".format(ret[key]['MAE'][0]))
        print("Time training: {:.3f}".format(time_measurement))


        df = pd.DataFrame(data=np.transpose(np.vstack([result.importances_mean, result.importances_std])),
                          columns=['mean', 'std'],
                          index=X.columns)
        df['mean'].plot(kind='bar', yerr=df['std'].values.T, alpha=0.5, error_kw=dict(ecolor='k', lw=1))
        plt.title("Permutation importance from Extra Trees Regressor", fontsize=25)
        plt.xlabel("Explanatory variables", fontsize=18)
        plt.ylabel("Permutation importance score", fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.show()

        return ret

"""
    Plot the training and validation loss from lgbm_ret dicts. The curves can be cross-validated.

    Inputs:
        - lgbm_ret_buffer: lists of lgbm dictionaries returns
        - time_delays: lists of integers ; the time delays of the regressors whose losses will be shown
        - period: distance between ticks


    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def lgbm_plot_loss(lgbm_ret_buffer, time_delays, period=25):
    for time_delay in time_delays:
        key = "t+{}".format(time_delay)
        # keys are epochs
        validation_losses = {}
        training_losses = {}
        for lgbm_ret in lgbm_ret_buffer:
            training_loss = lgbm_ret[key]['training_loss']
            validation_loss = lgbm_ret[key]['validation_loss']
            for i in range(len(training_loss)):
                if i + 1 not in validation_losses:
                    validation_losses[i + 1] = []

                validation_losses[i + 1].append(validation_loss[i])

                if i + 1 not in training_losses:
                    training_losses[i + 1] = []

                training_losses[i + 1].append(training_loss[i])

        validation_losses_std = {}
        training_losses_std = {}
        for epoch in validation_losses:
            validation_losses_std[epoch] = np.array(validation_losses[epoch]).std()
            validation_losses[epoch] = np.array(validation_losses[epoch]).mean()

            training_losses_std[epoch] = np.array(training_losses[epoch]).std()
            training_losses[epoch] = np.array(training_losses[epoch]).mean()

        plt.figure()
        x_ticks = list(training_losses.keys())[::period]
        if len(lgbm_ret_buffer) == 1:
            plt.plot(x_ticks,
                     list(training_losses.values())[::period],
                     linestyle='-', color='blue')

            plt.plot(x_ticks,
                     list(validation_losses.values())[::period],
                     linestyle='--', color='red')
        else:
            plt.errorbar(x_ticks,
                         list(training_losses.values())[::period],
                         yerr=list(training_losses_std.values())[::period], linestyle='-', color='blue', capsize=5)

            plt.errorbar(x_ticks,
                         list(validation_losses.values())[::period],
                         yerr=list(validation_losses_std.values())[::period], linestyle='--', color='red', capsize=5)

        plt.fill_between(x_ticks,
                         [training_losses[epoch] - training_losses_std[epoch] for epoch in
                          list(training_losses.keys())[::period]],
                         [training_losses[epoch] + training_losses_std[epoch] for epoch in
                          list(training_losses.keys())[::period]],
                         color='blue', alpha=.25)
        plt.fill_between(x_ticks,
                         [validation_losses[epoch] - validation_losses_std[epoch] for epoch in
                          list(validation_losses.keys())[::period]],
                         [validation_losses[epoch] + validation_losses_std[epoch] for epoch in
                          list(validation_losses.keys())[::period]],
                         color='red', alpha=.25)

        plt.legend(['Training Loss', 'Test Loss'], fontsize=15)
        plt.xticks([it - 1 for it in list(validation_losses.keys())[::period]], fontsize=15)

        plt.xlabel('Iterations', fontsize=18)
        plt.ylabel('Quantile loss', fontsize=18)
        if len(lgbm_ret_buffer) == 1:
            plt.title("Training/validation loss vs iterations for predicting time t+{}".format(time_delay), fontsize=22)
        else:
            plt.title(
                "Training/validation loss vs iterations (cross-validated) for predicting time t+{}".format(time_delay),
                fontsize=22)

        plt.show()


def predict_damage(X, y, mode="temperature"):
    swp_big = pd.read_csv("swp.csv", low_memory=False)

    customer_pref_ids = swp_big['swimming_pool_id'].loc[
        (~swp_big['custom_mr_orp_min'].isna()) & \
        (~swp_big['custom_mr_ph_min'].isna()) & \
        (~swp_big['custom_mr_temperature_min'].isna()) & \
        (~swp_big['custom_mr_orp_max'].isna()) & \
        (~swp_big['custom_mr_ph_max'].isna()) & \
        (~swp_big['custom_mr_temperature_max'].isna())
        ]

    lgbm_regressor, lgbm_names, lgbm_x_test, lgbm_y_test, lgbm_y_pred_low, lgbm_y_pred_high = lightGBM_regression(X, y,
                                                                                                                  kind="temp")

    # records prediction of problem vs ground truth
    records = pd.DataFrame(columns=['swimming_pool_id', 'time', 'predicted', 'ground_truth'])

    common_ids = set(customer_pref_ids).intersection(lgbm_x_test[:, 0])
    j = 0
    for id in common_ids:
        pool_id = swp_big[swp_big['swimming_pool_id'] == id]
        max_temp = pool_id['custom_mr_temperature_max'].values
        min_temp = pool_id['custom_mr_temperature_min'].values

        measurement_id = lgbm_x_test[:, 0] == id
        real_measurement = lgbm_y_test[measurement_id]
        time_measurement = lgbm_x_test[measurement_id, 1]
        low_prediction = lgbm_y_pred_low[measurement_id]
        high_prediction = lgbm_y_pred_high[measurement_id]
        mid = (low_prediction + high_prediction) / 2

        for i in range(len(real_measurement)):
            truth = int(real_measurement[i] > max_temp or real_measurement[i] < min_temp)
            predicted = int(mid[i] > max_temp or mid[i] < min_temp)
            records.loc[j] = [id, time_measurement[i], predicted, truth]
            j += 1

    tn, fp, fn, tp = confusion_matrix(records['predicted'].values.astype(int),
                                      records['ground_truth'].values.astype(int)).ravel()
    print("tn: {} ; fp: {} ; fn: {} ; tp: {}".format(tn, fp, fn, tp))
    return records


"""
    Uses MLP from tensorflow.

    Inputs:
        - X: explanatory variables
        - y: explained variables
        - alpha: prediction interval parameter, prediction interval will be from alpha to 1-alpha
        - train_index: index of the samples to be used for training ; if None: index is selected randomly
        - test_index : index of the samples to be used for testing ; if None: index is selected randomly
        - kind: only "temp" is implemented
        - multiple_target: whether the prediction is single point in time or every point in time from t+1 to t+time_delay
        - verbose: 0 = silent, 1 = accuracy printed progressively, 2 = training information, 3 = one line per epoch

    Returns
        results dictionary

    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def MLP_regression(X, y, time_delay, alpha=0.05, train_index=None, test_index=None, kind="temp", multiple_target=False,
                   verbose=0, random_state=0, low_memory=False):
    import tensorflow as tf

    import tensorflow.keras.backend as kb

    training_verbose = 0
    if verbose == 2:
        training_verbose = 1
    elif verbose == 3:
        training_verbose = 2

    # check https://www.tensorflow.org/api_docs/python/tf/keras/backend
    def quantile_loss_wrapper(quantile):

        def quantile_loss(y_true, y_pred):
            e = (y_true - y_pred)
            return kb.mean(kb.maximum(quantile * e, (quantile - 1) * e), axis=-1)

        return quantile_loss

    random.seed(random_state)

    X = X.copy()

    """
        Encode categorical features in dummies with custom column names.
    """

    def one_hot_encode(X, cat_features):
        cat_dummies = pd.get_dummies(X[cat_features], prefix=cat_features, dummy_na=True)
        return cat_dummies

    categorical_features = ['type', 'location', 'kind', 'sanitizer_process', 'equipment_protections',
                            'equipment_heatings']
    quant_tresh = 8
    dummies = one_hot_encode(X, categorical_features)
    # merge back
    X = pd.DataFrame(data=np.hstack([X.iloc[:, 0:2].values,
                                     dummies.values,
                                     X.iloc[:, quant_tresh:]]),
                     columns=list(X.columns[0:2]) + list(dummies.columns) + list(X.columns[quant_tresh:]))

    column_names_x = X.columns
    column_names_y = y.columns
    from sklearn.preprocessing import StandardScaler

    # scale explanatory variables
    # NB. Might be useless in this case (trees)
    x_scaler = StandardScaler()
    scaled_X = x_scaler.fit_transform(X.values[:, 2:], y)
    scaled_X = np.hstack([X.iloc[:, 0:2].values, scaled_X])

    X = pd.DataFrame(scaled_X, columns=column_names_x)

    # scale explained variables
    y_scaled_buffer = []
    y_scalers = []
    for i in range(len(y.columns)):
        y_scaler = StandardScaler()
        scaled_y = y_scaler.fit_transform(y.values[:, i].reshape(-1, 1))
        y_scaled_buffer.append(pd.DataFrame(scaled_y, columns=[column_names_y[i]]))
        y_scalers.append(y_scaler)

    y = pd.concat(y_scaled_buffer, axis=1, sort=False)

    # first variables to use
    var_to_use_base = [2 + i for i, _ in enumerate(dummies.columns)]
    var_to_use_base += [list(column_names_x).index(item) for item in ['volume_capacity', 'day_year', 'seconds_day']]

    # if no index is provided: generate paritition randomly
    if train_index is None and test_index is None:
        X_train, y_train, X_test, y_test = temperature_build_db_train_test(X, y, train_ratio=0.8,
                                                                           random_state=random_state)
    else:  # index is provided
        X_train = X.loc[train_index, :]
        y_train = y.loc[train_index, :]
        X_test = X.loc[test_index, :]
        y_test = y.loc[test_index, :]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    ret = dict()
    ret['alpha'] = alpha

    # save
    if low_memory is False:
        ret['X_train_normalized'] = X_train
        ret['X_test_normalized'] = X_test
        ret['y_train_normalized'] = y_train
        ret['y_test_normalized'] = y_test

    if low_memory is False:
        inverse_scaled_X_test = x_scaler.inverse_transform(X_test[:, 2:])
        inverse_scaled_X_test = np.hstack([X_test[:, 0:2], inverse_scaled_X_test])
        ret['y_truth'] = inverse_scaled_X_test[:, list(X.columns).index('temp. t')]  # the original time series
    # multiple point prediction
    if multiple_target:
        # get explanatory variables for this point in time
        var_to_use_x_, var_to_use_y = regression_var_to_use(time_delay=time_delay,
                                                            column_names_x=list(column_names_x),
                                                            column_names_y=list(column_names_y),
                                                            multiple_target=True)
        var_to_use_x = var_to_use_base + var_to_use_x_

        if low_memory is False:
            ret["var_to_use_x"] = list(np.array(column_names_x)[var_to_use_x])
            ret["var_to_use_y"] = list(np.array(column_names_y)[var_to_use_y])

        # time measurement
        start_time = time.clock()

        """outputs = []
        losses = []
        loss_weights = []
        main_input = tf.keras.Input(shape=(len(var_to_use_x),), name='main_input')

        for i in range(time_delay):
            #l = tf.keras.layers.Dense(32, activation="relu", name="Wt_{}".format(i+1))(main_input)
            if i == 0:
                l = tf.keras.layers.Dense(16, activation="linear")(main_input)
            else:
                l = tf.keras.layers.Dense(16, activation="linear")(l)


            name_low = "quantile_out_005_{}".format(i + 1)
            name_high = "quantile_out_095_{}".format(i + 1)
            out_l = tf.keras.layers.Dense(1, activation='linear', name=name_low)(l)
            out_h = tf.keras.layers.Dense(1, activation='linear', name=name_high)(l)
            outputs.append(out_l)
            outputs.append(out_h)
            losses.append(quantile_loss_wrapper(0.05))
            losses.append(quantile_loss_wrapper(0.95))
            loss_weights.append(1 / (2 * time_delay))
            loss_weights.append(1 / (2 * time_delay))

        model = tf.keras.Model(inputs=main_input, outputs=outputs)

        model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)

        trainable_count = np.sum([kb.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([kb.count_params(w) for w in model.non_trainable_weights])

        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))


        # save model image
        if False:
            file_name = 'model.png'
            print("Model image graph saved at {}".format(file_name))
            tf.keras.utils.plot_model(
                model, to_file=file_name, show_shapes=False, show_layer_names=True,
                rankdir='TB', expand_nested=False, dpi=96
            )

        from sklearn.utils import shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
        y_train_final = []
        y_validation_final = []
        for i, var_y in enumerate(var_to_use_y):
            y_train_final.append(y_train[:, var_y].astype(np.float))
            y_train_final.append(y_train[:, var_y].astype(np.float))

            y_validation_final.append(y_test[:, var_y].astype(np.float))
            y_validation_final.append(y_test[:, var_y].astype(np.float))

        history = model.fit(X_train[:, var_to_use_x].astype(np.float),
                            y_train_final,
                            epochs=150,
                            batch_size=2048,
                            validation_data=(X_test[:, var_to_use_x].astype(np.float), y_validation_final),
                            verbose=training_verbose)


        y_preds = model.predict(X_test[:, var_to_use_x].astype(np.float))"""

        BATCH_SIZE = 256
        BUFFER_SIZE = 20000

        y_train_ = []
        y_test_ = []
        for i, var_y in enumerate(var_to_use_y):
            y_train_.append(y_train[:, var_y].astype(np.float))
            y_train_.append(y_train[:, var_y].astype(np.float))

            y_test_.append(y_test[:, var_y].astype(np.float))
            y_test_.append(y_test[:, var_y].astype(np.float))

        """n_repeats = int(np.floor(32*32/len(var_to_use_x)))
        X_train_ = np.zeros((X_train.shape[0], 1024))
        X_test_ = np.zeros((X_test.shape[0], 1024))

        for i in range(n_repeats):
            X_train_[:, i*len(var_to_use_x):(i+1)*len(var_to_use_x)] = X_train[:, var_to_use_x]
            X_test_[:, i*len(var_to_use_x):(i+1)*len(var_to_use_x)] = X_test[:, var_to_use_x]

        X_train_ = X_train_.reshape((-1,32,32, 1)).astype(np.float)
        X_test_ = X_test_.reshape((-1,32,32, 1)).astype(np.float)"""

        X_train_ = X_train[:, var_to_use_x].reshape((-1, len(var_to_use_x), 1)).astype(np.float)
        X_test_ = X_test[:, var_to_use_x].reshape((-1, len(var_to_use_x), 1)).astype(np.float)

        # The next step's to ensure data is fed in expected format; for LSTM,
        # that'd be a 3D tensor with dimensions (batch_size, timesteps, features)
        # - or equivalently, (num_samples, timesteps, channels)
        # https://stackoverflow.com/questions/
        # 58636087/tensorflow-valueerror-failed-to-convert-a-numpy-array-to-a-tensor-unsupporte
        train_univariate = tf.data.Dataset.from_tensor_slices((X_train_, tuple(y_train_)))
        train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        val_univariate = tf.data.Dataset.from_tensor_slices((X_test_, tuple(y_test_)))
        val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

        """from tensorflow.keras.applications import MobileNetV2
        resnet50_imagenet_model = MobileNetV2(include_top=False, weights=None, input_shape=(32, 32, 1))
        l = tf.keras.layers.Flatten()(resnet50_imagenet_model.output)
        l = tf.keras.layers.Dense(64, activation="relu")(l)"""

        main_input = tf.keras.Input(shape=X_train_.shape[-2:], name='main_input')
        l = tf.keras.layers.LSTM(64, return_sequences=True, input_shape=X_train_.shape[-2:])(main_input)
        l = tf.keras.layers.LSTM(32, return_sequences=True, activation="relu")(l)
        l = tf.keras.layers.Flatten()(l)
        outputs = []
        losses = []
        loss_weights = []

        for i in range(time_delay):
            name_low = "quantile_out_005_{}".format(i + 1)
            name_high = "quantile_out_095_{}".format(i + 1)
            out_l = tf.keras.layers.Dense(1, activation='linear', name=name_low)(l)
            out_h = tf.keras.layers.Dense(1, activation='linear', name=name_high)(l)
            outputs.append(out_l)
            outputs.append(out_h)
            losses.append(quantile_loss_wrapper(0.05))
            losses.append(quantile_loss_wrapper(0.95))
            loss_weights.append(1 / (2 * time_delay))
            loss_weights.append(1 / (2 * time_delay))

        # model = tf.keras.Model(inputs=resnet50_imagenet_model.input, outputs=outputs)
        model = tf.keras.Model(inputs=main_input, outputs=outputs)

        model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss=losses, loss_weights=loss_weights)

        trainable_count = np.sum([kb.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([kb.count_params(w) for w in model.non_trainable_weights])

        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))

        # save model image
        if False:
            file_name = 'model.png'
            print("Model image graph saved at {}".format(file_name))
            tf.keras.utils.plot_model(
                model, to_file=file_name, show_shapes=True, show_layer_names=True,
                rankdir='TB', expand_nested=False, dpi=500
            )

        EVALUATION_INTERVAL = 125
        EPOCHS = 50

        history = model.fit(train_univariate,
                            epochs=EPOCHS,
                            steps_per_epoch=EVALUATION_INTERVAL,
                            validation_data=val_univariate, validation_steps=50,
                            verbose=training_verbose)

        y_preds = model.predict(X_test_)

        ret['history'] = history.history

        # loop through every target point from t+1 to t+time_delay (included)
        for i in range(time_delay):
            time_delay_i = i + 1
            key = 't+' + str(time_delay_i)
            ret[key] = {}

            y_pred_low = y_preds[i * 2].reshape((-1))
            y_pred_high = y_preds[(i * 2) + 1].reshape((-1))

            print(y_pred_low.shape)
            # y_pred_low = y_preds[0][:, i].reshape((-1))
            # y_pred_high = y_preds[1][:, i].reshape((-1))

            y_pred_low = y_scalers[i].inverse_transform(y_pred_low)
            y_pred_high = y_scalers[i].inverse_transform(y_pred_high)

            ret[key]['time_train_predict'] = time.clock() - start_time

            # standardize BACK
            y_test_i = y_scalers[i].inverse_transform(y_test[:, i])
            if low_memory is False:
                ret[key]['y_test_original'] = y_test_i

            # save results
            if low_memory is False:
                ret[key]['y_pred_low'] = y_pred_low
                ret[key]['y_pred_high'] = y_pred_high

            # save metrics
            ret[key]['mean_absolute_error'] = mean_absolute_error(y_pred=(y_pred_low + y_pred_high) / 2,
                                                                  y_true=y_test_i)
            ret[key]['std_absolute_error'] = std_absolute_error(y_true=y_test_i, y_pred=(y_pred_low + y_pred_high) / 2)
            ret[key]['quantile_loss'] = full_quantile_loss(y_test_i, y_pred_low, y_pred_high, alpha=alpha)
            ret[key]['coverage'] = coverage(y_test_i, y_pred_low, y_pred_high)
            ret[key]['average_quantile_span'] = average_quantile_span(y_pred_low, y_pred_high)

            if verbose > 0:
                print("Time delay {}".format(time_delay_i))
                print("Time spent for training and prediction {:.2f}".format(ret[key]['time_train_predict']))
                print("Mean absolute error {}".format(ret[key]['mean_absolute_error']))
                print("Std absolute error {}".format(ret[key]['std_absolute_error']))
                print("Quantile loss {}".format(ret[key]['quantile_loss']))
                print("Coverage {}".format(ret[key]['coverage']))
                print("Average quantile span {}".format(ret[key]['average_quantile_span']))


    # single point prediction
    else:
        key = 't+' + str(time_delay)
        ret[key] = {}
        # standardize BACK
        y_test_i = y_scalers[time_delay - 1].inverse_transform(y_test[:, time_delay - 1])

        if low_memory is False:
            ret[key]['y_test_original'] = y_test_i

        # get explanatory variables for this point in time
        var_to_use_x_, var_to_use_y = regression_var_to_use(time_delay=time_delay,
                                                            column_names_x=list(column_names_x),
                                                            column_names_y=list(column_names_y),
                                                            multiple_target=True)

        var_to_use_x = var_to_use_base + var_to_use_x_
        ret[key]["var_to_use_x"] = list(np.array(column_names_x)[var_to_use_x])
        ret[key]["var_to_use_y"] = list(np.array(column_names_y)[var_to_use_y])

        # time measurement
        start_time = time.clock()

        outputs = []
        main_input = tf.keras.Input(shape=(len(var_to_use_x),), name='main_input')
        x = tf.keras.layers.Dense(100, activation='relu')(main_input)
        outputs.append(tf.keras.layers.Dense(1, activation='linear', name='quantile_out005')(x))
        outputs.append(tf.keras.layers.Dense(1, activation='linear', name='quantile_out095')(x))

        losses = [quantile_loss_wrapper(0.05), quantile_loss_wrapper(0.95)]

        model = tf.keras.Model(inputs=main_input, outputs=outputs)

        model.compile(optimizer='adam', loss=losses, loss_weights=[0.5, 0.5])

        validation_data = (X_test[:, var_to_use_x].astype(np.float),
                           (y_test[:, var_to_use_y].astype(np.float), y_test[:, var_to_use_y].astype(np.float)))

        history = model.fit(X_train[:, var_to_use_x].astype(np.float),
                            (y_train[:, var_to_use_y].astype(np.float), y_train[:, var_to_use_y].astype(np.float)),
                            epochs=5,
                            batch_size=64,
                            validation_data=validation_data,
                            verbose=training_verbose)

        print(history)
        y_preds = model.predict(X_test[:, var_to_use_x].astype(np.float))

        y_pred_low = y_preds[0]
        y_pred_high = y_preds[1]
        y_pred_low = y_scalers[time_delay - 1].inverse_transform(y_pred_low)
        y_pred_high = y_scalers[time_delay - 1].inverse_transform(y_pred_high)

        ret[key]['time_train_predict'] = time.clock() - start_time

        # save results
        if low_memory is False:
            ret[key]['y_pred_low'] = y_pred_low
            ret[key]['y_pred_high'] = y_pred_high

        # save metrics
        ret[key]['mean_absolute_error'] = mean_absolute_error(y_pred=(y_pred_low + y_pred_high) / 2, y_true=y_test_i)
        ret[key]['std_absolute_error'] = std_absolute_error(y_true=y_test_i, y_pred=(y_pred_low + y_pred_high) / 2)
        ret[key]['quantile_loss'] = full_quantile_loss(y_test_i, y_pred_low, y_pred_high, alpha=alpha)
        ret[key]['coverage'] = coverage(y_test_i, y_pred_low, y_pred_high)
        ret[key]['average_quantile_span'] = average_quantile_span(y_pred_low, y_pred_high)

        if verbose > 0:
            print("Time delay {}".format(time_delay))
            print("Time spent for training and prediction {:.2f}".format(ret[key]['time_train_predict']))
            print("Mean absolute error {}".format(ret[key]['mean_absolute_error']))
            print("Std absolute error {}".format(ret[key]['std_absolute_error']))
            print("Quantile loss {}".format(ret[key]['quantile_loss']))
            print("Coverage {}".format(ret[key]['coverage']))
            print("Average quantile span {}".format(ret[key]['average_quantile_span']))

    return ret


"""
    Plot the cross-validated training loss and validation loss with a bar plot.
    Work with only one object as well (in a list).

    Inputs:
        - mlp_ret_buffer: list of mlp ret
"""


def plot_train_validation_loss(mlp_ret_buffer, period=1):
    # keys are epochs
    validation_losses = {}
    training_losses = {}
    for mlp_ret in mlp_ret_buffer:
        history = mlp_ret['history']
        training_loss = history['loss']
        validation_loss = history['val_loss']
        for i in range(len(training_loss)):
            if i + 1 not in validation_losses:
                validation_losses[i + 1] = []

            validation_losses[i + 1].append(validation_loss[i])

            if i + 1 not in training_losses:
                training_losses[i + 1] = []

            training_losses[i + 1].append(training_loss[i])

    validation_losses_std = {}
    training_losses_std = {}
    for epoch in validation_losses:
        validation_losses_std[epoch] = np.array(validation_losses[epoch]).std()
        validation_losses[epoch] = np.array(validation_losses[epoch]).mean()

        training_losses_std[epoch] = np.array(training_losses[epoch]).std()
        training_losses[epoch] = np.array(training_losses[epoch]).mean()

    if len(mlp_ret_buffer) == 1:
        plt.plot(list(training_losses.keys()),
                 list(training_losses.values()),
                 linestyle='-', color='blue')

        plt.plot(list(validation_losses.keys()),
                 list(validation_losses.values()),
                 linestyle='--', color='red')
    else:
        plt.errorbar(list(training_losses.keys()),
                     list(training_losses.values()),
                     yerr=list(training_losses_std.values()), linestyle='-', color='blue', capsize=5)

        plt.errorbar(list(validation_losses.keys()),
                     list(validation_losses.values()),
                     yerr=list(validation_losses_std.values()), linestyle='--', color='red', capsize=5)

    plt.fill_between(list(training_losses.keys()),
                     [training_losses[epoch] - training_losses_std[epoch] for epoch in training_losses],
                     [training_losses[epoch] + training_losses_std[epoch] for epoch in training_losses],
                     color='blue', alpha=.25)
    plt.fill_between(list(validation_losses.keys()),
                     [validation_losses[epoch] - validation_losses_std[epoch] for epoch in validation_losses],
                     [validation_losses[epoch] + validation_losses_std[epoch] for epoch in validation_losses],
                     color='red', alpha=.25)

    plt.legend(['Training Loss', 'Test Loss'], fontsize=15)
    plt.xticks([t - 1 for t in list(validation_losses.keys())][::period], fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    if len(mlp_ret_buffer) == 1:
        plt.title("Training/validation loss vs epochs", fontsize=22)
    else:
        plt.title("Training/validation loss vs epochs (cross-validated)", fontsize=22)

    plt.show()


def time_series_split(df, train_set_ratio):
    n_lines = df.shape[0]

    n_lines_train_set = int(np.ceil(n_lines * train_set_ratio))

    return df.iloc[0:n_lines_train_set, :], df.iloc[n_lines_train_set:n_lines - 1, :]


"""
    This is a DEMONSTRATION module. The procedure is written in order to mimic online forecasting which is NOT the way
    gluonTS seems to be designed for. The procedure is SLOW.

    This is a wrapper around gluonts_forecasting.py. Here we standardize the results the same way as for the other
    methods. 

    Inputs:
        - Three mods for dataset
            - X, y, swp: the windows dataset (the same as for other methods). We discourage to use this one in this case.
            - event, swp: the dataset is generated from scratch/from the measurements.
            - time_series, swp: the dataset is built from loaded time series 
        - time_delay: max range of prediction
        - alpha: quantiles will be alpha and 1-alpha
        - verbose: to implement
        - random_state: reproducibility seed ; used for shuffling and partition test/train


    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


def gluonts_regression(X=None, event=None, swp=None, time_series=None, time_delay=20, alpha=0.05, verbose=0,
                       random_state=0, low_memory=False):
    from gluonts_forecasting import GluonTSWrapper

    # time measurement
    start_time = time.clock()

    if X is not None:
        train_ds, test_ds = GluonTSWrapper.data_set_to_gluonts_format(X, mode="from_X", swp=swp,
                                                                      random_state=random_state)
    elif event is not None:
        time_series = GluonTSWrapper.event_to_time_series(event)
        train_ds, test_ds = GluonTSWrapper.data_set_to_gluonts_format(time_series, mode="from_time_series", swp=swp,
                                                                      random_state=random_state)
    elif time_series is not None:
        train_ds, test_ds = GluonTSWrapper.data_set_to_gluonts_format(time_series, mode="from_time_series", swp=swp,
                                                                      random_state=random_state)

    model = GluonTSWrapper.train_deepar(train_ds, epochs=150, context_length=20, prediction_length=time_delay)
    test_ret = GluonTSWrapper.test_deepar(model, test_ds, context_length=20, prediction_length=time_delay, alpha=alpha)
    evaluation = GluonTSWrapper.evaluate_deepar(test_ret, verbose=1, prediction_length=time_delay, alpha=alpha)

    evaluation['time_train_predict'] = time.clock() - start_time

    if low_memory is True:
        return evaluation
    else:
        return (test_ret, evaluation)


# full coverage: proportion of data indeed in the quantiles
def coverage(target, low_quantile_forecast, high_quantile_forecast):
    return np.mean(np.bitwise_and(target > low_quantile_forecast, target < high_quantile_forecast))


# https://stats.stackexchange.com/questions/213050/scoring-quantile-regressor
# compute the same loss used for optimization
def quantile_loss(target, quantile_forecast, quantile):
    return np.mean((target - quantile_forecast) * (quantile - (target < quantile_forecast).astype(int)))


# compare a pair of quantiles to the target value by averaging the low and high quantiles losses
def full_quantile_loss(target, low_quantile_forecast, high_quantile_forecast, alpha):
    return (quantile_loss(target, low_quantile_forecast, alpha) + quantile_loss(target, high_quantile_forecast,
                                                                                1 - alpha)) / 2


# average distance between the two quantiles
def average_quantile_span(low_quantile_forecast, high_quantile_forecast):
    return np.mean(high_quantile_forecast - low_quantile_forecast)


# standard deviation of absolute error
def std_absolute_error(y_true, y_pred):
    return np.std(np.abs(y_true - y_pred))


def sparsity(event):
    buffer = []
    sum_observed = 0
    sum_expected = 0

    for i, id in enumerate(list(event['swimming_pool_id'].unique())):
        print(str(i) + "/" + str(len(event['swimming_pool_id'].unique())))
        measurements = event[event['swimming_pool_id'] == id]
        times = pd.Series((pd.DatetimeIndex(measurements['created']).asi8 / 10 ** 9).astype(np.int))
        if len(measurements) < 10:
            continue

        first = times.iloc[0]
        last = times.iloc[-1]

        sum_observed += len(measurements)
        sum_expected += (last - first) / 4320

        buffer += [1 - (len(measurements) / ((last - first) / 4320))]

    print(1 - (sum_observed / sum_expected))
    pd.DataFrame(buffer).hist(bins=50)
    plt.title("Histogram of sparsity ratio", fontsize=20)
    plt.xlabel("Sparsity ratio", fontsize=15)
    plt.ylabel("Number of occurences", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlim((0, 1))
    plt.grid(b=None)


def time_differences(event):
    buffer = []
    for i, id in enumerate(list(event['swimming_pool_id'].unique())):
        print(str(i) + "/" + str(len(event['swimming_pool_id'].unique())))
        measurements = event[event['swimming_pool_id'] == id]
        times = pd.Series((pd.DatetimeIndex(measurements['created']).asi8 / 10 ** 9).astype(np.int))
        diffs = list(times.diff())
        buffer += diffs

    fig, ax = plt.subplots()
    pd.DataFrame(buffer).hist(ax=ax, cumulative=-1, bins=10000)
    # pd.DataFrame(buffer).hist(ax=ax, bins=1000)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_ticks([3600, 5 * 3600, 8 * 3600, 12 * 3600, 24 * 3600, 24 * 7 * 3600, 24 * 31 * 3600, 24 * 365 * 3600])
    a = ax.get_xticks().tolist()
    a[0] = "1 hour"
    a[1] = "5 hours"
    a[2] = "8 hours"
    a[3] = "12 hours"
    a[4] = "1 day"
    a[5] = "1 week"
    a[6] = "1 month"
    a[7] = "1 year"
    ax.set_xticklabels(a)
    ax.tick_params(axis='x', rotation=30)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel("Time difference between two consecutive measurement records (logscale)", fontsize=15)
    plt.ylabel("Number of occurences (logscale)", fontsize=15)
    plt.title("Inverse/reverse CDF of the time differences between two consecutive observations", fontsize=20)
    plt.subplots_adjust(bottom=0.15)
    plt.grid(b=None)


def resampling_example(time_series, event):
    id = time_series[0]['swimming_pool_id']
    times_t = [time_series[0]['start_time'] + pd.Timedelta(seconds=i * 4320) for i in
               range(len(time_series[0]['time_series']['temp. t']))]
    temps_t = time_series[0]['time_series']['temp. t']

    measurements = event[event['swimming_pool_id'] == id]
    times_o = measurements['created']
    times_o = pd.to_datetime(pd.Series((pd.DatetimeIndex(times_o).asi8 / 10 ** 9).astype(np.int)), unit='s')
    temps_o = measurements['data_temperature']

    plt.scatter(times_t, temps_t, label="Resampled", s=4)
    plt.scatter(times_o - pd.Timedelta(seconds=16000), temps_o, label="Original", s=15)
    plt.legend(fontsize=15)
    plt.title("Cubic splines resampling", fontsize=20)
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Temperature °", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)


# example from scipy documentation
def linear_vs_cubic():
    from scipy.interpolate import interp1d
    x = np.linspace(0, 10, num=11, endpoint=True)
    y = np.cos(-x ** 2 / 9.0)
    f = interp1d(x, y)
    f2 = interp1d(x, y, kind='cubic')

    xnew = np.linspace(0, 10, num=41, endpoint=True)
    import matplotlib.pyplot as plt
    plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
    plt.legend(['data', 'linear', 'cubic'], loc='best', fontsize=15)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.title("Linear interpolation vs cubic spline interpolation", fontsize=20)
    plt.show()


"""
    Groups every modules we tried but results were either bad or not useful for last presentation.

    Author: Baptiste Debes ; b.debes@student.uliege.be
"""


class NotPresented:
    def plot_linear_coeffs(model, columns):
        n_features = len(columns)

        d = dict(zip(columns, model.coef_))
        ss = sorted(d, key=d.get, reverse=True)
        top_names = ss[0:n_features]
        print(top_names)
        print(d)

        plt.figure(figsize=(5, 5))
        plt.title("Coefficients linear regression")
        plt.bar(range(n_features), [d[i] for i in top_names], color="r", align="center")
        plt.xlim(-1, n_features)
        plt.xticks(range(n_features), top_names, rotation=50, fontsize=7)

        plt.show()

    def compare(original, resampled):
        original = original.copy()
        timestamps = original['created']
        print(timestamps)
        plt.scatter(timestamps, original['data_temperature'], color="red", marker="+", label="original")

        plt.scatter(resampled['resampled_timestamp'], resampled['data_temperature'], color="blue", marker="x",
                    label="resampled")
        plt.xlabel("Timestamp")
        plt.ylabel("data_temperature")
        plt.legend(loc="upper right")
        plt.title("data_temprature original vs resampled series")
        plt.show()

    def compare_linear_xgb(linear_y_pred, xgb_y_pred, y_test):
        """plt.scatter(y_test, -y_test + linear_y_pred, s=1, color='red', label="Linear")
        plt.scatter(y_test, -y_test + xgb_y_pred, s=1, color='blue', label="XGBoost")
        plt.xlabel("Temperature to predict")
        plt.ylabel("Residuals")
        plt.legend()
        plt.title("Residuals vs temperature to predict")
        plt.show()"""

        fig = plt.figure()

        plt.subplot(1, 2, 1)
        plt.boxplot(y_test - linear_y_pred)
        plt.yticks(list(np.arange(-30, 30, 2.5)))
        plt.gca().yaxis.grid(True)
        plt.title("Boxplot residuals linear model")
        plt.xticks([])
        plt.ylim((-30, 30))

        plt.subplot(1, 2, 2)
        plt.boxplot(y_test - xgb_y_pred)
        plt.yticks(list(np.arange(-30, 30, 2.5)))
        plt.gca().yaxis.grid(True)
        plt.title("Boxplot residuals XGBoost model")
        plt.xticks([])
        plt.ylim((-30, 30))

        plt.show()

    def plot_how_empty(event_dict):
        sum_na = 0
        sum_len = 0
        for id in event_dict:
            measures = event_dict[id]
            na_index = pd.isna(measures['data_temperature'])
            sum_na += sum(na_index)
            sum_len += len(na_index)

        print(sum_na / sum_len)
        id = "1b0db400-c90b-4fb9-96ec-03d86b8a6cd6"

        measures = event_dict[id]
        na_index = pd.isna(measures['data_temperature'])
        print(sum(na_index) / len(na_index))
        plt.scatter(measures.loc[na_index, 'resampled_timestamp'], np.full(np.sum(na_index), 1), s=1, c='red',
                    label="missing observations")
        plt.scatter(measures.loc[~na_index, 'resampled_timestamp'], measures.loc[~na_index, 'data_temperature'], s=0.5,
                    c='blue', label="temperature measures")
        plt.title("Missingness of temperature observation = {:.2f}%".format(100 * sum(na_index) / len(na_index)))
        plt.xlabel("timestamps")
        plt.ylabel("temperature in ° C")
        plt.legend()
        plt.show(block=False)

    def gaussian_process_forecasting(time_series):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
        # import gpytorch
        # import torch

        time_series = time_series.dropna()
        swp_ids = pd.unique(list(time_series['swimming_pool_id']))
        y_pred_means = []
        y_pred_lows = []
        y_pred_highs = []
        y_tests = []
        j = 0
        for swp_id in swp_ids:
            if j == 5:
                break
            print("Pool {}/{}".format(j + 1, len(swp_ids)))
            swp_data = time_series.loc[time_series['swimming_pool_id'] == swp_id]
            y = swp_data['data_temperature'].values.reshape((-1))
            times = swp_data['resampled_timestamp'].values.reshape((-1, 1))

            prediction_length = 20
            train_size = len(y) - prediction_length

            min_length = 300
            if len(y) < min_length:
                continue

            regressor = BaggingGPytorch(kernel=None, max_samples=-1, n_estimators=1, max_training_iter=500, verbose=1)
            regressor.fit(times[0:train_size], y[0:train_size])

            y_pred_mean, y_pred_low, y_pred_high = regressor.predict(
                times[train_size:-1].reshape((-1, 1)).astype(np.float64))
            print(y_pred_mean)
            j += 1

            y_pred_lows.append(y_pred_low)
            y_pred_highs.append(y_pred_high)
            y_pred_means.append(y_pred_mean)
            y_tests.append(y[train_size:-1])
            """print(y_test.shape)
            print(y_pred_high.shape)
            print("Mean absolute error " + str(mean_absolute_error(y_pred=y_pred_mean, y_true=y_test)))
            print("Quantile loss {}".format(full_quantile_loss(y_test, y_pred_low, y_pred_high, alpha=0.05)))
            print("Coverage {}".format(coverage(y_test, y_pred_low, y_pred_high)))"""

        y_pred_low = np.concatenate(y_pred_lows)
        y_pred_high = np.concatenate(y_pred_highs)
        y_pred_mean = np.concatenate(y_pred_means)
        y_test = np.concatenate(y_tests)
        print("Mean absolute error " + str(mean_absolute_error(y_pred=y_pred_mean, y_true=y_test)))
        print("Root mean squared error " + str(mean_squared_error(y_pred=y_pred_mean, y_true=y_test) ** 0.5))
        print("Quantile loss {}".format(full_quantile_loss(y_test, y_pred_low, y_pred_high, alpha=0.05)))
        print("Coverage {}".format(coverage(y_test, y_pred_low, y_pred_high)))

    def sort_by_na(time_series):
        swp_ids = pd.unique(list(time_series['swimming_pool_id']))

        buffer = []
        ids = []
        lengths = []
        for swp_id in swp_ids:
            swp_data = time_series.loc[time_series['swimming_pool_id'] == swp_id]
            y = swp_data['data_temperature']

            print(y.isna().sum())
            ids.append(swp_id)
            buffer.append(y.isna().sum() / len(y))
            lengths.append(len(y))

        indexes = np.argsort(np.array(buffer))
        print(indexes)
        print(list(zip([buffer[i] for i in indexes], [lengths[i] for i in indexes], [ids[i] for i in indexes])))

    """class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.PeriodicKernel() #gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())  #gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    class BaggingGPytorch():

        def __init__(self, kernel, max_samples=1000, n_estimators=10, max_training_iter=25, verbose=0):
            self._n_estimators = n_estimators
            self._max_samples = max_samples
            self._kernel = kernel
            self._max_training_iter = max_training_iter
            self._models = []
            self._verbose = verbose

        def fit(self, X, y):

            for i in range(self._n_estimators):
                if self._max_samples != -1:
                    X_b, y_b = resample(X, y, n_samples=self._max_samples, random_state=i)
                else:
                    X_b = X
                    y_b = y

                X_b = torch.tensor(X_b)
                y_b = torch.tensor(y_b)

                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                regressor = ExactGPModel(X_b, y_b, likelihood).double()
                # Use the adam optimizer
                optimizer = torch.optim.Adam([
                    {'params': regressor.parameters()},  # Includes GaussianLikelihood parameters
                ], lr=0.1)

                # "Loss" for GPs - the marginal log likelihood
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, regressor)

                for i in range(self._max_training_iter):
                    # Zero gradients from previous iteration
                    optimizer.zero_grad()
                    # Output from model
                    output = regressor(X_b)
                    # Calc loss and backprop gradients
                    loss = -mll(output, y_b)
                    loss.backward()
                    if self._verbose != 0:
                        print('Iter %d/%d - Loss: %.3f  ' % (
                            i + 1, self._max_training_iter, loss.item(),
                            #regressor.covar_module.base_kernel.lengthscale.item(),
                            #regressor.covar_module.base_kernel.period_length.item(),
                            #regressor.likelihood.noise.item()
                        ))
                    optimizer.step()


                self._models.append((regressor, likelihood))

                if self._verbose > 0:
                    print("Fitting step: {}".format(i))

            return self

        def predict(self, X):
            n_lines = X.shape[0]
            X = torch.tensor(X).double()

            avg_mean = np.zeros(n_lines)
            avg_low = np.zeros(n_lines)
            avg_high = np.zeros(n_lines)
            for i in range(self._n_estimators):
                (model, likelihood) = self._models[i]

                # initialize likelihood and model
                model.eval()
                likelihood.eval()

                # Make predictions by feeding model through likelihood
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = likelihood(model(X))
                    y_pred_mean = observed_pred.mean.numpy()
                    y_pred_low, y_pred_high = observed_pred.confidence_region()
                    avg_mean += y_pred_mean
                    avg_low  += y_pred_low.numpy()
                    avg_high  += y_pred_high.numpy()

                if self._verbose > 0:
                    print("Bagging step: {}".format(i))

            avg_mean /= self._n_estimators
            avg_low /= self._n_estimators
            avg_high /= self._n_estimators

            return avg_mean, avg_low, avg_high
    """

    def gaussian_process_regression(X, y, kind="temp"):
        random_seed = 0
        random.seed(random_seed)

        def dummify(X, columns):

            coded_vars = list()
            coded_vars_length = list()
            for c in columns:
                enc = OneHotEncoder(handle_unknown='ignore')
                encoded = enc.fit_transform(X[c].to_numpy().reshape(-1, 1))
                coded_vars.append(encoded)
                coded_vars_length.append(encoded.shape[1])

            # merge horizontally
            buffer = coded_vars[0].toarray()
            for i in range(1, len(coded_vars)):
                buffer = np.hstack((buffer, coded_vars[i].toarray()))

            X = X.drop(columns, axis=1)  # delete newly encoded columns

            i = 0
            for c in columns:
                coding_size = coded_vars_length[i]
                for j in range(coding_size):
                    new_column_name = c + " OH_" + str(j)
                    X[new_column_name] = buffer[:, i]

                i += 1

            return X

        if kind == "temp":
            categorical_columns = X.columns[[2, 3, 4, 5, 6, 7]]
            quant_tresh = 8
        else:
            categorical_columns = X.columns[[2, 3, 4, 5]]
            quant_tresh = 6

        X = dummify(X, categorical_columns)

        column_names = X.columns

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X.values[:, quant_tresh:], y)
        scaled_X = np.hstack([X.iloc[:, 0:quant_tresh].values, scaled_X])
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.values)
        X = pd.DataFrame(scaled_X, columns=column_names)
        y = pd.DataFrame(y)

        # X = X.iloc[:,[0, 1, 10, 11, 12, 13]]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = random_seed)
        X_train, y_train, X_test, y_test = temperature_build_db_train_test(X, y, train_ratio=0.9)

        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy().ravel()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy().ravel()

        from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel
        kernel = DotProduct() + WhiteKernel()
        """regressor = BaggingGaussianProcess(kernel=kernel, max_samples=1000, n_estimators=5, verbose=1)
        regressor.fit(X_train[:, 2:], y_train)
        y_pred_mean, y_pred_low, y_pred_high = regressor.predict(X_test[:, 2:])"""

        regressor = BaggingGPytorch(kernel=kernel, max_samples=3500, n_estimators=1, verbose=0, max_training_iter=40)
        print(X_train[:, 2:].shape)
        print(y_train.shape)
        regressor.fit(X_train[:, 2:].astype(np.float64), y_train.astype(np.float64))
        y_pred_mean, y_pred_low, y_pred_high = regressor.predict(X_test[:, 2:].astype(np.float64))

        y_pred_low = y_scaler.inverse_transform(y_pred_low)
        y_pred_high = y_scaler.inverse_transform(y_pred_high)
        y_pred_mean = y_scaler.inverse_transform(y_pred_mean)
        y_test = y_scaler.inverse_transform(y_test)

        print("Mean absolute error " + str(mean_absolute_error(y_pred=y_pred_mean, y_true=y_test)))
        print("Quantile loss {}".format(full_quantile_loss(y_test, y_pred_low, y_pred_high, alpha=0.05)))
        print("Coverage {}".format(coverage(y_test, y_pred_low, y_pred_high)))
        print("Average quantile span {}".format(average_quantile_span(y_pred_low, y_pred_high)))
        # print("Max absolute error " + str(np.max(np.abs(y_pred-y_test))))

        regressor = None

        return regressor, column_names, X_test, y_test, y_pred_mean, y_pred_low, y_pred_high

    class BaggingGaussianProcess():

        def __init__(self, kernel, max_samples=1000, n_estimators=10, verbose=0):
            self._n_estimators = n_estimators
            self._max_samples = max_samples
            self._kernel = kernel
            self._models = []
            self._verbose = verbose

        def fit(self, X, y):
            from sklearn.gaussian_process import GaussianProcessRegressor

            for i in range(self._n_estimators):
                X_b, y_b = resample(X, y, n_samples=self._max_samples, random_state=i)

                regressor = GaussianProcessRegressor(kernel=self._kernel, random_state=i, normalize_y=True).fit(X_b,
                                                                                                                y_b)
                self._models.append(regressor)

                if self._verbose > 0:
                    print("Fitting step: {}".format(i))

            return self

        def predict(self, X):
            n_lines = X.shape[0]

            avg_mean = np.zeros(n_lines)
            avg_low = np.zeros(n_lines)
            avg_high = np.zeros(n_lines)
            for i in range(self._n_estimators):
                means, stds = self._models[i].predict(X, return_std=True)
                avg_mean += means
                avg_low += means - 2 * stds
                avg_high += means + 2 * stds

                if self._verbose > 0:
                    print("Bagging step: {}".format(i))

            avg_mean /= self._n_estimators
            avg_low /= self._n_estimators
            avg_high /= self._n_estimators

            return avg_mean, avg_low, avg_high

    def xgb_regression(X, y):
        random_seed = 0
        random.seed(random_seed)

        def dummify(X, columns):

            coded_vars = list()
            coded_vars_length = list()
            for c in columns:
                enc = OneHotEncoder(handle_unknown='ignore')
                encoded = enc.fit_transform(X[c].to_numpy().reshape(-1, 1))
                coded_vars.append(encoded)
                coded_vars_length.append(encoded.shape[1])

            # merge horizontally
            buffer = coded_vars[0].toarray()
            for i in range(1, len(coded_vars)):
                buffer = np.hstack((buffer, coded_vars[i].toarray()))

            X = X.drop(columns, axis=1)  # delete newly encoded columns

            i = 0
            for c in columns:
                coding_size = coded_vars_length[i]
                for j in range(coding_size):
                    new_column_name = c + " OH_" + str(j)
                    X[new_column_name] = buffer[:, i]

                i += 1

            return X

        categorical_columns = X.columns[[2, 3, 4, 5, 6, 7]]
        X = dummify(X, categorical_columns)

        column_names = X.columns

        # prepare data
        # X, y = shuffle(X, y, random_state=random_seed)
        # X = X[0:500000, :]
        # y = y[0:500000]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = random_seed)
        X_train, y_train, X_test, y_test = temperature_build_db_train_test(X, y, train_ratio=0.9)
        # X_train = preprocessing.scale(X_train)
        # X_test = preprocessing.scale(X_test)

        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy().ravel()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy().ravel()

        # regressor = ExtraTreesRegressor(n_estimators=200, criterion="mae", n_jobs=7, verbose=100, max_depth=4, random_state=random_seed)
        # regressor.fit(X_train, y_train)
        import xgboost as xgb

        regressor = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=7, learning_rate=0.15, max_depth=10,
                                     n_estimators=200, subsample=0.90,
                                     tree_method='gpu_hist', gpu_id=0, silent=False, verbosity=2, booster='gbtree')

        regressor.fit(X_train[:, 2:], y_train, eval_metric="rmse", verbose=True)

        y_pred = regressor.predict(X_test[:, 2:])

        print("Mean absolute error " + str(mean_absolute_error(y_pred=y_pred, y_true=y_test)))
        # print("Max absolute error " + str(np.max(np.abs(y_pred-y_test))))

        # fit on all dataset
        """regressor = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=7, learning_rate=0.1, max_depth=8,
                                     n_estimators=1000,
                                     tree_method='gpu_hist', gpu_id=0, verbosity=2, booster='gbtree')

        regressor.fit(X[:,2:], y)"""
        regressor = None

        return regressor, column_names, X_test, y_test, y_pred

    def error_repartition(y_test, y_pred):
        error = y_pred - y_test
        plt.scatter(y_test, error, s=1)
        plt.xlabel("Test set temperature")
        plt.ylabel("Error in the prediction")
        plt.title("Error and the temperature to predict")
        plt.show()

        """from sklearn.neighbors import KernelDensity
        X_plot = np.linspace(-50, 50, 1000).reshape((-1,1))
        kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(error.reshape((-1,1)))
        log_dens = kde.score_samples(X_plot)
        plt.plot(X_plot[:, 0], np.exp(log_dens))"""

        """hist, bins = np.histogram(error, bins=1000)
        sum = np.sum(hist)
        hist = [x/sum for x in hist]

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        import scipy.stats as stats

        mu = np.mean(error)
        sigma = np.std(error)
        x = np.linspace(mu - 6 * sigma, mu + 6 * sigma, 400)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r')


        plt.show()"""

        """ from scipy.stats import normaltest
        print(normaltest(error))

        from scipy.stats import anderson
        print(anderson(error))"""

        from statsmodels.graphics.gofplots import qqplot
        from matplotlib import pyplot

        qqplot(error, line='s')
        pyplot.show()

    def gluon_forecasting(time_series):
        from gluonts_forecasting.dataset.common import ListDataset
        from gluonts_forecasting.model.simple_feedforward import SimpleFeedForwardEstimator
        from gluonts_forecasting.model.deepar import DeepAREstimator
        from gluonts_forecasting.trainer import Trainer
        from gluonts_forecasting.evaluation.backtest import make_evaluation_predictions

        import mxnet as mx
        print("Number GPU's: " + str(mx.context.num_gpus()))
        # time_series = time_series.dropna()
        swp_ids = pd.unique(list(time_series['swimming_pool_id']))

        # create the train set and test set
        y_tests = []
        j = 0
        training_buffer = []
        test_buffer = []
        for swp_id in swp_ids:
            # swp_id =  '5c0a72f5-3377-41cc-8399-34d23804b5c2'#'d8516fed-a22c-467f-9d9c-86e592a5b666'
            if j == 100:
                break

            print("Pool {}/{}".format(j + 1, len(swp_ids)))

            swp_data = time_series.loc[time_series['swimming_pool_id'] == swp_id]
            y = swp_data['data_temperature'].values.reshape((-1))
            irradiance = swp_data['solar_irrandiance'].values.reshape((-1, 1))
            weather = swp_data['weather_temp'].values.reshape((-1, 1))
            times = swp_data['resampled_timestamp'].values.reshape((-1, 1))
            test_proportion = 0.1
            test_size = int(np.ceil(len(y) * test_proportion))
            train_size = len(y) - test_size
            plt.plot(times, y)
            # return
            context_length = 20

            training_buffer.append({'target': y[0:train_size],
                                    'start': pd.Timestamp(times[0, 0], unit='s'),
                                    # 'feat_dynamic_real': np.hstack([irradiance[train_size+k:train_size+k+num_samples], weather[train_size+k:train_size+k+num_samples]])
                                    'item_id': j
                                    })
            """for k in range(test_size):
                test_buffer.append({'target': y[train_size + k:train_size + k + context_length],
                                    'start': pd.Timestamp(times[train_size + k, 0], unit='s'),
                                    #'feat_dynamic_real': np.hstack([irradiance[train_size + k:train_size + k + num_samples],
                                    #                                weather[train_size + k:train_size + k + num_samples]
                                    'item_id': j
                                })"""

            test_buffer.append({'target': y[train_size:-1],
                                'start': pd.Timestamp(times[train_size, 0], unit='s'),
                                # 'feat_dynamic_real': np.hstack([irradiance[train_size+k:train_size+k+num_samples], weather[train_size+k:train_size+k+num_samples]])
                                'item_id': j
                                })

            y_test = y[train_size:]

            y_tests.append(y_test)

            j += 1

        # create model
        estimator = DeepAREstimator(
            prediction_length=10,
            context_length=context_length,
            freq='1.2H',
            trainer=Trainer(ctx="gpu",
                            epochs=10,
                            num_batches_per_epoch=1000
                            )
        )
        # train
        train_ds = ListDataset(training_buffer, freq='4320s')
        predictor = estimator.train(train_ds)

        test_ds = ListDataset(test_buffer, freq='4320s')
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,  # test dataset
            predictor=predictor,  # predictor
            num_samples=100,  # number of sample paths we want for evaluation
        )

        forecasts = list(forecast_it)
        tss = list(ts_it)
        """ts_entry = tss[0]
        test_ds_entry = next(iter(test_ds))
        forecast_entry = forecasts[0]

        def plot_prob_forecasts(ts_entry, forecast_entry):
            plot_length = 150
            prediction_intervals = (50.0, 90.0)
            legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][
                                                             ::-1]

            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
            forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
            plt.grid(which="both")
            plt.legend(legend, loc="upper left")
            plt.show()"""

        from gluonts_forecasting.evaluation import Evaluator
        import json
        evaluator = Evaluator(quantiles=[0.05, 0.5, 0.95])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
        print(json.dumps(agg_metrics, indent=4))

        return forecast_it, ts_it

        """k = 0
        y_pred_mean = []
        y_pred_low = []
        y_pred_high = []
        for t in test_buffer:
            test_ds = ListDataset([t], freq='4320s')
            forecast_it = predictor.predict(test_ds)
            forecasts = list(forecast_it)
            print([f.quantile(0.5) for f in forecasts])
            y_pred_mean.append(forecasts[0].quantile(0.5))
            y_pred_low.append(forecasts[0].quantile(0.05))
            y_pred_high.append(forecasts[0].quantile(0.95))
            print("{}/{}".format(k+1, len(test_buffer)))
            k += 1


        y_pred_low = np.array(y_pred_mean)
        y_pred_high = np.array(y_pred_low)
        y_pred_mean = np.array(y_pred_high)
        y_test = np.concatenate(y_tests)
        print(y_pred_low)
        print(y_pred_high)
        print(y_pred_mean)
        print(y_test)
        y_test[np.isnan(y_test)] = 0"""
        print("Mean absolute error " + str(mean_absolute_error(y_pred=y_pred_mean, y_true=y_test)))
        print(
            "Quantile loss {}".format(full_quantile_loss(y_test, np.abs(y_pred_low), np.abs(y_pred_high), alpha=0.05)))
        print("Coverage {}".format(coverage(y_test, np.abs(y_pred_low), np.abs(y_pred_high))))

    def knn_regression(X, y):
        from sklearn.preprocessing import StandardScaler

        random_seed = 0
        random.seed(random_seed)

        def dummify(X, columns):

            coded_vars = list()
            coded_vars_length = list()
            for c in columns:
                enc = OneHotEncoder(handle_unknown='ignore')
                encoded = enc.fit_transform(X[c].to_numpy().reshape(-1, 1))
                coded_vars.append(encoded)
                coded_vars_length.append(encoded.shape[1])

            # merge horizontally
            buffer = coded_vars[0].toarray()
            for i in range(1, len(coded_vars)):
                buffer = np.hstack((buffer, coded_vars[i].toarray()))

            X = X.drop(columns, axis=1)  # delete newly encoded columns

            i = 0
            for c in columns:
                coding_size = coded_vars_length[i]
                for j in range(coding_size):
                    new_column_name = c + " OH_" + str(j)
                    X[new_column_name] = buffer[:, i]

                i += 1

            return X

        quant_tresh = 8
        categorical_columns = X.columns[2:quant_tresh]
        columns = X.columns
        X = dummify(X, categorical_columns)
        quant_tresh = quant_tresh + (len(X.columns) - len(columns))  # update quant tresh
        columns = X.columns

        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X.values[:, quant_tresh:], y)
        scaled_X = np.hstack([X.iloc[:, 0:quant_tresh].values, scaled_X])
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.values)
        X = pd.DataFrame(scaled_X, columns=columns)
        y = pd.DataFrame(y)

        X_train, y_train, X_test, y_test = temperature_build_db_train_test(X, y, train_ratio=0.9)

        from sklearn.neighbors import KNeighborsRegressor
        regressor = KNeighborsRegressor(n_neighbors=25, weights='uniform', n_jobs=7, algorithm='auto').fit(
            X_train.iloc[:, 2:], y_train)
        y_pred = regressor.predict(X_test.iloc[:, 2:])

        print("Mean absolute error " + str(mean_absolute_error(y_pred=y_pred, y_true=y_test)))

        regressor = None
        return regressor, columns, y_test, y_pred

    def regressor_compare_plot_gt(X_test, y_test, y_pred):
        swp_ids = np.unique(X_test[:, 0])

        print(X_test[:, 1])
        # for id in swp_ids:
        if True:
            id = random.choice(swp_ids)

            y_t = y_test[X_test[:, 0] == id]
            y_p = y_pred[X_test[:, 0] == id]
            X_t = X_test[X_test[:, 0] == id]
            times = X_t[:, 1]

            plt.scatter(times, y_t, s=1, c='red')
            plt.plot(times, y_p)

            plt.show()

            return

    def regressor_compare_plot_gt_quantile(X_test, y_test, y_pred_low, y_pred_high, y_pred_mean=None, color='green',
                                           kind="temp"):
        swp_ids = np.unique(X_test[:, 0])
        # random.seed(0)
        print(X_test[:, 1])
        # for id in swp_ids:
        if True:
            id = random.choice(swp_ids)
            id = "a737200f-ca6a-46ac-9ad5-f32ad87e59ee"  # temp
            # id = "a1bb2c31-0660-445b-9479-a2a568e9c477" # ph
            print(id)
            print(id in X_test[:, 0])
            # id = "912ddc34-10b8-48b2-8d6a-89148f0f0baa"
            y_t = y_test[X_test[:, 0] == id]
            y_p_l = y_pred_low[X_test[:, 0] == id]
            y_p_h = y_pred_high[X_test[:, 0] == id]
            if y_pred_mean is not None:
                y_p_m = y_pred_mean[X_test[:, 0] == id]

            X_t = X_test[X_test[:, 0] == id]
            times = X_t[:, 1]

            datetime_index = pd.to_datetime(times, unit='s')

            plt.fill_between(list(datetime_index), list(y_p_l), list(y_p_h), color=color, alpha=0.3, interpolate=False,
                             step='pre', label="Quantiles space")
            plt.scatter(datetime_index, y_t, s=1, c='red', label="Measurements")

            if y_pred_mean is None:
                plt.step(datetime_index, (y_p_l + y_p_h) / 2, color=color, label="Main predictor")
            else:
                plt.step(datetime_index, y_p_m, color=color, label="Main predictor")

            plt.legend()

            plt.xlabel("Timestamps")
            if kind == "temp":
                plt.ylim((0, 50))
                plt.title("Temperature predictions with 0.05-0.95 quantiles/confidence region")
                plt.ylabel("Temperature in ° C")
            else:
                plt.ylim((0, 12))
                plt.title("pH predictions with 0.05-0.95 quantiles/confidence region")
                plt.ylabel("pH")

            # plt.plot(times, y_p_l)
            # plt.plot(times, y_p_h)
            plt.show()

            return

    def linear_stacking(X, y):
        random_seed = 0

        X_array, y_array = shuffle(X, np.array(y).ravel(), random_state=random_seed)
        X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.33, random_state=random_seed)

        linear_regressor, _, y_inside_test_linear, y_inside_pred = linear_regression(
            pd.DataFrame(X_train, columns=X.columns),
            pd.DataFrame(y_train))
        lgbm_regressor, _, y_inside_test_lightGBM, y_inside_pred = lightGBM_regression(
            pd.DataFrame(X_train, columns=X.columns),
            pd.DataFrame(y_train))
        from sklearn.linear_model import LinearRegression
        stacker = LinearRegression()
        stacker.fit(np.hstack((y_inside_test_linear.reshape(-1, 1), y_inside_test_lightGBM.reshape(-1, 1))),
                    y_inside_pred)

        X_test_linear = X_test.iloc[:, 9:]
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_test_linear = scaler.fit_transform(X_test_linear, y_test)

        def integer_encode(X, columns):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X[columns] = X[columns].apply(lambda col: le.fit_transform(col))
            return X

        categorical_columns = X.columns[[0, 1, 2, 3, 4, 5]]
        X_test_lightGBM = integer_encode(X_test.copy(), categorical_columns).to_numpy()

        linear_prediction = linear_regressor.predict(X_test_linear)
        lgbm_prediction = lgbm_regressor.predict(X_test_lightGBM)

        y_pred = stacker.predict(np.hstack((linear_prediction.reshape(-1, 1), lgbm_prediction.reshape(-1, 1))))

        print("Mean absolute error " + str(mean_absolute_error(y_pred=y_pred, y_true=y_test)))

    def SVR_regression(X, y):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import LassoCV
        from sklearn.svm import LinearSVR

        random_seed = 0
        random.seed(random_seed)

        X, y = shuffle(X, np.array(y).ravel(), random_state=random_seed)

        vars_to_keep = X.columns[9:]
        X = X[vars_to_keep]
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X, y)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.33, random_state=random_seed)

        regressor = LinearSVR().fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        print("Mean absolute error " + str(mean_absolute_error(y_pred=y_pred, y_true=y_test)))

        regressor = LinearSVR().fit(X, y)

        return regressor, vars_to_keep, y_test, y_pred

    def adaboost_regression(X, y):
        from sklearn.preprocessing import StandardScaler

        random_seed = 0
        random.seed(random_seed)

        def dummify(X, columns):

            coded_vars = list()
            coded_vars_length = list()
            for c in columns:
                enc = OneHotEncoder(handle_unknown='ignore')
                encoded = enc.fit_transform(X[c].to_numpy().reshape(-1, 1))
                coded_vars.append(encoded)
                coded_vars_length.append(encoded.shape[1])

            # merge horizontally
            buffer = coded_vars[0].toarray()
            for i in range(1, len(coded_vars)):
                buffer = np.hstack((buffer, coded_vars[i].toarray()))

            X = X.drop(columns, axis=1)  # delete newly encoded columns

            i = 0
            for c in columns:
                coding_size = coded_vars_length[i]
                for j in range(coding_size):
                    new_column_name = c + " OH_" + str(j)
                    X[new_column_name] = buffer[:, i]

                i += 1

            return X

        quantitative_columns_to_keep = [6] + list(range(9, X.shape[1]))
        categorical_columns = X.columns[[0, 1, 2, 3, 4, 5]]

        X = dummify(X, categorical_columns)
        columns = X.columns

        X = X.to_numpy()
        y = y.to_numpy().ravel()

        # scaler = StandardScaler()
        # scaled_X = scaler.fit_transform(X, y)

        X, y = shuffle(X, y, random_state=random_seed)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_seed)

        from sklearn.ensemble import AdaBoostRegressor
        regressor = AdaBoostRegressor(random_state=0, n_estimators=5).fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        print("Mean absolute error " + str(mean_absolute_error(y_pred=y_pred, y_true=y_test)))

        regressor = AdaBoostRegressor(random_state=0, n_estimators=5).fit(X, y)

        return regressor, columns, y_test, y_pred

    # V1
    def ph_build_db(event, swp, time_horizon, time_delay):
        def swp_del_miss(swp):
            n_rows = swp.shape[0]
            # eliminate pools for which
            #  - type is missing
            #  - location is missing
            #  - kind is missing
            # -  volume capacity is missing
            # -  sanitizer_process
            vars = ['type', 'location', 'kind', 'volume_capacity', 'sanitizer_process']
            to_keep = np.full((n_rows), True)
            for var in vars:
                to_keep = to_keep & np.logical_not(swp[var].isna())

            return swp[to_keep]

        # eliminate non-necessary swp variables
        def swp_var_of_interest(swp):
            vars_to_keep = set(['swimming_pool_id', 'type', 'location', 'kind', 'volume_capacity', 'sanitizer_process'])
            vars_to_drop = set(swp.columns) - vars_to_keep
            swp = swp.drop(columns=vars_to_drop)
            return swp

        # joint between swp pools and event pools : keep only pools in both
        def joint_over_swp_id(event, swp):
            swp_id_to_keep = swp['swimming_pool_id']
            return event.loc[event['swimming_pool_id'].isin(swp_id_to_keep)]

        def event_del_miss(event):
            # eliminate pools for which
            #  - data_temperature is missing
            #  - weather_temp is missing
            #  - weather_humidity is missing
            # -  weather_pressure capacity is missing
            n_rows = event.shape[0]
            vars = ["swimming_pool_id", "data_temperature", "weather_temp", "data_ph", "data_conductivity", "data_orp"]
            to_keep = np.full((n_rows), True)
            for var in vars:
                to_keep = to_keep & np.logical_not(event[var].isna())

            return event[to_keep]

        def event_var_of_interest(event):
            vars_to_keep = set(
                ["swimming_pool_id", "created", "swimming_pool_id", "data_temperature", "weather_temp", "data_ph",
                 "data_conductivity", "data_orp"])
            vars_to_drop = set(event.columns) - vars_to_keep
            event = event.drop(columns=vars_to_drop)
            return event

        def del_too_few_obs(event, swp, min_obs):
            swp_ids = swp['swimming_pool_id']

            for id in swp_ids:
                measures = event.loc[event['swimming_pool_id'] == id]
                if len(measures) < min_obs:
                    index = event[event['swimming_pool_id'] == id].index
                    event.drop(index, inplace=True)

            swp_id_swp = swp["swimming_pool_id"]
            swp_id_event = event["swimming_pool_id"]
            intersection = set(swp_id_swp).intersection(set(swp_id_event))

            event = event.loc[event['swimming_pool_id'].isin(intersection)]
            swp = swp.loc[swp['swimming_pool_id'].isin(intersection)]

            event.reset_index(inplace=True)
            swp.reset_index(inplace=True)

            return event, swp

        event = event.copy()

        swp = swp_del_miss(swp)
        swp = swp_var_of_interest(swp)
        event = joint_over_swp_id(event, swp)
        event = event_del_miss(event)
        event = event_var_of_interest(event)
        event, swp = del_too_few_obs(event, swp, min_obs=100)

        event['created'] = pd.Series((pd.DatetimeIndex(event['created']).asi8 / 10 ** 9).astype(np.int))

        def build_X_y(event, swp, time_horizon, time_delay):
            swp_ids = swp['swimming_pool_id']

            n_measures = event.shape[0]

            swp_explanatory_variables = ['type', 'location', 'kind', 'sanitizer_process', 'volume_capacity']
            event_explanatory_variables = ["data_temperature", "weather_temp", "data_ph", "data_conductivity",
                                           "data_orp"]

            categorical_variables = ['type', 'location', 'kind', 'sanitizer_process']

            def create_columns():
                columns = []
                columns += ["swimming_pool_id"]
                columns += ["timestamp"]
                columns += swp_explanatory_variables
                columns += ['day_year', 'seconds_day']

                for var in ["data_temperature", "data_ph", "data_conductivity", "data_orp"]:
                    for i in range(time_horizon):
                        if i == 0:
                            columns.append(var + " t")
                        else:
                            columns.append(var + " t-" + str(i))

                for i in range(time_delay):
                    if i == 0:
                        columns.append('solar_irradiance' + " t")
                    else:
                        columns.append('solar_irradiance' + " t+" + str(i))

                for i in range(time_delay):
                    if i == 0:
                        columns.append('weather_temp' + " t")
                    else:
                        columns.append('weather_temp' + " t+" + str(i))

                return columns

            print(create_columns())
            columns = create_columns()

            explanatory_variables = swp_explanatory_variables + event_explanatory_variables

            final_n_rows = n_measures - time_horizon * len(swp_ids) - time_delay * len(
                swp_ids)  # wait for the time_horizon first measures

            """X = pd.DataFrame(index=range(final_n_rows), columns=columns)
            y = pd.DataFrame(index=range(final_n_rows), columns=['temp. t+'+str(time_delay)])"""
            # swp id
            # timestamp
            # day_year
            # second_day
            # swp_explanatory_variables
            # time_horizon * (data_pH + data_temperature + data_orp)
            # time_delay * (weather_temp + soldar_irradiance)
            X = np.full((final_n_rows, 2 + 2 + len(swp_explanatory_variables) + 4 * time_horizon + 2 * time_delay),
                        None)
            y = np.zeros(final_n_rows)

            n_lines_filled = 0
            n_id = 0
            for id in swp_ids:
                print(n_id)
                n_id += 1
                swp_measures = event.loc[event['swimming_pool_id'] == id]
                # swp_measures.drop_duplicates(subset="created", keep='first', inplace=True)

                swp_data = swp.loc[swp['swimming_pool_id'] == id].iloc[0]

                timestamps = swp_measures['created']
                timestamps_diffs = timestamps.diff()[1:]

                def resample():
                    resampling_T = 4320  # 4320 sec
                    n_max_new_samples = int(np.ceil((timestamps.iloc[-1] - timestamps.iloc[0]) / resampling_T))

                    resampled_measures = pd.DataFrame(
                        data=np.empty((n_max_new_samples, 1 + len(event_explanatory_variables))),
                        columns=['resampled_timestamp'] + event_explanatory_variables)

                    # visit the time series
                    # cut the series into not too far apart episodes
                    time_series_ranges = []
                    range_start = 0
                    for measure_index in range(0, swp_measures.shape[0] - 1):

                        # there is a cutting point
                        if timestamps_diffs.iloc[measure_index] > 6 * 3600 or timestamps_diffs.iloc[
                            measure_index] < 0.2 * 3600:
                            index_range = (range_start, measure_index)

                            if range_start != measure_index:  # don't add undefined ranges
                                time_series_ranges.append(index_range)
                            range_start = measure_index + 1

                    # contains lists ; each list for a range
                    # in each list : models for each variable
                    interpolation_models_X = []
                    measures_array = swp_measures[event_explanatory_variables].to_numpy()
                    for time_range in time_series_ranges:
                        start = time_range[0]
                        end = time_range[1]

                        series_model = []
                        for index_var in range(measures_array.shape[1]):
                            series_model.append(
                                CubicSpline(timestamps[start:end + 1], measures_array[start:end + 1, index_var]))

                        interpolation_models_X.append(((start, end), series_model))

                    resampled_index = 0  # number of lines resampled
                    for time_range in interpolation_models_X:
                        index_range = time_range[0]
                        start = index_range[0]
                        end = index_range[1]
                        models = time_range[1]

                        times = np.arange(timestamps.iloc[start], timestamps.iloc[end], resampling_T)
                        resampled_measures.iloc[resampled_index:resampled_index + len(times), 0] = times
                        var = 1
                        for model in models:
                            interpolation = model(times)
                            resampled_measures.iloc[resampled_index:resampled_index + len(times), var] = interpolation

                            var += 1

                        resampled_index += len(times)

                    return resampled_measures.iloc[0:resampled_index, :]

                # return resample(), swp_measures[['created','data_temperature']+event_explanatory_variables]

                resampled_swp_measures = resample()
                resampled_timestamps_diffs = resampled_swp_measures['resampled_timestamp'].diff()[1:]

                # some time windows encounter great shift in the time difference between
                # two measures ; this makes the approximation incoherent if used as it is
                # everything from the time horizon past measures to the predictions on weather and the target
                # temperature must be time coherent
                def is_valid_time_window(measure_index):
                    # looks for anomalies
                    for i in range(-time_horizon, time_delay):
                        if resampled_timestamps_diffs.iloc[measure_index + i] > 6 * 3600:
                            return False

                    return True

                # rewrite
                for measure_index in range(time_horizon, len(resampled_swp_measures) - time_delay):
                    if is_valid_time_window(measure_index) is False:
                        continue

                    print(n_lines_filled)

                    y[n_lines_filled] = resampled_swp_measures['data_ph'].iloc[measure_index + time_delay]

                    # swp id
                    X[n_lines_filled, 0] = id
                    # timestamp
                    X[n_lines_filled, 1] = resampled_swp_measures['resampled_timestamp'].iloc[
                        measure_index + time_delay]
                    # variables about swp
                    i_var = 2  # starts
                    for var in swp_explanatory_variables:
                        X[n_lines_filled, i_var] = swp_data[var]
                        i_var += 1

                    # extract Italy time
                    date = datetime.fromtimestamp(resampled_swp_measures['resampled_timestamp'].iloc[measure_index],
                                                  tz=pytz.timezone("Europe/Rome"))
                    day_year = date.day
                    sec_in_day = date.hour * 3600 + date.minute * 60 + date.second
                    X[n_lines_filled, 2 + len(swp_explanatory_variables)] = day_year
                    X[n_lines_filled, 2 + len(swp_explanatory_variables) + 1] = sec_in_day

                    # swp id
                    # timestamp
                    # swp var
                    # day year
                    # second day
                    # solar irradiance * time_delay
                    start = 2 + len(swp_explanatory_variables) + 2
                    for i_temp in range(time_horizon):
                        X[n_lines_filled, i_temp + start] = resampled_swp_measures['data_temperature'].iloc[
                            measure_index - time_horizon + i_temp]

                    start += time_horizon
                    for i_temp in range(time_horizon):
                        X[n_lines_filled, i_temp + start] = resampled_swp_measures['data_ph'].iloc[
                            measure_index - time_horizon + i_temp]

                    start += time_horizon
                    for i_temp in range(time_horizon):
                        X[n_lines_filled, i_temp + start] = resampled_swp_measures['data_conductivity'].iloc[
                            measure_index - time_horizon + i_temp]

                    start += time_horizon
                    for i_temp in range(time_horizon):
                        X[n_lines_filled, i_temp + start] = resampled_swp_measures['data_orp'].iloc[
                            measure_index - time_horizon + i_temp]

                    vnorm = np.array([0, 0, -1])  # plane pointing zenith
                    h = 0  # sea-level
                    lat = 42  # lat of Roma
                    for i in range(time_delay):
                        date = datetime.fromtimestamp(
                            resampled_swp_measures['resampled_timestamp'].iloc[measure_index + i])
                        X[n_lines_filled, start + time_horizon + i] = irradiance_on_plane(vnorm, h, date, lat)

                    # swp id
                    # timestamp
                    # swp var
                    # day year
                    # second day
                    # solar irradiance * time_delay
                    # (data_ph + data_orp + data_temperature) * time_horizon
                    for var in ["weather_temp"]:
                        for i in range(time_delay):
                            X[n_lines_filled, i_var + i] = resampled_swp_measures[var].iloc[measure_index + i]
                        i_var += time_delay

                    n_lines_filled += 1

            X_df = pd.DataFrame(data=X[0:n_lines_filled], columns=columns)
            y_df = pd.DataFrame(data=y[0:n_lines_filled], columns=['ph. t+' + str(time_delay)])

            return X_df, y_df

        X, y = build_X_y(event, swp, time_horizon=time_horizon, time_delay=time_delay)

        return X, y

    def arima(event_dict):
        swp_id = random.choice(list(event_dict.keys()))
        # for swp_id in event_dict:
        if True:
            measures = event_dict[swp_id]
            print(measures['data_temperature'])

            na_index = pd.isna(measures['data_temperature'])
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.tsa.arima.model import ARIMAResults

            train_set, test_set = time_series_split(measures, 0.9)

            print(len(measures))
            print(len(train_set))
            print(len(test_set))
            dates = [datetime.fromtimestamp(d) for d in train_set['resampled_timestamp']]
            order = (5, 1, 5)
            model = ARIMA(endog=train_set['data_temperature'], order=order, missing="drop")
            # model = SARIMAX(train_set['data_temperature'])
            model = model.fit()
            # print(model.summary())
            predictions = model.predict(start=len(train_set), end=len(train_set) + len(test_set) - 1, type="levels",
                                        exog=test_set['solar_irrandiance'])

            plt.scatter(measures.loc[na_index, 'resampled_timestamp'], np.full(np.sum(na_index), 1), s=1, c='red',
                        label="temperature measurements")
            plt.scatter(measures.loc[~na_index, 'resampled_timestamp'], measures.loc[~na_index, 'data_temperature'],
                        s=0.5, c='blue', label="missing measurements")

            plt.scatter(test_set['resampled_timestamp'], predictions, s=0.5, c='green', label="predictions")

            plt.plot(train_set['resampled_timestamp'], model.fittedvalues, markeredgewidth=0.5)
            plt.ylim((0, 50))
            plt.ylabel("Temperature in ° C")
            plt.xlabel("Timestamps")
            plt.title("ARIMA with (p,d,q)={}".format(order))
            plt.legend()
            plt.show(block=False)

            return

    def gaussian_process(event_dict):
        import pyflux as pf

        for swp_id in event_dict:
            measures = event_dict[swp_id]
            print(measures['data_temperature'])

            na_index = pd.isna(measures['data_temperature'])

            train_set, test_set = time_series_split(measures.loc[~na_index, :], 0.9)
            model = pf.GPNARX(pd.Series.to_numpy(train_set['data_temperature']), ar=4, kernel=pf.ARD)

            model.fit(method="BBVI")
            predictions = model.predict(steps=len(test_set))

            plt.scatter(measures.loc[na_index, 'resampled_timestamp'], np.full(np.sum(na_index), 1), s=1, c='red')
            plt.scatter(measures.loc[~na_index, 'resampled_timestamp'], measures.loc[~na_index, 'data_temperature'],
                        s=0.5, c='blue')

            plt.plot(test_set['resampled_timestamp'], predictions)
            plt.show(block=True)

            return

    def plot_feature_importances(feature_importances, columns):
        """feature_importances = []
        f = model.get_booster().get_score(importance_type='weight')
        max_f_index = max([int(x[1:]) for x in list(f.keys())])
        used_columns = []

        for k in range(max_f_index):
            key = 'f'+str(k)
            if key in f:
                feature_importances.append(f[key])
                used_columns.append(columns[k])"""

        n_features = len(feature_importances)

        # d = dict(zip(used_columns, feature_importances))
        d = dict(zip(columns, feature_importances))
        ss = sorted(d, key=d.get, reverse=True)
        top_names = ss[0:n_features]
        print(top_names)
        print(d)

        plt.figure(figsize=(5, 5))
        plt.title("Feature importances")
        plt.bar(range(n_features), [d[i] for i in top_names], color="r", align="center")
        plt.xlim(-1, n_features)
        plt.xticks(range(n_features), top_names, rotation=50, fontsize=7)

        plt.show()