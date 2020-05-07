import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from gluonts.dataset.common import ListDataset
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.core.component import validated
from gluonts.trainer import Trainer
from gluonts.support.util import copy_parameters
from gluonts.transform import ExpectedNumInstanceSampler, Transformation, InstanceSplitter
from mxnet.gluon import HybridBlock
from gluonts.dataset.field_names import FieldName
from gluonts.model.deepar import DeepAREstimator

"""
    Wrapper class around deepar model from gluonts.
    
    Roles:
        - Takes a dataset either windows/slices dataset (used by other predictors) or event dataframe and generate
          a gluonTS friendly format dataset.
        - Train the deepar gluonTS format IN THE SAME set up as other modules. We want to make online predictions 
          and measure the quality of these predictions.
        - Evaluate the predictions in a standardize format
        
    Author: Baptiste Debes ; b.debes@student.uliege.be
"""
class GluonTSWrapper:
    """
        Build time series from event measurements table. Time series are resampled to given period. Gaps are filled using
        given imputation method. Returns dataset ready for gluonts conversion.
    
        Inputs :
            - event: event dataframe 
            - imputation method: 'NA' for np.nan, 'mean' to use the mean of the time series, 'zero' to impute with 0s
            - period: resampling period AND normal behaviour sampling period 
        Returns:
            - list of dictionaries ; ready for gluonts conversion
    
    
        Author: Baptiste Debes ; b.debes@student.uliege.be
    """
    @staticmethod
    def event_to_time_series(event, imputation='NA', period=4320):
        # time conversion
        from datetime import datetime
        import pytz
        # solar irradiance estimation
        from solarpy import irradiance_on_plane
        # resampling
        from scipy.interpolate import CubicSpline

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
        def del_too_few_obs(event, min_obs):
            counts = event['swimming_pool_id'].value_counts()
            event_ids_to_keep = counts[counts > min_obs].index
            event = event.loc[event['swimming_pool_id'].isin(event_ids_to_keep)]
            event.reset_index(inplace=True)
            return event

        event = event.copy()
        event = event_del_miss(event)
        event = event_var_of_interest(event)
        event = del_too_few_obs(event, min_obs=50)

        # datatime to timestamp
        event['created'] = pd.Series((pd.DatetimeIndex(event['created']).asi8 / 10 ** 9).astype(np.int))

        event_explanatory_variables = ["weather_temp", "weather_humidity", "weather_pressure"]

        # columns of the final dataset
        def create_columns():
            columns = []
            columns += ["timestamp"]
            columns.append('solar_irradiance' + " t")
            columns.append("temp. t")
            for var in event_explanatory_variables:
                columns.append(var + " t")

            return columns

        columns = create_columns()

        def build_time_series(event, id):
            swp_measures = event.loc[event['swimming_pool_id'] == id]

            timestamps = swp_measures['created']
            timestamps_diffs = timestamps.diff()[1:]
            """
                Resample a time series: the time diff. between two observations is irregular
                    Effect: apply cublic spline interpolation on one time series. The time series is sliced into
                    valid sub-time-series on which a cublic spline model is fitted and then sampled.
            """
            n_max_new_samples = int(np.ceil((timestamps.iloc[-1] - timestamps.iloc[0]) / period))
            resampled_measures = np.empty((n_max_new_samples * 2, len(columns)))
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
                if timestamps_diffs.iloc[measure_index] > 10 * 3600 or timestamps_diffs.iloc[measure_index] < 0.05 * 3600:
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
                    series_model.append(CubicSpline(timestamps[start:end + 1], measures_array[start:end + 1, index_var]))

                interpolation_models_X.append(((start, end), series_model))

            """
                Loop through the models of each valid piece and sample at regular time interval.
                Output : re-sampled time series
            """
            imputations = []
            for var in range(1, len(columns)):
                series = resampled_measures[:, var]
                if imputation == 'NA':
                    imputations.append(np.nan)
                elif imputation == 'mean':
                    imputations.append(np.nanmean(series))
                elif imputation == 'zero':
                    imputations.append(0)

            imputations[1] = np.nan

            resampled_index = 0  # number of lines re-sampled
            last_end = None
            for time_range in interpolation_models_X:
                index_range = time_range[0]
                start = index_range[0]
                end = index_range[1]
                models = time_range[1]

                # if the time difference of missing values is large enough ; fill with na
                if last_end is not None and timestamps.iloc[start] - timestamps.iloc[last_end] > 6 * 3600:
                    times_fill_na = np.arange(timestamps.iloc[last_end] + period,
                                              timestamps.iloc[start] - period, period)
                    resampled_measures[resampled_index:resampled_index + len(times_fill_na), 0] = times_fill_na
                    resampled_measures[resampled_index:resampled_index + len(times_fill_na), 1:] = imputations
                    resampled_index += len(times_fill_na)

                last_end = end

                times = np.arange(timestamps.iloc[start], timestamps.iloc[end], period)
                resampled_measures[resampled_index:resampled_index + len(times), 0] = times

                """ Create time and irradiance features"""
                # Add solar irradiance at time t
                vnorm = np.array([0, 0, -1])  # plane pointing zenith
                h = 0  # sea-level
                lat = 42  # latitude of Roma (middle Italy, very approximative)
                var = 1
                for i in range(len(times)):
                    # extract Italy time
                    date = datetime.fromtimestamp(times[i], tz=pytz.timezone("Europe/Rome"))
                    resampled_measures[resampled_index + i, var] = irradiance_on_plane(vnorm, h, date, lat)

                var += 1

                for model in models:
                    interpolation = model(times)
                    resampled_measures[resampled_index:resampled_index + len(times), var] = interpolation

                    var += 1

                resampled_index += len(times)

            if resampled_index < 50:
                return None
            else:
                start_time = resampled_measures[0, 0]
                resampled_measures = pd.DataFrame(data=resampled_measures[0:resampled_index, 1:], columns=columns[1:])
                return {'swimming_pool_id': id, 'start_time': pd.to_datetime(start_time, unit='s'),
                        'time_series': resampled_measures}

        buffer = []
        swp_ids = list(event['swimming_pool_id'].unique())
        for i, id in enumerate(swp_ids):
            print("Pool {}/{}".format(i + 1, len(swp_ids)), end="\r")
            time_series = build_time_series(event, id)
            if time_series is not None:
                buffer.append(time_series)

        return buffer


    """
        Given a windows dataset returns corresponding time series. Allows to use the same dataset for all the methods 
        presented. Fill gaps (time diff > 4320s) with NAs or mean.
            Handles : temp. t, solar_irradiance t, weather_temp t, weather_humidity t, weather_pressure t
        Input
            - X: windows dataset
            - period: normal sampling period
            - imputation: either 'NA' or 'mean'
    
        Returns:
            - list of dictionaries [{'swimming_pool_id': ..., 'start_time': ..., 'time_series': ...} ... {}]
    
        Author: Baptiste Debes ; b.debes@student.uliege.be
    """
    @staticmethod
    def windows_dataset_to_time_series(X, period=4320, imputation='NA'):
        time_series_to_handle = ['temp. t', 'solar_irradiance t', 'weather_temp t', 'weather_humidity t',
                                 'weather_pressure t']

        ids = list(X['swimming_pool_id'].unique())

        buffers = []
        for n_covered, id in enumerate(ids):
            print("Pool {}/{}".format(n_covered + 1, len(ids)), end='\r')
            # subset for this id
            X_id = X[X['swimming_pool_id'] == id]
            timestamps = X_id['timestamp']

            # compute maximum number of measurement
            time_difference = timestamps.iloc[-1] - timestamps.iloc[0]
            n_measures = int(np.ceil(time_difference / period))
            # pre-allocate memory (optimization)
            buffer = np.zeros((n_measures * 2, len(time_series_to_handle)))  # *2 to be sure

            # values for imputation
            impute_values = []
            for series_name in time_series_to_handle:
                series = X_id[series_name]

                if imputation == 'NA':
                    impute_values.append(np.nan)
                elif imputation == 'mean':
                    impute_values.append(series.mean())

            diffs = np.diff(timestamps)

            # the two reading/writing heads does not move at the same pace
            last_buffer = 0  # index in the buffer
            last_original = 0  # index in the original data
            n_valid = 0  # number of valid measurements in the current slice
            for diff in diffs:
                # there is a cut
                if diff > period * 1.01:  # *1.01 just to be safe
                    buffer[last_buffer:last_buffer + n_valid + 1, :] = X_id[time_series_to_handle].values[
                                                                       last_original:last_original + n_valid + 1]
                    last_buffer += n_valid + 1  # move forward
                    last_original += n_valid + 1  # move forward
                    n_imputations = int(np.floor(diff / period))  # number of imputations to add into buffer
                    buffer[last_buffer:last_buffer + n_imputations, :] = np.full((n_imputations, len(impute_values)),
                                                                                 impute_values)
                    last_buffer += n_imputations  # move forward
                    n_valid = 0  # restart
                else:
                    n_valid += 1

            # there might be some remaining values at the end
            buffer[last_buffer:last_buffer + n_valid + 1, :] = X_id[time_series_to_handle].values[
                                                               last_original:last_original + n_valid + 1]
            last_buffer += n_valid + 1  # move forward

            buffer = buffer[0:last_buffer, :]
            buffer = pd.DataFrame(data=buffer, columns=time_series_to_handle)

            state_date_time = pd.to_datetime(timestamps.iloc[0], unit='s')
            buffers.append({'swimming_pool_id': id, 'start_time': state_date_time, 'time_series': buffer})

        return buffers

    """
        Transform the dataset to gluonTS compatible format.
        
        Inputs:
            - data: windows dataset or time series dataset
            - mode: either 'from_X' or 'from_time_series' ; 'from_x' reconstruct time series from windows dataset
                    'from_time_series' builds time series directly from measurements
            - period: normal sampling period (in seconds)
            - train_ratio: proportion of pools to be used as training pools
            - random_state: reproducibility seed for shuffling 
        Returns:
            - ...
        
        
        Author: Baptiste Debes ; b.debes@student.uliege.be
    """
    @staticmethod
    def data_set_to_gluonts_format(data, mode, swp, period=4320, train_ratio=0.9, random_state=0):
        freq = "{}H".format(period / 3600)
        cat_features = ['type', 'kind', 'location'] # cat features to be used
        real_features = ['solar_irradiance t', 'weather_temp t'] # time series to be added as exogenous variables

        """
            Retrieve categorical features of pools from windowed dataset encoded as dummies. 
            NB. NAs are considered as a value.
            
            Inputs:
                - X: windows: dataset
                - cat_features: categorical features to retrieve
            Returns:
                - dataframe with dummied cat features for each swimming_pool id
                
            
            Author: Baptiste Debes ; b.debes@student.uliege.be
        """
        def retrieve_one_hot_cat_feat(swp, cat_features):
            cat_ids = swp[['swimming_pool_id'] + cat_features]
            cat_ids = cat_ids.drop_duplicates(keep='first', inplace=False)
            cat_dummies = pd.get_dummies(cat_ids[cat_features], prefix=cat_features, dummy_na=True)
            cat_ids = pd.concat([cat_ids['swimming_pool_id'], cat_dummies], axis=1)
            return cat_ids

        # retrieve dummies for categorical features
        cat_features_dummies = retrieve_one_hot_cat_feat(swp, cat_features)

        print("Generating time series dataset from X")
        if mode == "from_X":
            time_series = GluonTSWrapper.windows_dataset_to_time_series(data, period=period, imputation='NA')
        elif mode == "from_time_series":
            time_series = data

        print("Generating final datasets")
        buffer_list_dataset = []
        for i, ts_dict in enumerate(time_series):
            print("Pool {}/{}".format(i+1, len(time_series)), end='\r')
            swp_id = ts_dict['swimming_pool_id']


            if swp_id not in list(swp['swimming_pool_id'].unique()):
                continue

            start_time = ts_dict['start_time']
            pool_series = ts_dict['time_series']
            target = pool_series['temp. t'].values
            start = pd.Timestamp(start_time, unit='s')
            feature_dynamic_real = pool_series[real_features].values
            feature_static_cat = cat_features_dummies[cat_features_dummies['swimming_pool_id'] == swp_id].values[0,1:]
            # features must be fed as lists
            temp_dict = {FieldName.ITEM_ID: swp_id,
                         FieldName.TARGET: target,
                         FieldName.START: start,
                         FieldName.FEAT_DYNAMIC_REAL: [feature_dynamic_real[:,i] for i in range(feature_dynamic_real.shape[1])],
                         FieldName.FEAT_STATIC_CAT: feature_static_cat
                        }

            buffer_list_dataset.append(temp_dict)

        # shuffle inplace
        random.Random(random_state).shuffle(buffer_list_dataset)

        # split train/test
        n_series_train = int(np.ceil(len(buffer_list_dataset)*train_ratio))
        train_series_list = buffer_list_dataset[0:n_series_train]
        test_series_list = buffer_list_dataset[n_series_train:]

        train_ds = ListDataset(train_series_list, freq=freq)
        test_ds = ListDataset(test_series_list, freq=freq)

        return train_ds, test_ds


    """
        Train deepar model.
        
        Inputs:
            - train_ds: train gluonTS dataset
            - context_length: how many observation in the past are used by the model
            - prediction_length: how many observation in the future are performed by the model
            - period: normal sampling period
            
        Returns: trained model
        
        Author: Baptiste Debes ; b.debes@student.uliege.be
    """
    @staticmethod
    def train_deepar(train_ds, context_length=10, prediction_length=20, period=4320, epochs=2):
        freq = "{}H".format(period / 3600)
        estimator = DeepAREstimator(
            prediction_length=prediction_length,
            context_length=context_length,
            freq=freq,
            num_cells=50,
            trainer=Trainer(ctx="gpu",
                            epochs=epochs,
                            learning_rate=1e-3,
                            hybridize=False,
                            num_batches_per_epoch=100,
                            batch_size=64
                            ),
            num_parallel_samples=500
        )
        predictor = estimator.train(train_ds)

        return predictor

    """
        Mimic online predictions. Builds artificial dataset of what would be known in online mode. Merge predictions
        and class them by time delay (delay in the future).
        
        Inputs:
            - model: trained deepar model
            - test_ds: test gluonTS dataset
            - context_length: how many observation in the past are used by the model
            - prediction_length: how many observation in the future are performed by the model
            - period: normal sampling period
            - alpha: low and high quantile will be {alpha;1-alpha}
            
        Returns:
            - dictionary of predictions whose keys are swp_id. Each item is a dict containing predictions and ground
              truth.         
    """
    @staticmethod
    def test_deepar(model, test_ds, context_length=10, prediction_length=20, period=4320, alpha=0.05):
        """
            We only make predictions for valid time windows: no nan in context_length pre-window and prediction_length post
            window

            Inputs:
                - target: the array to predict
                - index: current position (last observation)
                - context_length: number observations in the past (including index)
                - prediction_length: number observations in the future (including index)

            Returns:
                - True if no missing value (NaN) in the window ; False otherwise


            Author: Baptiste Debes ; b.debes@student.uliege.be
        """
        def is_valid_window(target, index, context_length, prediction_length):
            return np.sum(np.isnan(target[index - context_length + 1:index + prediction_length])) == 0


        # series and data for each pool
        list_data = test_ds.list_data

        ret = {}

        # loop through each pool of test set
        for j, data in enumerate(list_data):
            print("Pool {}/{}".format(j + 1, len(list_data)), end="\r\n")
            swp_id = data['item_id'] # swp_id
            start = data['start'] # start time of the time series
            target = data['target'] # variable to predict
            feat_dynamic_real = data['feat_dynamic_real']
            feat_static_cat = data['feat_static_cat']
            n_points = len(target)

            ret[swp_id] = {}
            ret[swp_id]['y_truth'] = [] # ground truth to be predicted
            ret[swp_id]['y_truth_timestamp'] = [] # timestamp for each ground truth element

            # low and high predicitions for each time delay
            for i in range(1, prediction_length + 1):
                ret[swp_id]['low temp. t+{}'.format(i)] = []
                ret[swp_id]['high temp. t+{}'.format(i)] = []
                ret[swp_id]['median temp. t+{}'.format(i)] = []
                ret[swp_id]['mean temp. t+{}'.format(i)] = []

            for i in range(context_length, n_points - prediction_length):
                print("{} / {}".format(i + 1, n_points - prediction_length - context_length), end="\r")

                # if observation window has not enough valid observation in the past and the future one discards it
                # requires each point to be predicted to be surrounded by context_length points in the past
                # and prediction_length points in the future
                if not (is_valid_window(target, index=i, context_length=context_length,
                                        prediction_length=prediction_length)):
                    continue # go to next observation

                # ground truth
                ret[swp_id]['y_truth'].append(target[i])
                ret[swp_id]['y_truth_timestamp'].append(start + i * pd.Timedelta(seconds=period))

                # slice observation and temporal features to only what is necessary
                target_to_use = target[i - context_length:i + 1]
                feat_dynamic_real_to_use = [fdr[i - context_length:i + 1] for fdr in feat_dynamic_real]
                temp_dict = {FieldName.ITEM_ID: swp_id,
                             FieldName.TARGET: target_to_use,
                             FieldName.START: start,
                             FieldName.FEAT_DYNAMIC_REAL: feat_dynamic_real_to_use,
                             FieldName.FEAT_STATIC_CAT: feat_static_cat
                             }
                # create artificial dataset (to mimic online prediction)
                temp_test_ds = ListDataset([temp_dict], freq="1.2H")
                predictions = next(model.predict(temp_test_ds))

                low_predictions = predictions.quantile(q=alpha)
                high_predictions = predictions.quantile(q=1 - alpha)
                median_predictions = predictions.quantile(q=0.5)
                mean_predictions = predictions.mean

                for i in range(1, prediction_length + 1):
                    ret[swp_id]['low temp. t+{}'.format(i)].append(low_predictions[i - 1])
                    ret[swp_id]['high temp. t+{}'.format(i)].append(high_predictions[i - 1])
                    ret[swp_id]['median temp. t+{}'.format(i)].append(median_predictions[i - 1])
                    ret[swp_id]['mean temp. t+{}'.format(i)].append(mean_predictions[i - 1])

            # convert all lists to numpy arrays
            ret[swp_id]['y_truth'] = np.array(ret[swp_id]['y_truth'])
            ret[swp_id]['y_truth_timestamp'] = np.array(ret[swp_id]['y_truth_timestamp'])
            for i in range(1, prediction_length + 1):
                ret[swp_id]['low temp. t+{}'.format(i)] = np.array(ret[swp_id]['low temp. t+{}'.format(i)])
                ret[swp_id]['high temp. t+{}'.format(i)] = np.array(ret[swp_id]['high temp. t+{}'.format(i)])
                ret[swp_id]['median temp. t+{}'.format(i)] = np.array(ret[swp_id]['median temp. t+{}'.format(i)])
                ret[swp_id]['mean temp. t+{}'.format(i)] = np.array(ret[swp_id]['mean temp. t+{}'.format(i)])

        return ret

    """
        Convert the results from deepar predictions to standardize evaluation metrics.
        
        Inputs:
            - train_ret: return dict from training
            - prediction_length: number of observations to predict in the future (this is multi-target like)
            - verbose: 0 for silence, 1 for infos
            
        Returns:
        
            - metrics dict 
            
        Author: Baptiste Debes ; b.debes@student.uliege.be
    """

    @staticmethod
    def evaluate_deepar(test_ret, prediction_length=20, alpha=0.05, verbose=1):
        from question1 import std_absolute_error, full_quantile_loss, coverage, average_quantile_span
        from sklearn.metrics import mean_absolute_error

        ret = {}

        # concatenate ground truths
        y_test = []
        for swp_id in test_ret:
            y_test.append(test_ret[swp_id]['y_truth'])

        y_test = np.concatenate(y_test)

        for time_delay_i in range(1, prediction_length+1):
            key = 't+'+str(time_delay_i)

            ret[key] = {}

            # concatenate the predictions of each pool for a given time delay
            y_pred_low_i = []
            y_pred_high_i = []
            y_pred_median_i = []
            y_pred_mean_i = []

            for swp_id in test_ret:
                y_pred_low_i.append(test_ret[swp_id]['low temp. t+{}'.format(time_delay_i)])
                y_pred_high_i.append(test_ret[swp_id]['high temp. t+{}'.format(time_delay_i)])
                y_pred_median_i.append(test_ret[swp_id]['median temp. t+{}'.format(time_delay_i)])
                y_pred_mean_i.append(test_ret[swp_id]['mean temp. t+{}'.format(time_delay_i)])

            y_pred_low_i = np.concatenate(y_pred_low_i)
            y_pred_high_i = np.concatenate(y_pred_high_i)
            y_pred_median_i = np.concatenate(y_pred_median_i)

            # save metrics
            ret[key]['mean_absolute_error'] = mean_absolute_error(y_pred=y_pred_median_i, y_true=y_test)
            ret[key]['std_absolute_error'] = std_absolute_error(y_true=y_test, y_pred=y_pred_median_i)
            ret[key]['quantile_loss'] = full_quantile_loss(y_test, y_pred_low_i, y_pred_high_i, alpha=alpha)
            ret[key]['coverage'] = coverage(y_test, y_pred_low_i, y_pred_high_i)
            ret[key]['average_quantile_span'] = average_quantile_span(y_pred_low_i, y_pred_high_i)

            if verbose > 0:
                print("Time delay {}".format(time_delay_i))
                print("Mean absolute error {}".format(ret[key]['mean_absolute_error']))
                print("Std absolute error {}".format(ret[key]['std_absolute_error']))
                print("Quantile loss {}".format(ret[key]['quantile_loss']))
                print("Coverage {}".format(ret[key]['coverage']))
                print("Average quantile span {}".format(ret[key]['average_quantile_span']))

        return ret


    """
        Simply compare through OUR metrics the performance of the estimator for different epochs.
        NB 1. Results are not cross-validated. The computation is already VERY VERY long.
        NB 2. The same model is not retrained (not available with gluonTS). It is retrained from scratch each time.
        
        Inputs:
            - max_epochs: max number of epochs
            - n_estimations: number of different train/test partition to compute on
            - time_series: time_series dataset
            - swp: swp dataframe
            
        Author: Baptiste Debes ; b.debes@student.uliege.be
    """
    @staticmethod
    def compare_epochs(max_epochs, n_estimations, time_series, swp, context_length=10, prediction_length=20, period=4320, alpha=0.05):

        buffer = [] # list of results
        for n in range(n_estimations):
            print("{}th estimator over {}".format(n+1, n_estimations))
            ret = {}

            # generate a different train/test ds
            train_ds, test_ds = GluonTSWrapper.data_set_to_gluonts_format(data=time_series, mode='from_time_series', swp=swp, random_state=n)
            for epochs in range(1, max_epochs+1):
                print("Epochs from 1 to {}/{}".format(epochs, max_epochs))

                model = GluonTSWrapper.train_deepar(train_ds=train_ds,
                                                    context_length=context_length,
                                                    prediction_length=prediction_length,
                                                    period=period,
                                                    epochs=epochs)

                model_preds = GluonTSWrapper.test_deepar(model=model,
                                                         test_ds=test_ds,
                                                         context_length=context_length,
                                                         prediction_length=prediction_length,
                                                         period=period,
                                                         alpha=alpha)

                evaluation = GluonTSWrapper.evaluate_deepar(test_ret=model_preds,
                                                            prediction_length=prediction_length,
                                                            alpha=alpha,
                                                            verbose=1)

                ret[epochs] = evaluation

            buffer.append(ret)

        return buffer

    @staticmethod
    def plot_compare_epochs(epochs_comparison_ret_buffer, alpha=0.05, prediction_length=20):

        # fill in the results
        mean_absolute_errors = {}
        average_quantile_spans = {}
        coverages = {}
        # loop through buffer
        for epochs_comparison_ret in epochs_comparison_ret_buffer:
            # loop through epochs
            for epochs in epochs_comparison_ret:
                epoch_key = '{} epoch(s)'.format(epochs)
                if epoch_key not in mean_absolute_errors:
                    mean_absolute_errors[epoch_key] = {}

                if epoch_key not in average_quantile_spans:
                    average_quantile_spans[epoch_key] = {}

                if epoch_key not in coverages:
                    coverages[epoch_key] = {}

                # loop through time delays
                for delay in range(1, prediction_length+1):
                    delay_key = 't+{}'.format(delay)

                    if delay_key not in mean_absolute_errors[epoch_key]:
                        mean_absolute_errors[epoch_key][delay_key] = []

                    mean_absolute_errors[epoch_key][delay_key].append(epochs_comparison_ret[epochs]['t+{}'.format(delay)]['mean_absolute_error'])

                    if delay_key not in average_quantile_spans[epoch_key]:
                        average_quantile_spans[epoch_key][delay_key] = []

                    average_quantile_spans[epoch_key][delay_key].append(epochs_comparison_ret[epochs]['t+{}'.format(delay)]['average_quantile_span'])

                    if delay_key not in coverages[epoch_key]:
                        coverages[epoch_key][delay_key] = []

                    coverages[epoch_key][delay_key].append(epochs_comparison_ret[epochs]['t+{}'.format(delay)]['coverage'])

        # compute average and std
        mean_absolute_errors_std = {}
        average_quantile_spans_std = {}
        coverages_std = {}

        for epochs in epochs_comparison_ret_buffer[0]:
            epoch_key = '{} epoch(s)'.format(epochs)

            if epoch_key not in mean_absolute_errors_std:
                mean_absolute_errors_std[epoch_key] = {}

            if epoch_key not in average_quantile_spans_std:
                average_quantile_spans_std[epoch_key] = {}

            if epoch_key not in coverages_std:
                coverages_std[epoch_key] = {}

            for delay in range(1, prediction_length+1):
                delay_key = 't+{}'.format(delay)

                mean_absolute_errors[epoch_key][delay_key] = np.array(mean_absolute_errors[epoch_key][delay_key])
                mean_absolute_errors_std[epoch_key][delay_key] = mean_absolute_errors[epoch_key][delay_key].std()
                mean_absolute_errors[epoch_key][delay_key] = mean_absolute_errors[epoch_key][delay_key].mean()

                average_quantile_spans[epoch_key][delay_key] = np.array(average_quantile_spans[epoch_key][delay_key])
                average_quantile_spans_std[epoch_key][delay_key] = average_quantile_spans[epoch_key][delay_key].std()
                average_quantile_spans[epoch_key][delay_key] = average_quantile_spans[epoch_key][delay_key].mean()

                coverages[epoch_key][delay_key] = np.array(coverages[epoch_key][delay_key])
                coverages_std[epoch_key][delay_key] = coverages[epoch_key][delay_key].std()
                coverages[epoch_key][delay_key] = coverages[epoch_key][delay_key].mean()


        # means
        mean_absolute_errors = pd.DataFrame(mean_absolute_errors)
        average_quantile_spans = pd.DataFrame(average_quantile_spans)
        coverages = pd.DataFrame(coverages)

        # stds
        mean_absolute_errors_std = pd.DataFrame(mean_absolute_errors_std)
        average_quantile_spans_std = pd.DataFrame(average_quantile_spans_std)
        coverages_std = pd.DataFrame(coverages_std)

        if len(epochs_comparison_ret_buffer) == 1:
            mean_absolute_errors.plot(kind='bar', alpha=0.5, error_kw=dict(ecolor='k'))
        else:
            mean_absolute_errors.plot(kind='bar', yerr=mean_absolute_errors_std.values.T, alpha=0.5, error_kw=dict(ecolor='k'))

        plt.xlabel("Time targets")
        plt.ylabel("Mean absolute error")
        plt.title("Test set mean absolute error for time targets vs number of epochs (lower is better)")
        plt.show()

        if len(epochs_comparison_ret_buffer) == 1:
            average_quantile_spans.plot(kind='bar', alpha=0.5, error_kw=dict(ecolor='k'))
        else:
            average_quantile_spans.plot(kind='bar', yerr=average_quantile_spans_std.values.T, alpha=0.5, error_kw=dict(ecolor='k'))

        plt.xlabel("Time targets")
        plt.ylabel("Average quantile span")
        plt.title("Test set average quantile span for time targets vs number of epochs (lower is subjectively better)")
        plt.show()

        if len(epochs_comparison_ret_buffer) == 1:
            coverages.plot(kind='bar', alpha=0.5, error_kw=dict(ecolor='k'))
        else:
            coverages.plot(kind='bar', yerr=coverages_std.values.T, alpha=0.5, error_kw=dict(ecolor='k'))

        plt.xlabel("Time targets")
        plt.ylabel("Coverage")
        plt.title("Test set coverage for time targets vs number of epochs (closer to {:.2f} is better)".format(1-2*alpha))
        plt.show()



    @staticmethod
    def save_time_series(time_series, file_name):
        import pickle
        with open(file_name+'.pickle', 'wb') as handle:
            pickle.dump(time_series, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def read_time_series(file_name):
        import pickle
        with open(file_name+'.pickle', 'rb') as handle:
            b = pickle.load(handle)

        return b

    @staticmethod
    def save_test_predictions(time_series, file_name):
        import pickle
        with open(file_name + '.pickle', 'wb') as handle:
            pickle.dump(time_series, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def read_test_predictions(file_name):
        import pickle
        with open(file_name + '.pickle', 'rb') as handle:
            b = pickle.load(handle)

        return b

    @staticmethod
    def save_model(model, file_name):
        from pathlib import Path
        model.serialize(Path(file_name))

    @staticmethod
    def read_model(file_name):
        from pathlib import Path
        from gluonts.model.predictor import Predictor
        predictor_deserialized = Predictor.deserialize(Path(file_name))
        return predictor_deserialized



def plt_res(ret):
    len_list = [len(ret[key]['y_truth']) for key in ret]
    ordered_indexes = np.argsort(len_list)

    key = list(ret.keys())[ordered_indexes[-1]]
    item = ret[key]
    print(item)
    timestamps = item['y_truth_timestamp']
    y_truth = item['y_truth']


    low = item['low temp. t+2']
    high = item['high temp. t+2']
    #mean = item['median temp. t+2']
    mean = (low + high)/2
    print(low)

    plt.fill_between(timestamps, low, high,
                     color='green', alpha=0.3, interpolate=False, step='pre', label='Quantiles space of t+2')
    plt.step(timestamps, mean, color='green', label='Main predictor of t+2')

    low = item['low temp. t+20']
    high = item['high temp. t+20']
    #mean = item['median temp. t+20']
    mean = (low + high)/2

    plt.fill_between(timestamps, low, high,
                     color='orange', alpha=0.3, interpolate=False, step='pre', label='Quantiles space of t+20')
    plt.step(timestamps, mean, color='orange', label='Main predictor of t+20')

    low = item['low temp. t+30']
    high = item['high temp. t+30']
    #mean = item['median temp. t+30']
    mean = (low + high)/2

    plt.fill_between(timestamps, low, high,
                     color='red', alpha=0.3, interpolate=False, step='pre', label='Quantiles space of t+30')
    plt.step(timestamps, mean, color='red', label='Main predictor of t+30')

    plt.scatter(timestamps, y_truth)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel("Datetime", fontsize=15)
    plt.title("Temperature predictions with 0.05-0.95 quantiles", fontsize=22)
    plt.ylabel("Temperature in ° C", fontsize=15)
    plt.show()



