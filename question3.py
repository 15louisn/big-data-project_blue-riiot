import pandas as pd
import numpy as np
import matplotlib

# resampling
from scipy.interpolate import CubicSpline

# time conversion
from datetime import datetime
import pytz

# solar irradiance estimation
from solarpy import irradiance_on_plane

matplotlib.use('TkAgg')
import question1

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

"""
    From event dataframe create an intermediate dataframe to be used for cluster dataset construction.
    Resampled the time series to 4320 s per period. Fill the gaps with NAs.

    NB. This code is very ugly GluonTSWrapper.event_to_time_series does exactly the same thing but with
    a different output format. (not enough time to merge the codes)

    Inputs:
        - event: dataframe

    Returns
        - Resample dataframes with gaps filled by NAs


    Author: Baptiste Debes ; b.debes@student.uliege.be
"""
def time_series_temperature_build_db(event):
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

    """
        Resample the time series and fill gaps with NAs => output is much bigger than input event.
    """
    def resample_and_na_fill(event, swp_id):
        # fixed because one case (this one)
        resampling_T = 4320

        swp_measures = question1.extract_pool(event, swp_id=swp_id)
        swp_measures = swp_measures[
            ['created', 'data_temperature', 'data_conductivity', 'data_orp', 'data_ph', 'weather_temp']]
        swp_measures = swp_measures.dropna()
        timestamps = pd.Series((pd.DatetimeIndex(swp_measures['created']).asi8 / 10 ** 9).astype(np.int))
        timestamps_diffs = timestamps.diff()[1:]

        # if too few observations
        if len(timestamps) < 50:
            return None

        # max number of sample for this pool after resampling and imputation
        n_max_new_samples = len(
            np.arange(timestamps.iloc[0], timestamps.iloc[-1], resampling_T)) * 2  # *2 to spare time debug
        print(n_max_new_samples)

        resampled_measures = pd.DataFrame(
            data=np.empty((n_max_new_samples, 2 + 1 + 1 + 1 + 2)),
            columns=['resampled_timestamp', 'solar_irrandiance', 'data_temperature', 'data_conductivity', 'data_orp',
                     'data_ph', 'weather_temp'])

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
        measures_array = swp_measures[
            ['data_temperature', 'data_conductivity', 'data_orp', 'data_ph', 'weather_temp']].to_numpy()
        for time_range in time_series_ranges:
            start = time_range[0]
            end = time_range[1]

            series_model = []
            var_to_resample = [0, 1, 2, 3, 4]
            for index_var in var_to_resample:
                series_model.append(
                    CubicSpline(timestamps[start:end + 1], measures_array[start:end + 1, index_var]))

            interpolation_models_X.append(((start, end), series_model))

        vnorm = np.array([0, 0, -1])  # plane pointing zenith
        h = 0  # sea-level
        lat = 42  # lat of Roma
        resampled_index = 0  # number of lines resampled
        last_end = None
        for time_range in interpolation_models_X:
            index_range = time_range[0]
            start = index_range[0]
            end = index_range[1]
            models = time_range[1]

            # if the time difference of missing values is large enough ; fill with na
            if last_end is not None and timestamps.iloc[start] - timestamps.iloc[last_end] > 6 * 3600:
                times_fill_na = np.arange(timestamps.iloc[last_end] + resampling_T,
                                          timestamps.iloc[start] - resampling_T, resampling_T)
                resampled_measures.iloc[resampled_index:resampled_index + len(times_fill_na), 0] = times_fill_na
                resampled_measures.iloc[resampled_index:resampled_index + len(times_fill_na), 2] = float('nan')
                resampled_measures.iloc[resampled_index:resampled_index + len(times_fill_na), 3] = float('nan')
                resampled_measures.iloc[resampled_index:resampled_index + len(times_fill_na), 4] = float('nan')
                resampled_measures.iloc[resampled_index:resampled_index + len(times_fill_na), 5] = float('nan')
                resampled_measures.iloc[resampled_index:resampled_index + len(times_fill_na), 6] = float('nan')
                for time in times_fill_na:
                    date = datetime.fromtimestamp(time)
                    resampled_measures.iloc[t, 1] = irradiance_on_plane(vnorm, h, date, lat)
                resampled_index += len(times_fill_na)

            last_end = end

            times = np.arange(timestamps.iloc[start], timestamps.iloc[end], resampling_T)
            resampled_measures.iloc[resampled_index:resampled_index + len(times), 0] = times
            var = 2
            for model in models:
                interpolation = model(times)
                resampled_measures.iloc[resampled_index:resampled_index + len(times), var] = interpolation

                var += 1

            for t in range(resampled_index, resampled_index + len(times)):
                date = datetime.fromtimestamp(resampled_measures["resampled_timestamp"].iloc[t])
                resampled_measures.iloc[t, 1] = irradiance_on_plane(vnorm, h, date, lat)

            resampled_index += len(times)

        return resampled_measures.iloc[0:resampled_index, :]

    buffer = []
    i = 0
    swp_ids = pd.unique(list(event['swimming_pool_id']))
    for swp_id in swp_ids:
        print("{}/{}".format(i, len(swp_ids)))
        i += 1
        res = resample_and_na_fill(event, swp_id)
        if res is not None: # pool is valid
            res['swimming_pool_id'] = np.full(len(res), swp_id)

            buffer.append(res)


    buffer = pd.concat(buffer)

    return buffer

def save_time_series(time_series, file_path):
    time_series.to_pickle("{}.pkl".format(file_path))

def read_time_series(file_path):
    return pd.read_pickle("{}.pkl".format(file_path))


"""
    From event, swp and time series dataframe, summarizes pool's time series to some numbers.
    This is useful for clusterization and missing value imputation. The time series dataframe to be used is one generated
    by time_series_temperature_build_db (above).

    Inputs:
        - event, swp, time_series dataframes
    Returns:
        - cluster_dataset dataframe: one row per swp_id, variables summerazing what is known about the pool


    Author: Baptiste Debes ; b.debes@student.uliege.be
"""
def create_cluster_dataset(event, swp, time_series):
    def del_too_few_obs(time_series, swp, min_obs):
        value_counts = time_series["swimming_pool_id"].value_counts()
        not_too_few_obs_ids = value_counts[value_counts > min_obs].index

        swp_id_swp = swp["swimming_pool_id"]
        intersection = set(swp_id_swp).intersection(set(not_too_few_obs_ids))

        time_series = time_series.loc[time_series['swimming_pool_id'].isin(intersection)]
        swp = swp.loc[swp['swimming_pool_id'].isin(intersection)]

        time_series.reset_index(inplace=True)
        swp.reset_index(inplace=True)

        return time_series, swp

    swp = swp.dropna(subset=['volume_capacity'])

    event, swp = del_too_few_obs(event, swp, min_obs=200)
    swp_id_swp = swp["swimming_pool_id"]
    swp_id_resampled_measurements = time_series["swimming_pool_id"]
    intersection = set(swp_id_swp).intersection(set(swp_id_resampled_measurements))

    # clusterization variables
    # average temperature
    # temperature variance
    # average diff
    # pool volume
    # ratio of missing values

    columns = [
        'swimming_pool_id',
        'average_temp',
        'std_temp',
        'absolute_average_local_diff_temp',
        'quantile_25_temp',
        'quantile_75_temp',
        'autocorr_1_temp',
        'autocorr_4_temp',
        'autocorr_7_temp',
        'autocorr_10_temp',
        'norm_fourier_0_temp',
        'norm_fourier_1_temp',
        'norm_fourier_2_temp',
        'norm_fourier_3_temp',
        'norm_fourier_4_temp',
        'average_cond',
        'std_cond',
        'absolute_average_local_diff_cond',
        'quantile_25_cond',
        'quantile_75_cond',
        'autocorr_1_cond',
        'autocorr_4_cond',
        'autocorr_7_cond',
        'autocorr_10_cond',
        'norm_fourier_0_cond',
        'norm_fourier_1_cond',
        'norm_fourier_2_cond',
        'norm_fourier_3_cond',
        'norm_fourier_4_cond',
        'average_ph',
        'std_ph',
        'absolute_average_local_diff_ph',
        'quantile_25_ph',
        'quantile_75_ph',
        'autocorr_1_ph',
        'autocorr_4_ph',
        'autocorr_7_ph',
        'autocorr_10_ph',
        'norm_fourier_0_ph',
        'norm_fourier_1_ph',
        'norm_fourier_2_ph',
        'norm_fourier_3_ph',
        'norm_fourier_4_ph',
        'average_orp',
        'std_orp',
        'absolute_average_local_diff_orp',
        'quantile_25_orp',
        'quantile_75_orp',
        'autocorr_1_orp',
        'autocorr_4_orp',
        'autocorr_7_orp',
        'autocorr_10_orp',
        'norm_fourier_0_orp',
        'norm_fourier_1_orp',
        'norm_fourier_2_orp',
        'norm_fourier_3_orp',
        'norm_fourier_4_orp',
        'n_measurements',
        # rajouter partial autocorrelation et autocorrelation jusque 3
        # 'skew_temp',
        # 'kurtosis_temp',
        # 'average_abs_diff',
        'volume_capacity',
        'ratio_missing_measurement',
        'location',
        'type',
        'kind',
        'equipment_heatings']

    df = pd.DataFrame(index=list(range(len(intersection))), columns=columns)
    df['swimming_pool_id'] = intersection
    i = 0
    for swp_id in intersection:
        print("Pool {}/{}".format(i, len(intersection)))
        measurements = time_series.loc[time_series['swimming_pool_id'] == swp_id]
        swp_specs = swp.loc[swp['swimming_pool_id'] == swp_id]
        selection = df['swimming_pool_id'] == swp_id

        # temp
        df.loc[selection, 'average_temp'] = measurements['data_temperature'].mean()
        df.loc[selection, 'std_temp'] = measurements['data_temperature'].std()
        exponential_average = measurements['data_temperature'].ewm(alpha=0.2).mean()
        df.loc[selection, 'absolute_average_local_diff_temp'] = (
                    measurements['data_temperature'] - exponential_average).abs().mean()
        df.loc[selection, 'quantile_25_temp'] = measurements['data_temperature'].quantile(0.25)
        df.loc[selection, 'quantile_75_temp'] = measurements['data_temperature'].quantile(0.75)
        autocorr = acf(measurements['data_temperature'], missing='drop', nlags=10)
        df.loc[selection, 'autocorr_1_temp'] = autocorr[1]
        df.loc[selection, 'autocorr_4_temp'] = autocorr[4]
        df.loc[selection, 'autocorr_7_temp'] = autocorr[7]
        df.loc[selection, 'autocorr_10_temp'] = autocorr[10]
        # get signal
        signal = measurements['data_temperature'].values
        # impute missing by the mean
        signal[np.isnan(signal)] = np.mean(signal[np.invert(np.isnan(signal))])
        fourier = np.absolute(np.fft.fft(signal))
        df.loc[selection, 'norm_fourier_0_temp'] = fourier[0]
        df.loc[selection, 'norm_fourier_1_temp'] = fourier[1]
        df.loc[selection, 'norm_fourier_2_temp'] = fourier[2]
        df.loc[selection, 'norm_fourier_3_temp'] = fourier[3]
        df.loc[selection, 'norm_fourier_4_temp'] = fourier[4]

        # cond
        df.loc[selection, 'average_cond'] = measurements['data_conductivity'].mean()
        df.loc[selection, 'std_cond'] = measurements['data_conductivity'].std()
        exponential_average = measurements['data_conductivity'].ewm(alpha=0.2).mean()
        df.loc[selection, 'absolute_average_local_diff_cond'] = (
                    measurements['data_conductivity'] - exponential_average).abs().mean()
        df.loc[selection, 'quantile_25_cond'] = measurements['data_conductivity'].quantile(0.25)
        df.loc[selection, 'quantile_75_cond'] = measurements['data_conductivity'].quantile(0.75)
        autocorr = acf(measurements['data_conductivity'], missing='drop', nlags=10)
        df.loc[selection, 'autocorr_1_cond'] = autocorr[1]
        df.loc[selection, 'autocorr_4_cond'] = autocorr[4]
        df.loc[selection, 'autocorr_7_cond'] = autocorr[7]
        df.loc[selection, 'autocorr_10_cond'] = autocorr[10]
        # get signal
        signal = measurements['data_conductivity'].values
        # impute missing by the mean
        signal[np.isnan(signal)] = np.mean(signal[np.invert(np.isnan(signal))])
        fourier = np.absolute(np.fft.fft(signal))
        df.loc[selection, 'norm_fourier_0_cond'] = fourier[0]
        df.loc[selection, 'norm_fourier_1_cond'] = fourier[1]
        df.loc[selection, 'norm_fourier_2_cond'] = fourier[2]
        df.loc[selection, 'norm_fourier_3_cond'] = fourier[3]
        df.loc[selection, 'norm_fourier_4_cond'] = fourier[4]

        # ph
        df.loc[selection, 'average_ph'] = measurements['data_ph'].mean()
        df.loc[selection, 'std_ph'] = measurements['data_ph'].std()
        exponential_average = measurements['data_ph'].ewm(alpha=0.2).mean()
        df.loc[selection, 'absolute_average_local_diff_ph'] = (
                    measurements['data_ph'] - exponential_average).abs().mean()
        df.loc[selection, 'quantile_25_ph'] = measurements['data_ph'].quantile(0.25)
        df.loc[selection, 'quantile_75_ph'] = measurements['data_ph'].quantile(0.75)
        autocorr = acf(measurements['data_ph'], missing='drop', nlags=10)
        df.loc[selection, 'autocorr_1_ph'] = autocorr[1]
        df.loc[selection, 'autocorr_4_ph'] = autocorr[4]
        df.loc[selection, 'autocorr_7_ph'] = autocorr[7]
        df.loc[selection, 'autocorr_10_ph'] = autocorr[10]
        # get signal
        signal = measurements['data_ph'].values
        # impute missing by the mean
        signal[np.isnan(signal)] = np.mean(signal[np.invert(np.isnan(signal))])
        fourier = np.absolute(np.fft.fft(signal))
        df.loc[selection, 'norm_fourier_0_ph'] = fourier[0]
        df.loc[selection, 'norm_fourier_1_ph'] = fourier[1]
        df.loc[selection, 'norm_fourier_2_ph'] = fourier[2]
        df.loc[selection, 'norm_fourier_3_ph'] = fourier[3]
        df.loc[selection, 'norm_fourier_4_ph'] = fourier[4]

        # orp
        df.loc[selection, 'average_orp'] = measurements['data_orp'].mean()
        df.loc[selection, 'std_orp'] = measurements['data_orp'].std()
        exponential_average = measurements['data_orp'].ewm(alpha=0.2).mean()
        df.loc[selection, 'absolute_average_local_diff_orp'] = (
                    measurements['data_orp'] - exponential_average).abs().mean()
        df.loc[selection, 'quantile_25_orp'] = measurements['data_orp'].quantile(0.25)
        df.loc[selection, 'quantile_75_orp'] = measurements['data_orp'].quantile(0.75)
        autocorr = acf(measurements['data_orp'], missing='drop', nlags=10)
        df.loc[selection, 'autocorr_1_orp'] = autocorr[1]
        df.loc[selection, 'autocorr_4_orp'] = autocorr[4]
        df.loc[selection, 'autocorr_7_orp'] = autocorr[7]
        df.loc[selection, 'autocorr_10_orp'] = autocorr[10]
        # get signal
        signal = measurements['data_orp'].values
        # impute missing by the mean
        signal[np.isnan(signal)] = np.mean(signal[np.invert(np.isnan(signal))])
        fourier = np.absolute(np.fft.fft(signal))
        df.loc[selection, 'norm_fourier_0_orp'] = fourier[0]
        df.loc[selection, 'norm_fourier_1_orp'] = fourier[1]
        df.loc[selection, 'norm_fourier_2_orp'] = fourier[2]
        df.loc[selection, 'norm_fourier_3_orp'] = fourier[3]
        df.loc[selection, 'norm_fourier_4_orp'] = fourier[4]

        df.loc[selection, 'n_measurements'] = sum(measurements['data_temperature'].isnull())

        # df.loc[selection, 'skew_temp'] = measurements['data_temperature'].skew()
        # df.loc[selection, 'kurtosis_temp'] = measurements['data_temperature'].kurtosis()
        # df.loc[selection, 'average_abs_diff'] = measurements['data_temperature'].diff().abs().mean()
        df.loc[selection, 'volume_capacity'] = swp_specs['volume_capacity'].values[0]
        df.loc[selection, 'ratio_missing_measurement'] = sum(measurements['data_temperature'].isnull()) / len(
            measurements)
        df.loc[selection, 'location'] = swp_specs['location'].values[0]
        df.loc[selection, 'type'] = swp_specs['type'].values[0]
        df.loc[selection, 'kind'] = swp_specs['kind'].values[0]
        df.loc[selection, 'equipment_heatings'] = swp_specs['equipment_heatings'].values[0]

        i += 1

    print(df)
    return df


def save_cluster_dataset(cluster_dataset, file_path):
    cluster_dataset.to_pickle("{}.pkl".format(file_path))


def read_cluster_dataset(file_path):
    return pd.read_pickle("{}.pkl".format(file_path))


def read_strip():
    data = pd.read_csv("strip.csv")

    return data

def read_swp_guidance():
    data = pd.read_csv("swpguidance.csv")

    return data

def scree_plot(explained_variance_ratio):
    plt.figure()
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'ro-', linewidth=2)
    plt.xlabel("PCs")
    plt.ylabel("Explained variance ratio")
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.show()

"""
    Performs dimension reduction and clusterization on the dataset. Clusters are shown on the reduced space.


    Author: Baptiste Debes ; b.debes@student.uliege.be
            Louis Nelissen ; louis.nelissen@student.uliege.be
"""
def show_clusters(cluster_dataset):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cluster_dataset = cluster_dataset.__deepcopy__()
    var_to_use = [
        'average_temp',
        'std_temp',
        'absolute_average_local_diff_temp',
        'quantile_25_temp',
        'quantile_75_temp',
        'autocorr_1_temp',
        'autocorr_4_temp',
        # 'autocorr_7_temp',
        # 'autocorr_10_temp',
        # 'norm_fourier_0_temp',
        # 'norm_fourier_1_temp',
        # 'norm_fourier_2_temp',
        # 'norm_fourier_3_temp',
        # 'norm_fourier_4_temp',
        'average_cond',
        'std_cond',
        'absolute_average_local_diff_cond',
        'quantile_25_cond',
        'quantile_75_cond',
        'autocorr_1_cond',
        'autocorr_4_cond',
        # 'autocorr_7_cond',
        # 'autocorr_10_cond',
        # 'norm_fourier_0_cond',
        # 'norm_fourier_1_cond',
        # 'norm_fourier_2_cond',
        # 'norm_fourier_3_cond',
        # 'norm_fourier_4_cond',
        'average_ph',
        'std_ph',
        'absolute_average_local_diff_ph',
        'quantile_25_ph',
        'quantile_75_ph',
        'autocorr_1_ph',
        'autocorr_4_ph',
        # 'autocorr_7_ph',
        # 'autocorr_10_ph',
        # 'norm_fourier_0_ph',
        # 'norm_fourier_1_ph',
        # 'norm_fourier_2_ph',
        # 'norm_fourier_3_ph',
        # 'norm_fourier_4_ph',
        'average_orp',
        'std_orp',
        'absolute_average_local_diff_orp',
        'quantile_25_orp',
        'quantile_75_orp',
        'autocorr_1_orp',
        'autocorr_4_orp',
        # 'autocorr_7_orp',
        # 'autocorr_10_orp',
        # 'norm_fourier_0_orp',
        # 'norm_fourier_1_orp',
        # 'norm_fourier_2_orp',
        # 'norm_fourier_3_orp',
        # 'norm_fourier_4_orp',
        'volume_capacity',
        'ratio_missing_measurement'
    ]

    cat_col = ['location',
               'kind',
               'equipment_heatings',
               'type']
    # cluster_dataset = cluster_dataset.dropna(subset=cat_col)
    for col in cat_col:
        cluster_dataset[col] = cluster_dataset[col].factorize()[0]

    X_cat = cluster_dataset.loc[:, cat_col]
    X_quant = cluster_dataset.loc[:, var_to_use]

    X_quant = X_quant.dropna()
    # print(X_quant)
    # X_cat = StandardScaler().fit_transform(X_cat)
    X_quant = StandardScaler().fit_transform(X_quant)
    # X_emb = X
    X = np.hstack([X_cat, X_quant])
    # X = X_quant
    X_emb = TSNE(n_components=2, perplexity=10, n_jobs=1, random_state=0).fit_transform(X)
    # pca = PCA(n_components=5)
    # X_emb = pca.fit_transform(X)
    # explained_variance_ratio = pca.explained_variance_ratio_
    # scree_plot(explained_variance_ratio)

    # color = cluster_dataset.iloc[:, -4].values
    # print(color)
    # location
    # color[color == 'Indoor'] = 'red'
    # color[color == 'Outdoor'] = 'blue'

    # kind
    # color[color == 'Spa'] = 'red'
    # color[color == 'SwimmingPool'] = 'blue'

    # type
    # color[color == 'Buried'] = 'red'
    # color[color == 'Aboveground'] = 'blue'

    # equipment_heatings
    # color[color == 1] = 'red'
    # color[color == 0] = 'blue'

    # color[pd.isnull(color)] = 'grey'

    # from sklearn.cluster import KMeans
    # n_clusters = 7
    # clusters = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    # from sklearn.cluster import OPTICS
    # clusters = OPTICS(min_samples=5).fit(X)
    from sklearn.cluster import AgglomerativeClustering
    clusters = AgglomerativeClustering(n_clusters=7).fit(X)
    n_clusters = len(np.unique(clusters.labels_))

    plot_name = 'Clusters/tSNE_'+ AgglomerativeClustering.__name__+'.pdf'

    def rgb2hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    np.random.seed(0)
    base_colors = [rgb2hex(list(np.random.choice(range(256), size=3))) for _ in range(n_clusters)]
    color = []

    for i in range(len(clusters.labels_)):
        color.append(base_colors[clusters.labels_[i]])
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(cluster_dataset.loc[clusters.labels_ == 1, :])

    from matplotlib.lines import Line2D
    legend_elements = []

    for i in range(0,len(base_colors)):
        legend_elements.append( Line2D([0], [0], marker='.', color='w', label=i,
                          markerfacecolor=base_colors[i], markersize=15))

    points = []
    n = 5

    def onpick(event):
        if len(points) < n:
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            point = tuple(zip(xdata[ind], ydata[ind]))
            points.append(point)
            print('onpick point:', point)
        else:
            print('already have {} points'.format(len(points)))

    def get_data_on_click(event):
        ind = event.ind
        print("ID: {} x-value: {} y-value: {}".format(ind, np.take(x1, ind), np.take(x2, ind)))
        # add/change some code here

    """fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(X_emb[:,0], X_emb[:,1], c=color)
    fig.canvas.mpl_connect('pick_event', onpick)"""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = X_emb[:, 0]
    x2 = X_emb[:, 1]
    line = ax.scatter(x1, x2, c=color, picker=True, s=6)
    fig.canvas.mpl_connect('pick_event', get_data_on_click)
    plt.title("Clusters shown on 2D tSNE variables")
    plt.xlabel("tSNE 1")
    plt.ylabel("tSNE 2")
    ax.legend(handles=legend_elements) #labels = ['0','1','2','3','4','5','6'])
    plt.savefig(plot_name,dpi=fig.dpi)
    plt.clf()

"""
    Performs dimension reduction and clusterization on the dataset.
    Produces graphs to help analyse and differentiate the clusters.


    Author: Louis Nelissen ; louis.nelissen@student.uliege.be
            Baptiste Debes ; b.debes@student.uliege.be
"""
def analyse_clusters(cluster_dataset):
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    # scaler = StandardScaler()
    cluster_dataset = cluster_dataset.__deepcopy__()
    var_to_use = [
        'average_temp',
        'std_temp',
        'absolute_average_local_diff_temp',
        'quantile_25_temp',
        'quantile_75_temp',
        'autocorr_1_temp',
        'autocorr_4_temp',
        # 'autocorr_7_temp',
        # 'autocorr_10_temp',
        # 'norm_fourier_0_temp',
        # 'norm_fourier_1_temp',
        # 'norm_fourier_2_temp',
        # 'norm_fourier_3_temp',
        # 'norm_fourier_4_temp',
        'average_cond',
        'std_cond',
        'absolute_average_local_diff_cond',
        'quantile_25_cond',
        'quantile_75_cond',
        'autocorr_1_cond',
        'autocorr_4_cond',
        # 'autocorr_7_cond',
        # 'autocorr_10_cond',
        # 'norm_fourier_0_cond',
        # 'norm_fourier_1_cond',
        # 'norm_fourier_2_cond',
        # 'norm_fourier_3_cond',
        # 'norm_fourier_4_cond',
        'average_ph',
        'std_ph',
        'absolute_average_local_diff_ph',
        'quantile_25_ph',
        'quantile_75_ph',
        'autocorr_1_ph',
        'autocorr_4_ph',
        # 'autocorr_7_ph',
        # 'autocorr_10_ph',
        # 'norm_fourier_0_ph',
        # 'norm_fourier_1_ph',
        # 'norm_fourier_2_ph',
        # 'norm_fourier_3_ph',
        # 'norm_fourier_4_ph',
        'average_orp',
        'std_orp',
        'absolute_average_local_diff_orp',
        'quantile_25_orp',
        'quantile_75_orp',
        'autocorr_1_orp',
        'autocorr_4_orp',
        # 'autocorr_7_orp',
        # 'autocorr_10_orp',
        # 'norm_fourier_0_orp',
        # 'norm_fourier_1_orp',
        # 'norm_fourier_2_orp',
        # 'norm_fourier_3_orp',
        # 'norm_fourier_4_orp',
        'volume_capacity',
        'ratio_missing_measurement'
    ]

    cat_col = ['location',
               'kind',
               'equipment_heatings',
               'type']
    # cluster_dataset = cluster_dataset.dropna(subset=cat_col)
    for col in cat_col:
        cluster_dataset[col] = cluster_dataset[col].factorize()[0]

    X_cat = cluster_dataset.loc[:, cat_col]
    X_quant = cluster_dataset.loc[:, var_to_use]

    X_quant = X_quant.dropna()
    # X_cat = StandardScaler().fit_transform(X_cat)
    X_cat = pd.DataFrame(scaler.fit_transform(X_cat), columns=X_cat.columns)

    # X_quant = StandardScaler().fit_transform(X_quant)
    X_quant = pd.DataFrame(scaler.fit_transform(X_quant), columns=X_quant.columns)
    # X_emb = X
    X_df = pd.concat([X_cat, X_quant], axis=1)
    X_np = np.hstack([X_cat, X_quant])

    # from sklearn.cluster import KMeans
    # n_clusters = 7
    # clusters = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    # from sklearn.cluster import OPTICS
    # clusters = OPTICS(min_samples=5).fit(X)
    from sklearn.cluster import AgglomerativeClustering
    clusters = AgglomerativeClustering(n_clusters=7).fit(X_np)
    n_clusters = len(np.unique(clusters.labels_))

    import seaborn as sns

    X_df['clusters'] = clusters.labels_
    # Does this work with qualitative data? Use median?
    X_df_mean = (X_df.groupby('clusters').mean())
    results = pd.DataFrame(columns=['Variable', 'Var'])

    for column in X_df_mean.columns[1:]:
        results.loc[len(results), :] = [column, np.var(X_df_mean[column])]

    selected_columns = list(results.sort_values('Var', ascending=False,
                        ).head(7).Variable.values) + ['clusters']
    tidy = X_df[selected_columns].melt(id_vars='clusters')
    sns.barplot(x='clusters', y='value', hue='variable', data=tidy)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    # plt.savefig('Clusters/Clusters_value_comp.pdf')
    plt.clf()

    # Plot the relative importance of each variable in the clusterization
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = X_df.iloc[:,:-1], X_df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # Parameters can be modified for better accuracy
    clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy of classifier: ", accuracy_score(y_test, y_pred))
    data = np.array([clf.feature_importances_, X.columns]).T
    columns = pd.DataFrame(data, columns=['Importance', 'Feature']).sort_values("Importance", ascending=False).head(7)
    tidy = X_df[list(columns.Feature.values)+['clusters']].melt(id_vars='clusters')
    print(columns)
    sns.barplot(x='clusters', y='value', hue='variable', data=tidy)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    # plt.savefig('Clusters/Clusters_feature_imp.pdf')
    plt.clf()


"""
    From summary/clusterization variables, plot a correlation matrix between all the variables.
    The subset of variables is hardcoded.

    Inputs:
        - cluster_dataset: generated by create_cluseter_dataset (above)


    Author: Baptiste Debes ; b.debes@student.uliege.be
            Louis Nelissen ; louis.nelissen@student.uliege.be
"""
def correlation_matrix(cluster_dataset):
    cluster_dataset = cluster_dataset.__deepcopy__()
    cat_col = ['location',
               'kind',
               'equipment_heatings',
               'type']
    cluster_dataset = cluster_dataset.dropna(subset=cat_col)
    for col in cat_col:
        cluster_dataset[col] = cluster_dataset[col].factorize()[0]

    var_to_use = [
        'average_temp',
        'std_temp',
        'absolute_average_local_diff_temp',
        'quantile_25_temp',
        'quantile_75_temp',
        'autocorr_1_temp',
        'autocorr_4_temp',
        'autocorr_7_temp',
        'autocorr_10_temp',
        'average_cond',
        'std_cond',
        'absolute_average_local_diff_cond',
        'quantile_25_cond',
        'quantile_75_cond',
        'autocorr_1_cond',
        'autocorr_4_cond',
        'autocorr_7_cond',
        'autocorr_10_cond',
        'average_ph',
        'std_ph',
        'absolute_average_local_diff_ph',
        'quantile_25_ph',
        'quantile_75_ph',
        'autocorr_1_ph',
        'autocorr_4_ph',
        'autocorr_7_ph',
        'autocorr_10_ph',
        'average_orp',
        'std_orp',
        'absolute_average_local_diff_orp',
        'quantile_25_orp',
        'quantile_75_orp',
        'autocorr_1_orp',
        'autocorr_4_orp',
        'autocorr_7_orp',
        'autocorr_10_orp',
        'volume_capacity',
        'location',
        'kind',
        'equipment_heatings',
        'type']

    from sklearn.preprocessing import StandardScaler
    cluster_dataset_ = StandardScaler().fit_transform(cluster_dataset[var_to_use].values)
    cluster_dataset = pd.DataFrame(data=cluster_dataset_, columns=var_to_use)

    fig = plt.figure()
    import seaborn as sns

    ax = fig.add_subplot(111)
    # mat = ax.matshow(cluster_dataset.corr())
    ax = sns.heatmap(
                cluster_dataset.corr(),
                vmin=-1, vmax=1, center=0,
                cmap=sns.diverging_palette(20, 220, n=200),
                square=True)
    # fig.colorbar(mat, ax=ax)
    ticks = np.arange(0, len(cluster_dataset.columns), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(cluster_dataset.columns)
    ax.set_xticklabels(cluster_dataset.columns, rotation=90)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    # plt.show()
    plt.tight_layout()
    plt.savefig('Clusters/Corr_plot.pdf', dpi=fig.dpi)

"""
    From times series and user preferences, compute how far are the ph, orp and temperature of the pool from
    the users' ideals. The scores are mean absolute differences.

    Inputs:
        - time_series: generated by time_series_temperature_build_db (above)


    Returns:
        - scores dataframe with one row per swp_id


    Author: Baptiste Debes ; b.debes@student.uliege.be
"""
def derive_problematic_measure(time_series):
    swp_big = pd.read_csv("swp.csv", low_memory=False)

    # take only pool for which required pref. are defined
    customer_pref_ids = swp_big['swimming_pool_id'].loc[
                        (~swp_big['custom_mr_orp_ideal'].isna()) &\
                        (~swp_big['custom_mr_ph_ideal'].isna()) &\
                        (~swp_big['custom_mr_temperature_ideal'].isna())
                        ]

    columns = ['swimming_pool_id', 'temp_problem_score', 'ph_problem_score', 'orp_problem_score']
    problem_scores = pd.DataFrame(columns=columns)
    j = 0
    for id in customer_pref_ids:
        measures = time_series[time_series['swimming_pool_id'] == id]
        if len(measures) == 0:
            continue

        temp_measures = measures['data_temperature'].dropna().values
        orp_measures = measures['data_orp'].dropna().values
        ph_measures = measures['data_ph'].dropna().values

        swp_row = swp_big[swp_big['swimming_pool_id'] == id]
        temp_ideal = swp_row['custom_mr_temperature_ideal'].values
        orp_ideal = swp_row['custom_mr_orp_ideal'].values
        ph_ideal = swp_row['custom_mr_ph_ideal'].values

        temp_problem_score = np.mean(np.abs(temp_measures-temp_ideal))
        orp_problem_score = np.mean(np.abs(orp_measures-orp_ideal))
        ph_problem_score = np.mean(np.abs(ph_measures-ph_ideal))
        problem_scores.loc[j] = [id, temp_problem_score, ph_problem_score, orp_problem_score]
        j += 1


    return problem_scores


"""
    From times series and user preferences, compute how far are the ph, orp and temperature of the pool from
    the users' min/max preferences. Look whether the measure is out of range and if so, compute by how much, then
    the average.

    Inputs:
        - time_series: generated by time_series_temperature_build_db (above)

    Returns:
        - scores dataframe with one row per swp_id


    Author: Baptiste Debes ; b.debes@student.uliege.be
            Louis Nelissen ; louis.nelissen@student.uliege.be
"""
def derive_damage_measure(time_series):
    swp_big = pd.read_csv("swp.csv", low_memory=False)

    # take only pool for which required pref. are defined
    customer_pref_ids = swp_big['swimming_pool_id'].loc[
         (~swp_big['custom_mr_orp_min'].isna()) &\
         (~swp_big['custom_mr_ph_min'].isna()) &\
         (~swp_big['custom_mr_temperature_min'].isna()) & \
         (~swp_big['custom_mr_orp_max'].isna()) & \
         (~swp_big['custom_mr_ph_max'].isna()) & \
         (~swp_big['custom_mr_temperature_max'].isna())
        ]

    columns = ['swimming_pool_id', 'temp_damage_score', 'ph_damage_score', 'orp_damage_score']
    damage_scores = pd.DataFrame(columns=columns)
    j = 0
    for id in customer_pref_ids:
        measures = time_series[time_series['swimming_pool_id'] == id]
        if len(measures) == 0:
            continue

        temp_measures = measures['data_temperature'].dropna().values
        orp_measures = measures['data_orp'].dropna().values
        ph_measures = measures['data_ph'].dropna().values

        swp_row = swp_big[swp_big['swimming_pool_id'] == id]

        temp_min = swp_row['custom_mr_temperature_ideal'].values
        orp_min = swp_row['custom_mr_orp_ideal'].values
        ph_min = swp_row['custom_mr_ph_ideal'].values

        temp_max = swp_row['custom_mr_temperature_ideal'].values
        orp_max = swp_row['custom_mr_orp_ideal'].values
        ph_max = swp_row['custom_mr_ph_ideal'].values

        temp_damage_score = np.mean([int(x < temp_min)*np.abs(x - temp_min) +
                                int(x > temp_max)*np.abs(x - temp_max)
                                for x in temp_measures])


        ph_damage_score = np.mean([int(x < ph_min)*np.abs(x - ph_min) +
                                int(x > ph_max)*np.abs(x - ph_max)
                                for x in ph_measures])

        orp_damage_score = np.mean([int(x < orp_min)*np.abs(x - orp_min) +
                                int(x > orp_max)*np.abs(x - orp_max)
                                for x in orp_measures])

        damage_scores.loc[j] = [id, temp_damage_score, ph_damage_score, orp_damage_score]
        j += 1

    return damage_scores

"""
    Plot correlation matrix of either problematic scores or damage scores with some variables of the cluster_dataset

    Inputs:
        - scores: either damage_scores or problematic_scores
        - cluster_dataset: generated above
        - mode: either "problem" or "damage"


    Author: Baptiste Debes ; b.debes@student.uliege.be
"""
def correlation_new_measure(scores, cluster_dataset, mode):

    cluster_dataset = cluster_dataset.__deepcopy__()

    cat_col = ['location',
               'kind',
               'equipment_heatings',
               'type']
    cluster_dataset = cluster_dataset.dropna(subset=cat_col)
    # cluster_dataset = cluster_dataset.dropna(subset=cat_col)
    for col in cat_col:
        cluster_dataset[col] = cluster_dataset[col].factorize()[0].astype(float)


    scores_ids = set(scores['swimming_pool_id'].unique())
    cluster_dataset_ids = set(cluster_dataset['swimming_pool_id'].unique())

    common_ids = scores_ids.intersection(cluster_dataset_ids)

    var_to_use = [  'swimming_pool_id',
                    'average_temp',
                    'std_temp',
                    'average_cond',
                    'std_cond',
                    'average_ph',
                    'std_ph',
                    'average_orp',
                    'std_orp',
                    'location',
                    'type',
                    'kind',
                    'equipment_heatings'
                 ]

    if mode == "problem":
        new_data_set = pd.DataFrame(columns=var_to_use+['temp_problem_score', 'ph_problem_score', 'orp_problem_score'])
    elif mode == "damage":
        new_data_set = pd.DataFrame(columns=var_to_use+['temp_damage_score', 'ph_damage_score', 'orp_damage_score'])

    j = 0
    for id in common_ids:
        cluster_var = cluster_dataset[cluster_dataset['swimming_pool_id'] == id][var_to_use[1:]].values
        scores_id = scores[scores['swimming_pool_id'] == id].iloc[0,1:].values
        new_data_set.loc[j] = [id] + cluster_var[0].tolist() + scores_id.tolist()
        j += 1

    new_data_set = new_data_set.iloc[:, 1:]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    mat = ax.matshow(new_data_set.corr().iloc[-3:,:])
    fig.colorbar(mat, ax=ax)
    ax.set_xticks(np.arange(0, len(new_data_set.columns), 1))
    ax.set_yticks(np.arange(0, 3, 1))
    ax.set_yticklabels(new_data_set.columns[-3:])
    ax.set_xticklabels(new_data_set.columns, rotation=90)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    # plt.show()
    plt.tight_layout()
    plt.savefig("Clusters/Corr_plot_"+mode+".pdf")

    return new_data_set

def correlation_between_measures(scores1, scores2):

    scores1_ids = set(scores1['swimming_pool_id'].unique())
    scores2_ids = set(scores2['swimming_pool_id'].unique())

    common_ids = scores1_ids.intersection(scores2_ids)

    new_data_set = pd.DataFrame(columns=['temp_problem_score', 'ph_problem_score', 'orp_problem_score']+ ['temp_damage_score', 'ph_damage_score', 'orp_damage_score'])

    j = 0
    for id in common_ids:
        scores1_id = scores1[scores1['swimming_pool_id'] == id].iloc[0,1:].values
        scores2_id = scores2[scores2['swimming_pool_id'] == id].iloc[0,1:].values
        print(scores1_id, scores2_id)
        new_data_set.loc[j] = [id] + scores1_id.tolist() + scores2_id.tolist()
        j += 1

    new_data_set = new_data_set.iloc[:, 1:]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    mat = ax.matshow(new_data_set.corr().iloc[-3:,:])
    fig.colorbar(mat, ax=ax)
    ax.set_xticks(np.arange(0, len(new_data_set.columns), 1))
    ax.set_yticks(np.arange(0, 3, 1))
    ax.set_yticklabels(new_data_set.columns[-3:])
    ax.set_xticklabels(new_data_set.columns, rotation=90)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    # plt.show()
    plt.tight_layout()
    plt.savefig("Clusters/Corr_plot_mutual.pdf")

    return new_data_set
def compare_model_based_strip(strip, X, y, measurement_type="ph"):
    X_ids = list(X['swimming_pool_id'].unique())
    # extract pools in strip from the training set
    strip_cleaned = extract_strip_measurements(strip)
    strip_ids = list(strip_cleaned['swimming_pool_id'].unique())
    train_ids = set(X_ids).difference(set(strip_ids))
    test_ids = set(X_ids).intersection(set(strip_ids))

    train_indexes = X['swimming_pool_id'].isin(train_ids)
    test_indexes = X['swimming_pool_id'].isin(test_ids)

    # lgbm
    _, _, X_test, y_test,y_pred_low, y_pred_high  = question1.lightGBM_regression(X,
                                                                        y, time_delay=20,
                                                                        train_index=train_indexes,
                                                                        test_index=test_indexes,
                                                                        kind=measurement_type)

    y_pred_mean = (y_pred_low + y_pred_high)/2

    number_of_valid_samples = 0
    number_of_samples_in_bounds = 0
    for id in test_ids:
        X_times = X_test[X_test[:, 0] == id, 1]
        y_test_id = y_test[X_test[:, 0] == id]
        y_pred_low_id = y_pred_low[X_test[:, 0] == id]
        y_pred_high_id = y_pred_high[X_test[:, 0] == id]
        y_pred_mean_id = y_pred_mean[X_test[:, 0] == id]
        strip_measurements_id = strip_cleaned[strip_cleaned['swimming_pool_id'] == id]
        strip_time = strip_measurements_id['measurement_time']

        if measurement_type == "ph":
            strip_measure = strip_measurements_id['measurement_ph']
        elif measurement_type == "temp":
            strip_measure = strip_measurements_id['measurement_temperature']

        for j in range(len(strip_measure)):
            measure = strip_measure.iloc[j]
            if np.isnan(measure) == True:
                continue


            time = strip_time.iloc[j]
            if time < X_times[0] or time > X_times[-1]:
                continue

            left_index = np.argmax(X_times >= time) - 1 # index of the last smaller time
            right_index = np.argmax(X_times >= time) # index of the first larger time



            X_time_left = X_times[left_index] # time of left sample
            X_time_right = X_times[right_index] # time of right sample

            # left or right blue measurement is too far to be valid w.r.t the strip measurement
            if abs(X_time_left - time) > 2*3600 or abs(X_time_right - time) > 2 * 3600:
                continue

            number_of_valid_samples += 1


            left_low = y_pred_low_id[left_index]
            left_high = y_pred_high_id[left_index]
            right_low = y_pred_low_id[right_index]
            right_high = y_pred_high_id[right_index]

            # linear interpolation
            slope_low = (left_low - right_low)/(X_time_left - X_time_right)
            low_interpolated = left_low + slope_low * (time - X_time_left)
            slope_high = (left_high - right_high)/(X_time_left - X_time_right)
            high_interpolated = left_high + slope_high * (time - X_time_left)

            print("{} ; {} ; {}".format(y_test_id[left_index], measure, y_test_id[right_index]))

            if low_interpolated <= measure and measure <= high_interpolated:
                number_of_samples_in_bounds += 1

    print("{}/{}".format(number_of_samples_in_bounds, number_of_valid_samples))

def extract_strip_measurements(strip):
    ret = pd.DataFrame(columns=["swimming_pool_id", "measurement_time", "measurement_temperature", "measurement_ph"])
    ret['swimming_pool_id'] = strip['swimming_pool_id']
    ret['measurement_time'] = pd.Series((pd.DatetimeIndex(strip['created']).asi8 / 10 ** 9).astype(np.int))
    ret['measurement_temperature'] = strip['temperature']
    ret['measurement_ph'] = strip['ph']

    return ret

def plot_strip_vs_time_series(X, y, strip):
    X_ids = list(X['swimming_pool_id'].unique())
    # extract pools in strip from the training set
    strip_cleaned = extract_strip_measurements(strip)
    strip_cleaned = strip_cleaned.dropna(subset=['measurement_ph'])
    strip_ids = list(strip_cleaned['swimming_pool_id'].unique())

    common_ids = set(strip_ids).intersection(set(X_ids))

    for id in common_ids:
        measurements = y[X['swimming_pool_id']==id]
        times = X[X['swimming_pool_id']==id]['timestamp']
        strip_measurements_id = strip_cleaned[strip_cleaned['swimming_pool_id'] == id]
        strip_time = strip_measurements_id['measurement_time']
        strip_measure = strip_measurements_id['measurement_ph']


        if len(strip_measure) > 20:
            print(strip_measure)

            def rgb2hex(rgb):
                return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

            color = rgb2hex(list(np.random.choice(range(256), size=3)))
            plt.plot(pd.to_datetime(times, unit='s'), measurements, c=color, label="device measurement")
            plt.scatter(pd.to_datetime(strip_time, unit='s'), strip_measure, c=color, label="strip measurement")
            plt.ylabel("pH")
            plt.xlabel("Time")
            plt.legend()
            plt.title("Device's probe measurement vs paper strip measurement")

"""
    Split train/test set for missing values imputation. Split is performed on pools directly.

    Inputs:
        - X/y the datasets
        - test_size: test ratio
    Returns
        - X_train, X_test, y_train, y_test sets


    Author: Baptiste Debes ; b.debes@student.uliege.be
"""
def split_train(X, y, test_size=0.2):
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    X, y = shuffle(X, y, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0)

    return X_train, X_test, y_train, y_test

def integer_encode(X, columns):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[columns] = X[columns].apply(lambda col: le.fit_transform(col))
        return X

def binarize_equipment(serie):
    for i,v in serie.items():
        # print(v)
        if v == '[\"None\"]':
            serie[i] = 'No'
        elif isinstance(v,str):
            serie[i] = 'Yes'

    return serie

"""
    Demonstration modules. Taking the cluster dataset, one takes the pool for which the variables of interest are available.
    Then one divides the dataset into train set and test set. The purpose is to predict the location, type, kind (classification)
    or even volumne capacity (regression) of the pool on the test set. The accuracies are relevant to assess the
    potential of imputation on really missing values.


    Author: Baptiste Debes ; b.debes@student.uliege.be
            Louis Nelissen ; louis.nelissen@student.uliege.be
"""
class FillingMissing:

    @staticmethod
    def fill_volume_capacity(cluster_dataset):
        cluster_dataset = cluster_dataset.__deepcopy__()
        categorical_columns = ['type',
                               'kind',
                               'location',
                               'equipment_heatings']
        # integer encoding
        cluster_dataset[categorical_columns] = cluster_dataset[categorical_columns].astype('category')
        for col in categorical_columns:
            cluster_dataset[col] = cluster_dataset[col].cat.codes

        valid_pools = cluster_dataset.dropna(subset=['volume_capacity'])

        explanatory_variables = ['average_temp',
                                 'std_temp',
                                 #'absolute_average_local_diff_temp',
                                 'quantile_25_temp',
                                 'quantile_75_temp',
                                 #'autocorr_1_temp',
                                 #'autocorr_4_temp',
                                 #'autocorr_7_temp',
                                 # 'autocorr_10_temp',
                                 'average_cond',
                                 'std_cond',
                                 #'absolute_average_local_diff_cond',
                                 'quantile_25_cond',
                                 'quantile_75_cond',
                                 #'autocorr_1_cond',
                                 #'autocorr_4_cond',
                                 #'autocorr_7_cond',
                                 # 'autocorr_10_cond',
                                 'average_ph',
                                 'std_ph',
                                 #'absolute_average_local_diff_ph',
                                 'quantile_25_ph',
                                 'quantile_75_ph',
                                 #'autocorr_1_ph',
                                 #'autocorr_4_ph',
                                 #'autocorr_7_ph',
                                 # 'autocorr_10_ph',
                                 'average_orp',
                                 'std_orp',
                                 #'absolute_average_local_diff_orp',
                                 'quantile_25_orp',
                                 'quantile_75_orp',
                                 #'autocorr_1_orp',
                                 #'autocorr_4_orp',
                                 #'autocorr_7_orp',
                                 # 'autocorr_10_orp',
                                 # 'n_measurements',
                                 # 'ratio_missing_measurement',
                                 'type',
                                 'kind',
                                 'location',
                                 'equipment_heatings']

        X = valid_pools[explanatory_variables]
        y = valid_pools['volume_capacity']

        X_train, X_test, y_train, y_test = split_train(X.values, y.values, test_size=0.2)
        import lightgbm as lgb

        classifier_low = lgb.LGBMRegressor(num_leaves=2,
                                        max_depth=-1,
                                        alpha=0.05,
                                        n_estimators=10,
                                        learning_rate=0.25,
                                        boosting_type="gbdt",
                                        objective="quantile",
                                        random_state=0
                                        )
        classifier_high= lgb.LGBMRegressor(num_leaves=2,
                                        max_depth=-1,
                                        alpha=0.95,
                                        n_estimators=10,
                                        learning_rate=0.25,
                                        boosting_type="gbdt",
                                        objective="quantile",
                                        random_state=0
                                        )

        index_categorical_features = [idx for idx, element in enumerate(explanatory_variables) if
                                      element in categorical_columns]

        classifier_low.fit(X_train, y=y_train, categorical_feature=index_categorical_features)
        classifier_high.fit(X_train, y=y_train, categorical_feature=index_categorical_features)

        y_pred_low = classifier_low.predict(X_test, num_iteration=classifier_low.best_iteration_)
        y_pred_high = classifier_high.predict(X_test, num_iteration=classifier_high.best_iteration_)

        from sklearn.metrics import mean_squared_error

        for i in range(len(y_test)):
            print("{} in [{} ; {}]".format(y_test[i], y_pred_low[i], y_pred_high[i]))

        print("RMSE: {}".format(mean_squared_error(y_test, (y_pred_low+y_pred_high)/2)**0.5))
        print("Quantile loss {}".format(question1.full_quantile_loss(y_test, y_pred_low, y_pred_high, alpha=0.05)))
        print("Coverage {}".format(question1.coverage(y_test, y_pred_low, y_pred_high)))

    @staticmethod
    def fill_location(cluster_dataset):
        cluster_dataset = cluster_dataset.__deepcopy__()
        categorical_columns = ['type',
                               'kind',
                               'equipment_heatings']
        # integer encoding
        cluster_dataset['equipment_heatings'] = binarize_equipment(cluster_dataset['equipment_heatings'])
        # print(cluster_dataset['equipment_heatings'].value_counts())
        cluster_dataset[categorical_columns] = cluster_dataset[categorical_columns].astype('category')
        for col in categorical_columns:
            cluster_dataset[col] = cluster_dataset[col].cat.codes

        valid_pools = cluster_dataset.dropna(subset=['location'])
        valid_pools['location'] = valid_pools['location'].astype('category')
        valid_pools['location'] = valid_pools['location'].cat.codes

        invalid_pools = cluster_dataset[cluster_dataset['location'].isnull()]

        explanatory_variables = [  'average_temp',
                                   'std_temp',
                                   'absolute_average_local_diff_temp',
                                   'quantile_25_temp',
                                   'quantile_75_temp',
                                   'autocorr_1_temp',
                                   'autocorr_4_temp',
                                   'autocorr_7_temp',
                                   'autocorr_10_temp',
                                   'average_cond',
                                   'std_cond',
                                   'absolute_average_local_diff_cond',
                                   'quantile_25_cond',
                                   'quantile_75_cond',
                                   'autocorr_1_cond',
                                   'autocorr_4_cond',
                                   'autocorr_7_cond',
                                   'autocorr_10_cond',
                                   'autocorr_10_cond',
                                   'average_ph',
                                   'std_ph',
                                   'absolute_average_local_diff_ph',
                                   'quantile_25_ph',
                                   'quantile_75_ph',
                                   'autocorr_1_ph',
                                   'autocorr_4_ph',
                                   'autocorr_7_ph',
                                   'autocorr_10_ph',
                                   'average_orp',
                                   'std_orp',
                                   'absolute_average_local_diff_orp',
                                   'quantile_25_orp',
                                   'quantile_75_orp',
                                   'autocorr_1_orp',
                                   'autocorr_4_orp',
                                   'autocorr_7_orp',
                                   'autocorr_10_orp',
                                   'n_measurements',
                                   'volume_capacity',
                                   'ratio_missing_measurement',
                                   'type',
                                   'kind',
                                   'equipment_heatings']

        X = valid_pools[explanatory_variables]
        y = valid_pools['location']

        X_train, X_test, y_train, y_test = split_train(X.values, y.values, test_size=0.2)
        import lightgbm as lgb
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_ = compute_class_weight('balanced', np.unique(y_train), y_train)
        class_weights = dict(zip(np.unique(y_train), class_weights_))

        classifier = lgb.LGBMClassifier(num_leaves=5,
                                        max_depth=-1,
                                        n_estimators=100,
                                        learning_rate=0.2,
                                        boosting_type="dart",
                                        objective="cross_entropy",
                                        random_state=0,
                                        class_weight=class_weights
                                        )

        index_categorical_features = [idx for idx, element in enumerate(explanatory_variables) if element in categorical_columns]
        classifier.fit(X_train, y= y_train, categorical_feature=index_categorical_features)

        # The most useful features can be selected for efficency.
        print("Feature importance: ", classifier.feature_importances_)


        # Predict directly
        y_pred = classifier.predict(X_test, num_iteration=classifier.best_iteration_)

        print("{}/{}".format(sum(y_test==y_pred), len(y_test)))

        tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
        print("tn: {} ; fp: {} ; fn: {} ; tp: {}".format(tn, fp, fn, tp))
        print("F1-score: {}".format(f1_score(y_test, y_pred)))


        # Predict with threshhold
        threshhold = 0.7

        y_pred_cert = classifier.predict_proba(X_test, num_iteration=classifier.best_iteration_)
        y_pred_cert = [np.argmax(x) if (threshhold < max(x)) else -1 for x in y_pred_cert]

        y_test_drop, y_pred_cert_drop = [], []
        for i in range(len(y_pred_cert)):
            if(y_pred_cert[i] != -1):
                y_test_drop.append(y_test[i])
                y_pred_cert_drop.append(y_pred_cert[i])

        print("Predicted correctly: {}/{}".format(sum((y_pred_cert != -1) == (y_test==y_pred_cert)), len(y_test_drop)))
        print("Not predicted: {}/{}".format(sum([1 if x == -1 else 0 for x in y_pred_cert]), len(y_test)))

        tn, fp, fn, tp = confusion_matrix(y_test_drop,y_pred_cert_drop).ravel()
        print("tn: {} ; fp: {} ; fn: {} ; tp: {}".format(tn, fp, fn, tp))
        print("F1-score: {}".format(f1_score(y_test_drop, y_pred_cert_drop)))


    @staticmethod
    def fill_type(cluster_dataset):
        cluster_dataset = cluster_dataset.__deepcopy__()
        categorical_columns = ['location',
                               'kind',
                               'equipment_heatings']
        # integer encoding
        cluster_dataset['equipment_heatings'] = binarize_equipment(cluster_dataset['equipment_heatings'])
        cluster_dataset[categorical_columns] = cluster_dataset[categorical_columns].astype('category')
        for col in categorical_columns:
            cluster_dataset[col] = cluster_dataset[col].cat.codes

        valid_pools = cluster_dataset.dropna(subset=['type'])
        valid_pools['type'] = valid_pools['type'].astype('category')
        valid_pools['type'] = valid_pools['type'].cat.codes

        invalid_pools = cluster_dataset[cluster_dataset['type'].isnull()]

        explanatory_variables = [
                                'average_temp',
                                'std_temp',
                                'absolute_average_local_diff_temp',
                                'quantile_25_temp',
                                'quantile_75_temp',
                                'autocorr_1_temp',
                                'autocorr_4_temp',
                                'autocorr_7_temp',
                                'autocorr_10_temp',
                                'norm_fourier_0_temp',
                                'norm_fourier_1_temp',
                                'norm_fourier_2_temp',
                                'norm_fourier_3_temp',
                                'norm_fourier_4_temp',
                                'average_cond',
                                'std_cond',
                                'absolute_average_local_diff_cond',
                                'quantile_25_cond',
                                'quantile_75_cond',
                                'autocorr_1_cond',
                                'autocorr_4_cond',
                                'autocorr_7_cond',
                                'autocorr_10_cond',
                                'autocorr_10_cond',
                                'norm_fourier_0_cond',
                                'norm_fourier_1_cond',
                                'norm_fourier_2_cond',
                                'norm_fourier_3_cond',
                                'norm_fourier_4_cond',
                                'average_ph',
                                'std_ph',
                                'absolute_average_local_diff_ph',
                                'quantile_25_ph',
                                'quantile_75_ph',
                                'autocorr_1_ph',
                                'autocorr_4_ph',
                                'autocorr_7_ph',
                                'autocorr_10_ph',
                                'norm_fourier_0_ph',
                                'norm_fourier_1_ph',
                                'norm_fourier_2_ph',
                                'norm_fourier_3_ph',
                                'norm_fourier_4_ph',
                                'average_orp',
                                'std_orp',
                                'absolute_average_local_diff_orp',
                                'quantile_25_orp',
                                'quantile_75_orp',
                                'autocorr_1_orp',
                                'autocorr_4_orp',
                                'autocorr_7_orp',
                                'autocorr_10_orp',
                                'norm_fourier_0_orp',
                                'norm_fourier_1_orp',
                                'norm_fourier_2_orp',
                                'norm_fourier_3_orp',
                                'norm_fourier_4_orp',
                                'n_measurements',
                                'volume_capacity',
                                'ratio_missing_measurement',
                                'location',
                                'kind',
                                'equipment_heatings'
                                ]

        X = valid_pools[explanatory_variables]
        y = valid_pools['type']
        X_train, X_test, y_train, y_test = split_train(X.values, y.values, test_size=0.2)
        import lightgbm as lgb
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_ = compute_class_weight('balanced', np.unique(y_train), y_train)
        class_weights = dict(zip(np.unique(y_train), class_weights_))

        classifier = lgb.LGBMClassifier(num_leaves=5,
                                        max_depth=-1,
                                        n_estimators=100,
                                        learning_rate=0.2,
                                        boosting_type="gbdt",
                                        objective="cross_entropy",
                                        random_state=0,
                                        class_weight=class_weights
                                        )

        index_categorical_features = [idx for idx, element in enumerate(explanatory_variables) if
                                      element in categorical_columns]
        classifier.fit(X_train, y=y_train, categorical_feature=index_categorical_features)

        """from sklearn.ensemble import ExtraTreesClassifier
        classifier = ExtraTreesClassifier(n_estimators=250 , max_depth=5, random_state=0, class_weight='balanced')
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)"""
        """from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        X_quant = StandardScaler().fit_transform(X[list(set(explanatory_variables).difference(set(categorical_columns)))])
        X_train, X_test, y_train, y_test = split_train(X_quant, y.values, test_size=0.2)
        classifier = KNeighborsClassifier(n_neighbors=1, weights='distance')
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)"""

        # for i in range(len(y_pred)):
        #    print("{}/{}".format(y_test[i], y_pred[i]))

        # The most useful features can be selected for efficency.
        print("Feature importance: ", classifier.feature_importances_)


        # Predict directly
        y_pred = classifier.predict(X_test, num_iteration=classifier.best_iteration_)

        print("{}/{}".format(sum(y_test == y_pred), len(y_test)))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("tn: {} ; fp: {} ; fn: {} ; tp: {}".format(tn, fp, fn, tp))
        print("F1-score: {}".format(f1_score(y_test, y_pred)))


        # Predict with threshhold
        threshhold = 0.7

        y_pred_cert = classifier.predict_proba(X_test, num_iteration=classifier.best_iteration_)
        y_pred_cert = [np.argmax(x) if (threshhold < max(x)) else -1 for x in y_pred_cert]

        y_test_drop, y_pred_cert_drop = [], []
        for i in range(len(y_pred_cert)):
            if(y_pred_cert[i] != -1):
                y_test_drop.append(y_test[i])
                y_pred_cert_drop.append(y_pred_cert[i])

        print("Predicted correctly: {}/{}".format(sum((y_pred_cert != -1) == (y_test==y_pred_cert)), len(y_test_drop)))
        print("Not predicted: {}/{}".format(sum([1 if x == -1 else 0 for x in y_pred_cert]), len(y_test)))

        tn, fp, fn, tp = confusion_matrix(y_test_drop,y_pred_cert_drop).ravel()
        print("tn: {} ; fp: {} ; fn: {} ; tp: {}".format(tn, fp, fn, tp))
        print("F1-score: {}".format(f1_score(y_test_drop, y_pred_cert_drop)))


    @staticmethod
    def fill_kind(cluster_dataset):
        cluster_dataset = cluster_dataset.__deepcopy__()
        categorical_columns = ['location',
                               'type',
                               'equipment_heatings']
        # integer encoding
        cluster_dataset['equipment_heatings'] = binarize_equipment(cluster_dataset['equipment_heatings'])
        cluster_dataset[categorical_columns] = cluster_dataset[categorical_columns].astype('category')
        for col in categorical_columns:
            cluster_dataset[col] = cluster_dataset[col].cat.codes

        valid_pools = cluster_dataset.dropna(subset=['kind'])
        valid_pools['kind'] = valid_pools['kind'].astype('category')
        valid_pools['kind'] = valid_pools['kind'].cat.codes

        invalid_pools = cluster_dataset[cluster_dataset['type'].isnull()]

        explanatory_variables = ['average_temp',
                                 'std_temp',
                                 'absolute_average_local_diff_temp',
                                 'quantile_25_temp',
                                 'quantile_75_temp',
                                 'autocorr_1_temp',
                                 'autocorr_4_temp',
                                 'autocorr_7_temp',
                                 'autocorr_10_temp',
                                 'average_cond',
                                 'std_cond',
                                 'absolute_average_local_diff_cond',
                                 'quantile_25_cond',
                                 'quantile_75_cond',
                                 'autocorr_1_cond',
                                 'autocorr_4_cond',
                                 'autocorr_7_cond',
                                 'autocorr_10_cond',
                                 'autocorr_10_cond',
                                 'average_ph',
                                 'std_ph',
                                 'absolute_average_local_diff_ph',
                                 'quantile_25_ph',
                                 'quantile_75_ph',
                                 'autocorr_1_ph',
                                 'autocorr_4_ph',
                                 'autocorr_7_ph',
                                 'autocorr_10_ph',
                                 'average_orp',
                                 'std_orp',
                                 'absolute_average_local_diff_orp',
                                 'quantile_25_orp',
                                 'quantile_75_orp',
                                 'autocorr_1_orp',
                                 'autocorr_4_orp',
                                 'autocorr_7_orp',
                                 'autocorr_10_orp',
                                 'n_measurements',
                                 'volume_capacity',
                                 'ratio_missing_measurement',
                                 'location',
                                 'type',
                                 'equipment_heatings']

        X = valid_pools[explanatory_variables]
        y = valid_pools['kind']
        X_train, X_test, y_train, y_test = split_train(X.values, y.values, test_size=0.2)
        import lightgbm as lgb
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_ = compute_class_weight('balanced', np.unique(y_train), y_train)
        class_weights = dict(zip(np.unique(y_train), class_weights_))

        classifier = lgb.LGBMClassifier(num_leaves=5,
                                        max_depth=-1,
                                        n_estimators=750,
                                        learning_rate=0.2,
                                        boosting_type="gbdt",
                                        objective="cross_entropy",
                                        random_state=0,
                                        class_weight=class_weights
                                        )

        index_categorical_features = [idx for idx, element in enumerate(explanatory_variables) if
                                      element in categorical_columns]
        classifier.fit(X_train, y=y_train, categorical_feature=index_categorical_features)

        # Predict directly
        y_pred = classifier.predict(X_test, num_iteration=classifier.best_iteration_)

        print("{}/{}".format(sum(y_test == y_pred), len(y_test)))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("tn: {} ; fp: {} ; fn: {} ; tp: {}".format(tn, fp, fn, tp))
        print("F1-score: {}".format(f1_score(y_test, y_pred)))


        # Predict with threshhold
        threshhold = 0.7

        y_pred_cert = classifier.predict_proba(X_test, num_iteration=classifier.best_iteration_)
        y_pred_cert = [np.argmax(x) if (threshhold < max(x)) else -1 for x in y_pred_cert]

        y_test_drop, y_pred_cert_drop = [], []
        for i in range(len(y_pred_cert)):
            if(y_pred_cert[i] != -1):
                y_test_drop.append(y_test[i])
                y_pred_cert_drop.append(y_pred_cert[i])

        print("Predicted correctly: {}/{}".format(sum((y_pred_cert != -1) == (y_test==y_pred_cert)), len(y_test_drop)))
        print("Not predicted: {}/{}".format(sum([1 if x == -1 else 0 for x in y_pred_cert]), len(y_test)))

        tn, fp, fn, tp = confusion_matrix(y_test_drop,y_pred_cert_drop).ravel()
        print("tn: {} ; fp: {} ; fn: {} ; tp: {}".format(tn, fp, fn, tp))
        print("F1-score: {}".format(f1_score(y_test_drop, y_pred_cert_drop)))
