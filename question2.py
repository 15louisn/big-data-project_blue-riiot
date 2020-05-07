
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle as cPickle

from sklearn.metrics import mean_absolute_error

from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

def load_files(id = "swp_event"):
    return pd.read_csv("swp_db.csv", low_memory=False)
    # return pd.read_csv("swp_event.csv", low_memory=False)

# provide the indexes that bounds a particular swimming pool (id) in a dataset (data)
def bound_swp(id,data):
    bounds = np.zeros(2)
    bool = 0
    for i in range(0,data.shape[0]):
        if data.swimming_pool_id[i] == id:
            if bool == 0:
                bool = 1
                bounds[0] = i

        if data.swimming_pool_id[i] != id and bool == 1:
            bool = 2
            bounds[1] = i-1
            break

        if bool == 1 and i == data.shape[0]:
            bounds[1] = i
            bool = 2

    return bounds

def get_pool_data(data, id):
    return data.loc[data['swimming_pool_id'] == id]

def autocorr(data,lags = 500 ):
    return acf(data, unbiased=True,nlags = lags)
def partial_autocorr(data, lags = 20):
    return pacf(data, nlags=lags)

def pool_data(data, nlags=500, lags =20):
    plt.plot()
# data = specific time series of a pool / db


#  Count the number of valid sample one could build using a particular dataset (time-series)
#  It requires the time difference in minutes between two measures, bool=True means we operate the duration test
#  that consist in validating a sample if the time difference between each measurement of the sample is smaller
#  or equal than 100minutes
# It ouputs the number of sample and a binary vector, 1 in position i means a sample can be built to forecast i
#  i.e. the three measurements before i are well-spaced
def count_sample(pool_time, bool):
    pool_time.reset_index(inplace=True, drop=True)
    index = np.zeros(pool_time.shape[0])
    counter = 0
    for t in range(2, pool_time.shape[0]):
        if(good(pool_time[t-1], bool) and good(pool_time[t-2], bool) and good(pool_time[t], bool)):
            index[t] = 1  # mean we can build a sample on 't'
            counter = counter +1
    return counter, index

# def good(val, bool):
#     if bool:
#         return ( 50 <= val and val <= 100)
#     else :
#         return True

def good(val, bool):
    if bool:
        return ( val <= 100)
    else :
        return True

# db being the database and nb the number of biggest pools we want to use to build the sample
#  when it is called from the database builder
def build_conductivity_db(db, nb):
    histo = db.swimming_pool_id.value_counts()
    df = get_pool_data(db, histo.index[0])
    # df2 = get_pool_data(db, histo.index[1])
    # df = df.append(df2, sort=False)
    # print(df.shape)
    for i in range(1,nb):
        temp = get_pool_data(db, histo.index[i])
        df = df.append(temp, sort=False)
    return df


def get_large_swp(db):
    histo = db.swimming_pool_id.value_counts()
    return histo

# Generate orp data base i.e. samples of the 3 previous measurements as they are used to forecast the next measurement
# pool is a database
# bool = True means we take into account the condition on the dataset which is to respect
# a maximum duration of 100min between two consecutive measurements. If not, the sample is discarded
# nb is the number of biggest pools we should use in order build the set of samples
def orp_db_generator(pool, horizon=3, nb = 0, bool = True):

    if nb != 0:
        pool = build_conductivity_db(pool, nb)
    print("Original database size: ", pool.shape[0])
    pool.reset_index(inplace=True, drop=True)

    size, index = count_sample(pool.time_step, bool)
    db_size = size # is the number of samples

    X = np.zeros((db_size, 3))
    y = np.zeros(db_size)
    created = pd.DataFrame(index=range(db_size),columns=range(1))
    db_index = 0
    for i in range(3, pool.shape[0]):
        if (index[i] == 1):
            y[db_index] = pool.data_orp[i]
            X[db_index, 0] = pool.data_orp[i-1]
            X[db_index, 1] = pool.data_orp[i-2]
            X[db_index, 2] = pool.data_orp[i-3]
            created.at[db_index,0] = pool.created[i]
            db_index = db_index + 1
    return X, y, index, created

# Generate ph data base i.e. samples of the 3 previous measurements
def ph_db_generator(pool, horizon=3, nb = 0, bool = True):

    if nb != 0:
        pool = build_conductivity_db(pool, nb)
    print("Original database size: ", pool.shape[0])
    pool.reset_index(inplace=True, drop=True)

    size, index = count_sample(pool.time_step, bool)
    db_size = size # is the number of samples

    X = np.zeros((db_size, 3))
    y = np.zeros(db_size)
    created = pd.DataFrame(index=range(db_size),columns=range(1))
    db_index = 0
    for i in range(3, pool.shape[0]):
        if (index[i] == 1):
            y[db_index] = pool.data_ph[i]
            X[db_index, 0] = pool.data_ph[i-1]
            X[db_index, 1] = pool.data_ph[i-2]
            X[db_index, 2] = pool.data_ph[i-3]
            created.at[db_index,0] = pool.created[i]
            db_index = db_index + 1
    return X, y, index, created


def coverage(predictions):

    anomaly_h = sum(np.greater_equal(predictions[0], predictions['upper']))
    anomaly_l= sum(np.greater_equal(predictions['lower'], predictions[0]))
    test_size = predictions[0].shape[0]
    coverage = (test_size - (anomaly_h + anomaly_l)) / test_size

    print(" The coverage is : ", coverage)
    return coverage

# https://stats.stackexchange.com/questions/213050/scoring-quantile-regressor
# compute the same loss used for optimization
def quantile_loss(target, quantile_forecast, quantile):
    return np.mean((target - quantile_forecast) * (quantile - (target < quantile_forecast).astype(int)))

# compare a pair of quantiles to the target value by averaging the low and high quantiles losses
def full_quantile_loss(target, low_quantile_forecast, high_quantile_forecast, alpha):
    return (quantile_loss(target, low_quantile_forecast, alpha) + quantile_loss(target, high_quantile_forecast, 1-alpha))/2


def save_training_data(filename, X, y):
    #np.save(filename+"_X", X)
    #np.save(filename+"_y", y)
    X.to_pickle(filename+"_X")
    y.to_pickle(filename+"_y")

def np_save_training_data(filename, X, y):
    np.save(filename+"_X", X)
    np.save(filename+"_y", y)

def read_training_data(filename):
    #return np.load(filename+"_X.npy", allow_pickle=True), np.load(filename+"_y.npy", allow_pickle=True)
    return pd.read_pickle(filename+"_X"),  pd.read_pickle(filename+"_y")
# read_training_data("v1_pres_3_1")  for pH data

def np_read_training_data(filename):
    #return np.load(filename+"_X.npy", allow_pickle=True), np.load(filename+"_y.npy", allow_pickle=True)
    return np.load(filename+"_X" + '.npy'),  np.load(filename+"_y" + '.npy')


def GBoost2(db, test_swp, nb1 = 5):
    # predictions = GBoost(swp_data, test_pool, 80)
    LOWER_ALPHA = 0.05
    UPPER_ALPHA = 0.95

    # Load model and set

    # Each model has to be separate
    lower_model = GradientBoostingRegressor(loss="quantile",alpha=LOWER_ALPHA)
    # The mid model will use the default loss
    mid_model = GradientBoostingRegressor(loss="ls")
    # mid_model = GradientBoostingRegressor(loss="quantile", alpha = 0.5)
    upper_model = GradientBoostingRegressor(loss="quantile",alpha=UPPER_ALPHA)

    # db.reset_index(inplace=True, drop=True)
    # X_train, y_train, index, cr = ph_db_generator(db, nb = nb1)

    filename = "pH_80_biggest2"
    # np_save_training_data(filename,X_train, y_train)
    # print("saved")
    X_train, y_train = np_read_training_data(filename)
    print("read")

    X_test, y_test, index2, created = ph_db_generator(test_swp, nb = 0, bool = True)
    # False to consider all samples - unconditionally

    print("X test size", X_test.shape)
    print("ytest size", y_test.shape)

    print("Database size: ", y_train.shape[0])

    # X_train,y_train = generate_db(pool)
    print("Data base generated")

    # Fit models
    # lower_model.fit(X_train, y_train)
    # print("lower model fitted")
    # mid_model.fit(X_train, y_train)
    # print("mid model fitted")
    # upper_model.fit(X_train, y_train)
    # print("upper model fitted")
    #
    # with open('802_lower_ph.pkl', 'wb') as fid:
    #     cPickle.dump(lower_model, fid)
    #
    # with open('802_mid_ph.pkl', 'wb') as fid:
    #     cPickle.dump(mid_model, fid)
    #
    # with open('802_upper_ph.pkl', 'wb') as fid:
    #     cPickle.dump(upper_model, fid)
    # print("all dumped")

    with open('802_lower_ph.pkl', 'rb') as fid:
        lower_model = cPickle.load(fid)

    with open('802_mid_ph.pkl', 'rb') as fid:
        mid_model = cPickle.load(fid)

    with open('802_upper_ph.pkl', 'rb') as fid:
        upper_model = cPickle.load(fid)
    print("models read")

    # Displays metrics

    # Record actual values on test set
    predictions = pd.DataFrame(y_test)
    # Predict
    predictions['lower'] = lower_model.predict(X_test)
    print("done")
    predictions['mid'] = mid_model.predict(X_test)
    print("done")
    predictions['upper'] = upper_model.predict(X_test)
    print("done")
    predictions = predictions.set_index(pd.DatetimeIndex(created[0].values))
    predictions.rename(columns={0: 'measures'}, inplace=True)

    axes = predictions.plot(style='.-', color=['blue', 'red', 'green', 'red'])
    # plt.plot(range(0, len(y_test)), predictions['lower'], color='r', marker = '.', linewidth=2)
    # plt.plot(range(0, len(y_test)), predictions['upper'], color='r', marker = '.', linewidth=2)
    # plt.plot(range(0, len(y_test)), predictions['mid'], color='g', marker = '.', linewidth=2)
    # plt.plot(range(0, len(y_test)),  y_test, color='b', marker = '.', linewidth=2)


    print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions['mid']))

    zz1 = np.greater_equal(y_test, predictions['upper'])
    zz2 = np.greater_equal(predictions['lower'], y_test)
    print(sum(zz1))
    print(sum(zz2))
    predictions.rename(columns={'measures' : 0}, inplace=True)
    coverage(predictions)
    print("Quantile loss {}".format(full_quantile_loss(y_test, predictions['lower'], predictions['upper'], alpha=0.05)))

    anomalies = [x or y for (x, y) in zip(zz1, zz2)]
    # anomalies = int(anomalies)
    anomalies = list(map(int, anomalies))
    anomalies = [element * 7 for element in anomalies]
    print(anomalies)
    # anomalies = np.asarray(anomalies)
    anomalies = pd.DataFrame(anomalies)
    print(anomalies[0].sum())
    anomalies = anomalies.set_index(pd.DatetimeIndex(created[0].values))
    anomalies.plot( color='r', marker = "*", linewidth=0, ax = axes)
    # axes.plot(range(0, len(y_test)),  anomalies, color='r', marker = "*", linewidth=0)
    # plt.plot(range(0, len(y_train)),  y_train, color='b', marker = '.', linewidth=0)



    return predictions


def plot_pool(test_pool):
    data = test_pool[["created", "data_ph"]]
    datetime_index = pd.DatetimeIndex(data["created"].values)
    data = data.set_index(datetime_index)
    data.plot(style='.-')


#  Outputs main metrics and display the results (interval bounds + time series)
#  db is the database, test_swp is the data of the swimming pool being tested, nb is the number of
#  biggest pools used for training
def linear_regression(test_swp, nb = 0):
    from statsmodels.regression.linear_model import OLS
    random_seed = 0
    random.seed(random_seed)

    filename = "pH_80_biggest2"
    X_train, y_train = np_read_training_data(filename)
    X_test, y_test, index2, created = ph_db_generator(test_swp, nb = 0, bool = True)
    # X_test, y_test, index2, created = orp_db_generator(test_swp, nb = 0, bool = True)


    print("X test size", X_test.shape)
    print("ytest size", y_test.shape)

    print("Database size: ", y_train.shape[0])

    # Define the interval bound from a gaussian centered in the forecast
    from scipy.stats import norm
    def ols_quantile(m, X, q):
        # m: OLS statsmodels model.
        # X: X matrix.
        # q: Quantile.
        mean_pred = m.predict(X)
        se = np.sqrt(m.scale)
        return mean_pred + norm.ppf(q) * se

    model = OLS(y_train[:].astype(float), X_train.astype(float))
    model = model.fit()
    print('model fitted')
    predictions = pd.DataFrame(y_test)
    predictions['lower'] = ols_quantile(model, X_test.astype(float), 0.3)
    predictions['upper'] = ols_quantile(model, X_test.astype(float), 0.7)
    # predictions['lower'] = ols_quantile(model, X_test.astype(float), 0.05)
    # predictions['upper'] = ols_quantile(model, X_test.astype(float), 0.95)

    predictions = predictions.set_index(pd.DatetimeIndex(created[0].values))
    predictions.rename(columns={0: 'measures'}, inplace=True)
    axes = predictions.plot(style='.-', color=['blue', 'red', 'green', 'red'])

    # Displays the main metrics

    print("Mean absolute error " + str(mean_absolute_error(y_pred=(predictions['lower']+predictions['upper'])/2, y_true=y_test)))
    print("Quantile loss {}".format(full_quantile_loss(y_test, predictions['lower'], predictions['upper'], alpha=0.05)))
    predictions.rename(columns={'measures' : 0}, inplace=True)
    print("Coverage {}".format(coverage(predictions)))
    predictions.rename(columns={0: 'measures'}, inplace=True)

    #  Count the number of value under the min interval bound and above the upper interval bound
    zz1 = np.greater_equal(y_test, predictions['upper'])
    zz2 = np.greater_equal(predictions['lower'], y_test)
    print(sum(zz1))
    print(sum(zz2))
    anomalies = [x or y for (x, y) in zip(zz1, zz2)]
    # anomalies = int(anomalies)
    anomalies = list(map(int, anomalies))
    anomalies = [element * 7 for element in anomalies]
    # print(anomalies)
    # anomalies = np.asarray(anomalies)
    anomalies = pd.DataFrame(anomalies)
    # print(anomalies[0].sum())
    anomalies = anomalies.set_index(pd.DatetimeIndex(created[0].values))

    # Displays a star as an anomaly is detected
    anomalies.plot( color='r', marker = "*", linewidth=0, ax = axes)

    predictions.rename(columns={'measures': 0}, inplace=True)
    zz1 = np.greater_equal(predictions[0], predictions['upper'])
    zz2 = np.greater_equal(predictions['lower'], predictions[0])
    deriv = [x or y for (x, y) in zip(zz1, zz2)]

    filename = "lin_swp_ph_" + str(nb)
    np.save(filename, deriv)

    filename = "lin_index_ph_" + str(nb)
    np.save(filename, index2)
    predictions.rename(columns={0: 'measures'}, inplace=True)


    return predictions

#  Outputs main metrics and display the results (interval bounds + time series)
#  db is the database, test_swp is the data of the swimming pool being tested, nb is the number of
#  biggest pools used for training
def linear_regression_orp(db, test_swp, nb):
    from statsmodels.regression.linear_model import OLS
    random_seed = 0
    random.seed(random_seed)

    # Load data
    db.reset_index(inplace=True, drop=True)
    # X_train, y_train, index, cr = orp_db_generator(db, nb = 80)

    filename = "orp_80_biggest2"
    # np_save_training_data(filename,X_train, y_train)
    # print('saved')

    X_train, y_train = np_read_training_data(filename)
    X_test, y_test, index2, created = orp_db_generator(test_swp, nb = 0, bool = True)


    from scipy.stats import norm
    # Build the quantile
    def ols_quantile(m, X, q):
        # m: OLS statsmodels model.
        # X: X matrix.
        # q: Quantile.
        mean_pred = m.predict(X)
        se = np.sqrt(m.scale)
        print(se)
        return mean_pred + norm.ppf(q) * se

    model = OLS(y_train[:].astype(float), X_train.astype(float))
    model = model.fit()
    print('model fitted')
    predictions = pd.DataFrame(y_test)
    predictions['lower'] = ols_quantile(model, X_test.astype(float), 0.48)
    predictions['upper'] = ols_quantile(model, X_test.astype(float), 0.52)


    # Displays the main metrics
    predictions = predictions.set_index(pd.DatetimeIndex(created[0].values))
    predictions.rename(columns={0: 'measures'}, inplace=True)
    axes = predictions.plot(style='.-', color=['blue', 'red', 'green', 'red'])

    print("Mean absolute error " + str(
        mean_absolute_error(y_pred=(predictions['lower'] + predictions['upper']) / 2, y_true=y_test)))
    print("Quantile loss {}".format(full_quantile_loss(y_test, predictions['lower'], predictions['upper'], alpha=0.05)))
    predictions.rename(columns={'measures': 0}, inplace=True)
    print("Coverage {}".format(coverage(predictions)))
    predictions.rename(columns={0: 'measures'}, inplace=True)

    #  Count the number of value under the min interval bound and above the upper interval bound
    zz1 = np.greater_equal(y_test, predictions['upper'])
    zz2 = np.greater_equal(predictions['lower'], y_test)
    print(sum(zz1))
    print(sum(zz2))
    anomalies = [x or y for (x, y) in zip(zz1, zz2)]
    # anomalies = int(anomalies)
    anomalies = list(map(int, anomalies))
    anomalies = [element * 400 for element in anomalies]
    # print(anomalies)
    # anomalies = np.asarray(anomalies)
    anomalies = pd.DataFrame(anomalies)
    # print(anomalies[0].sum())

    # Displays a star for each anomaly
    anomalies = anomalies.set_index(pd.DatetimeIndex(created[0].values))
    anomalies.plot(color='r', marker="*", linewidth=0, ax=axes)

    predictions.rename(columns={'measures': 0}, inplace=True)
    zz1 = np.greater_equal(predictions[0], predictions['upper'])
    zz2 = np.greater_equal(predictions['lower'], predictions[0])
    deriv = [x or y for (x, y) in zip(zz1, zz2)]

    filename = "lin_swp_orp_" + str(nb)
    np.save(filename, deriv)

    filename = "lin_index_orp_" + str(nb)
    np.save(filename, index2)
    predictions.rename(columns={0: 'measures'}, inplace=True)

    return predictions


def calculate_error(predictions):
    """
    Calculate the absolute error associated with prediction intervals

    :param predictions: dataframe of predictions
    :return: None, modifies the prediction dataframe

    """
    predictions['absolute_error_lower'] = (predictions['lower'] - predictions[0]).abs()
    predictions['absolute_error_upper'] = (predictions['upper'] - predictions[0]).abs()

    predictions['absolute_error_interval'] = (predictions['absolute_error_lower'] + predictions['absolute_error_upper']) / 2
    predictions['absolute_error_mid'] = (predictions['mid'] - predictions[0]).abs()
    predictions['in_bounds'] = predictions[0].between(left=predictions['lower'], right=predictions['upper'])

    metrics = predictions[['absolute_error_lower', 'absolute_error_upper', 'absolute_error_interval', 'absolute_error_mid',
                    'in_bounds']].copy()
    # metrics.describe()
    print("Absolute error mid ", metrics["absolute_error_mid"].mean())
    print("Absolute error interval ",metrics["absolute_error_interval"].mean())

if __name__ == "__main__":
    swp_data = load_files()
    
    # swp_db = load_files()
    list(swp_data.columns)
    swp_data.rename(columns={'Unnamed: 0': 'x'}, inplace=True)
    # swp_db.rename(columns={'Unnamed: 0': 'x'}, inplace=True)
    mypool = "f0d1d161-9f96-465a-9900-f13012b481c6"
    pool = get_pool_data(swp_data, mypool)
    mypool2 = "2820a378-7ac0-4096-ada4-70350813a2bf"
    pool2 = get_pool_data(swp_data, mypool2)
    mypool3 = "80210e99-9886-4281-b857-c3075f827258"
    pool3 = get_pool_data(swp_data, mypool3)

    mypool4 = "935af500-6b63-49ae-af29-117ad46d20af"
    pool4 = get_pool_data(swp_data, mypool4)

    # refgression(pool4.data_conductivity)