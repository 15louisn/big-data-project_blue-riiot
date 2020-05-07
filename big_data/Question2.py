
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import pools
import pickle as cPickle


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

def load_files(id = "swp_event"):
    return pd.read_csv("swp_db.csv", low_memory=False)
    # return pd.read_csv("swp_event.csv", low_memory=False)

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
def generate_db(data, horizon = 3):
    nbrows = data.shape[0]
    db_size = nbrows-horizon
    X = np.zeros((db_size, horizon))
    y = np.zeros(db_size)
    i = 0
    index = 0
    for index in range(horizon, nbrows):
        y[index-horizon] = data[index]

    for i in range(horizon, nbrows):
        for j in range(0,horizon):
            X[i-horizon, j] = data[i-(j+1)]

    return X,y

# def count_sample(pool_time, bool):
#     pool_time.reset_index(inplace=True, drop=True)
#     index = np.zeros(pool_time.shape[0])
#     counter = 0
#     for t in range(2, pool_time.shape[0]):
#         if(good(pool_time[t-1], bool) and good(pool_time[t-2], bool)):
#             index[t] = 1  # mean we can build a sample on 't'
#             counter = counter +1
#     return counter, index

def count_sample(pool_time, bool):
    pool_time.reset_index(inplace=True, drop=True)
    index = np.zeros(pool_time.shape[0])
    counter = 0
    for t in range(2, pool_time.shape[0]):
        if(good(pool_time[t-1], bool) and good(pool_time[t-2], bool) and good(pool_time[t], bool)):
            index[t] = 1  # mean we can build a sample on 't'
            counter = counter +1
    return counter, index

def good(val, bool):
    if bool:
        return (val <= 100)
    else :
        return True

def build_conductivity_db(db, nb):
    histo = db.swimming_pool_id.value_counts()
    df = get_pool_data(db, histo.index[0])
    # df2 = get_pool_data(db, histo.index[1])
    # df = df.append(df2, sort=False)
    # print(df.shape)
    l = list(range(1,nb))
    forbidden = list(range(5,11))
    forbidden.append(14)
    new_list = []
    for e in l:
        if e not in forbidden:
            new_list.append(e)

    for i in new_list:
        temp = get_pool_data(db, histo.index[i])
        df = df.append(temp, sort=False)
    return df


# pool is a database
# bool = False creates test sets
def cd_db_generator(pool, horizon=3, nb = 0, bool = True):

    if nb != 0:
        pool = build_conductivity_db(pool, nb)
    print("Original database size: ", pool.shape[0])
    pool.reset_index(inplace=True, drop=True)

    size, index = count_sample(pool.time_step, bool)
    db_size = size # is the number of samples

    X = np.zeros((db_size, 3))
    y = np.zeros(db_size)

    db_index = 0

    for i in range(3, pool.shape[0]):
        if (index[i] == 1):
            y[db_index] = pool.data_orp[i]
            X[db_index, 0] = pool.data_orp[i-1]
            X[db_index, 1] = pool.data_orp[i-2]
            X[db_index, 2] = pool.data_orp[i-3]
            db_index = db_index + 1

    # for i in range(3, pool.shape[0]):
    #     if (index[i] == 1):
    #         y[db_index] = pool.data_ph[i]
    #         X[db_index, 0] = pool.data_ph[i-1]
    #         X[db_index, 1] = pool.data_ph[i-2]
    #         X[db_index, 2] = pool.data_ph[i-3]
    #         db_index = db_index + 1


    return X, y, index

def test_db_build(pool):
    X = np.zeros((pool.shape[0], 3))
    y = np.zeros(pool.shape[0])
    pool = pool.to_numpy()
    for i in range(3, pool.shape[0]):
        y[i] = pool[i]
        X[i, 0] = pool[i-1]
        X[i, 1] = pool[i-2]
        X[i, 2] = pool[i-3]
    return X, y

# def regression(swp_data, id):
def regression(pool):
    # pool = get_pool_data(swp_data, swp_id)
    pool.reset_index(inplace=True, drop=True)
    X, y = generate_db(pool)

    reg_model = LinearRegression().fit(X,y)
    y_pred = reg_model.predict(X)
    plt.plot(range(0, len(y)), y, color='r')
    plt.plot(range(-1, len(y) - 1), y_pred, color='b')

    # X1, y1 = shuffle(X,np.array(y).ravel(), random_state=0)
    # reg_model = LinearRegression().fit(X1,y1)
    # y_pred1 = reg_model.predict(X1)
    # plt.plot(range(0, len(y1)), y1, color='r')
    # plt.plot(range(-1, len(y1) - 1), y_pred1, color='b')
#
def is_outlier(perc, upper, lower):
    return (perc > upper) or (perc < lower)

def GBoostTrain(db, nb1 = 5):
    LOWER_ALPHA = 0.05
    UPPER_ALPHA = 0.95
    # Each model has to be separate
    lower_model = GradientBoostingRegressor(loss="quantile",alpha=LOWER_ALPHA)
    # The mid model will use the default loss
    mid_model = GradientBoostingRegressor(loss="ls")
    # mid_model = GradientBoostingRegressor(loss="quantile", alpha = 0.5)
    upper_model = GradientBoostingRegressor(loss="quantile",alpha=UPPER_ALPHA)

    db.reset_index(inplace=True, drop=True)

    X_train, y_train, index = cd_db_generator(db, nb = nb1)

    print("Database size: ", y_train.shape[0])

    # X_train,y_train = generate_db(pool)
    print("Data base generated")

    # Fit models
    lower_model.fit(X_train, y_train)
    print("lower model fitted")
    mid_model.fit(X_train, y_train)
    print("mid model fitted")
    upper_model.fit(X_train, y_train)
    print("upper model fitted")

    with open('lower.pkl', 'wb') as fid:
        cPickle.dump(lower_model, fid)

    with open('mid.pkl', 'wb') as fid:
        cPickle.dump(mid_model, fid)

    with open('upper.pkl', 'wb') as fid:
        cPickle.dump(upper_model, fid)

def GBoostTest(test_swp):
    X_test, y_test = test_db_build(test_swp)
    print("X test size", X_test.shape)
    # print("ytest size", y_test.shape)

    with open('lower.pkl', 'rb') as fid:
        lower_model= cPickle.load(fid)

    with open('mid.pkl', 'rb') as fid:
        mid_model = cPickle.load(fid)

    with open('upper.pkl', 'rb') as fid:
        upper_model = cPickle.load(fid)

    # Record actual values on test set
    # predictions = pd.DataFrame(y_test)
    # Predict
    # predictions['lower'] = lower_model.predict(X_test)
    lowers = lower_model.predict(X_test)

    # print("done")
    # predictions['mid'] = mid_model.predict(X_test)
    # print("done")
    # predictions['upper'] = upper_model.predict(X_test)
    uppers = upper_model.predict(X_test)
    # print("done")

    out_prediction = np.zeros(X_test.shape[0])

    for i in range(len(X_test)):
        out_prediction[i] = is_outlier(X_test[i,0], uppers[i], lowers[i])
    # print(out_prediction)

    zz1 = np.greater_equal(y_test, uppers)
    zz2 = np.greater_equal(lowers, y_test)
    # print(zz1)
    # print(zz2)
    deriv = [ x or  y for (x,y) in zip (zz1, zz2)]
    # print(deriv)
    deriv = [int(elem) for elem in deriv]
    # print(deriv)
    # return out_prediction
    return deriv


def GBoostPlot(test_swp, out, index):
    X_test, y_test = test_db_build(test_swp)
    print(index)
    print("X test size", X_test.shape)
    # print("ytest size", y_test.shape)

    with open('lower.pkl', 'rb') as fid:
        lower_model= cPickle.load(fid)

    with open('mid.pkl', 'rb') as fid:
        mid_model = cPickle.load(fid)

    with open('upper.pkl', 'rb') as fid:
        upper_model = cPickle.load(fid)

    plot_data = pd.DataFrame(y_test)
    plot_data['down'] = lower_model.predict(X_test)
    # print(plot_data['down'])
    plot_data['prediction'] = mid_model.predict(X_test)
    plot_data['up'] = upper_model.predict(X_test)
    # print(plot_data['up'])
    # index = pd.DatetimeIndex(index.values)
    print(index.shape)
    p0 = plt.fill_between(index.to_numpy(), plot_data['up'].to_numpy(), plot_data['down'].to_numpy(),
                     color='b', alpha=.5)
    p1 = plt.plot(index.to_numpy(), plot_data['prediction'].to_numpy(),color='b')
    p2 = plt.plot(index.to_numpy(), plot_data['actual value'].to_numpy(),color='r')
    p3 = plt.fill(np.NaN, np.NaN, 'b', alpha=0.5)
    plt.xlabel('Timestamp')
    plt.ylabel('ORP (mV)')
    plt.legend([(p3[0], p1[0]), p2[0]], ['PCI','Recorded value'], loc='lower right')
    plt.show()

def GBoost(db, test_swp, nb1 = 5):
    LOWER_ALPHA = 0.05
    UPPER_ALPHA = 0.95
    # Each model has to be separate
    lower_model = GradientBoostingRegressor(loss="quantile",alpha=LOWER_ALPHA)
    # The mid model will use the default loss
    mid_model = GradientBoostingRegressor(loss="ls")
    # mid_model = GradientBoostingRegressor(loss="quantile", alpha = 0.5)
    upper_model = GradientBoostingRegressor(loss="quantile",alpha=UPPER_ALPHA)

    db.reset_index(inplace=True, drop=True)

    X_train, y_train, index = cd_db_generator(db, nb = nb1)

    print("Database size: ", y_train.shape[0])

    # X_train,y_train = generate_db(pool)
    print("Data base generated")

    # Fit models
    lower_model.fit(X_train, y_train)
    print("lower model fitted")
    mid_model.fit(X_train, y_train)
    print("mid model fitted")
    upper_model.fit(X_train, y_train)
    print("upper model fitted")


    # Record actual values on test set
    predictions = pd.DataFrame(y_test)
    # Predict
    predictions['lower'] = lower_model.predict(X_test)
    print("done")
    predictions['mid'] = mid_model.predict(X_test)
    print("done")
    predictions['upper'] = upper_model.predict(X_test)
    print("done")
    plt.plot(range(0, len(y_test)), predictions['lower'], color='r', marker = '.', linewidth=2)
    plt.plot(range(0, len(y_test)), predictions['upper'], color='r', marker = '.', linewidth=2)
    plt.plot(range(0, len(y_test)), predictions['mid'], color='g', marker = '.', linewidth=2)
    plt.plot(range(0, len(y_test)),  y_test, color='b', marker = '.', linewidth=2)

    # predictions = pd.DataFrame(y_train)
    ## Predict
    # predictions['lower'] = lower_model.predict(X_train)
    # print("done")
    # predictions['mid'] = mid_model.predict(X_train)
    # print("done")
    # predictions['upper'] = upper_model.predict(X_train)
    # print("done")
    # plt.plot(range(0, len(y_train)), predictions['lower'], color='r', marker = '.', linewidth=0)
    # plt.plot(range(0, len(y_train)), predictions['upper'], color='r', marker = '.', linewidth=0)
    # plt.plot(range(0, len(y_train)), predictions['mid'], color='g', marker = '.', linewidth=0)
    # plt.plot(range(0, len(y_train)),  y_train, color='b', marker = '.', linewidth=0)

    print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions['mid']))

    zz1 = np.greater_equal(y_test, predictions['upper'])
    zz2 = np.greater_equal(predictions['lower'], y_test)
    print(zz1)
    print(zz2)

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

# def transform()
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
