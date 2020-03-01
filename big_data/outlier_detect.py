import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random
import scipy.stats as stats

def autoreg(k, array):
    weights = np.arange(1,2*k+1)
    # print("Array :", array)
    # print("weights :", weights)
    output = np.sum(array * weights)/(np.sum(weights))
    return output

def PCI(p,k,array):
    a = stats.t.ppf(p,2*k-1)
    # print(a)
    b = np.std(array)*np.sqrt(1+1/(2*k))
    # print(b)
    interval = a * b
    # print(interval)
    return interval

def is_outlier(pred, perc, interval):
    if ((pred > (perc + interval)) or (pred < (perc - interval))):
        return 1
    else:
        return 0


if __name__ == "__main__":
    # swp_data = pd.read_csv('reduced_swp.csv')
    event_data = pd.read_csv('reduced_event.csv')

    random.seed(0)

    event_data = event_data.set_index('blue_device_serial')
    event_data = event_data.loc['000C2925']

    var_of_interest = "data_conductivity"
    event_data = event_data.reset_index(drop=True)
    time_serie = event_data[["created",var_of_interest]]

    time_serie.loc[:,"created"] = time_serie["created"].apply(pd.to_datetime)
    time_serie.loc[:,"created"] = time_serie["created"].apply(pd.Timestamp.timestamp)
    # time_serie = time_serie.set_index('created')

    # print(time_serie.head())
    # print(time_serie.shape)
    # time_serie.plot(y=var_of_interest)
    # plt.show()

    k = 3
    p = 0.95
    t = 100
    known = time_serie[0:t-1]
    # print(known)
    k_vals = known[var_of_interest]
    plot_data = pd.concat([k_vals.rename('actual value'),
                           k_vals.rename('prediction'),
                           k_vals.rename('up'),
                           k_vals.rename('down')], axis=1)
    # print(plot_data)

    for i in range(0,time_serie.shape[0]-t):
        perc = time_serie.loc[t+i][var_of_interest]
        nbrs = NearestNeighbors(n_neighbors=2*k, algorithm='ball_tree').fit(known.to_numpy())
        # print('Element:',time_serie.loc[t])
        distances, indices = nbrs.kneighbors(time_serie.loc[t+i].to_numpy().reshape(1, -1))
        # print(indices)
        nn_window = known.loc[indices[0]][var_of_interest].to_numpy()
        # print(nn_window)
        pred = autoreg(k,nn_window)
        interval = PCI(p,k,nn_window)

        # Update knowledge for next iteration
        known = known.append(time_serie.loc[t+i], ignore_index=True)
        # print(known)

        # print(perc, pred, pred+interval, pred-interval)
        # Update for plot
        new_s = pd.Series([perc, pred, pred+interval, pred-interval], index = ['actual value','prediction','up','down'])
        plot_data = plot_data.append(new_s, ignore_index=True)
        # print(plot_data)
        # print(is_outlier(pred, perc, interval))

    # plot_data.plot()
    plt.fill_between(plot_data.index.to_numpy()[500:700], plot_data['up'].to_numpy()[500:700], plot_data['down'].to_numpy()[500:700],
                     color='b', alpha=.5)
    plt.plot(plot_data.index.to_numpy()[500:700], plot_data['prediction'].to_numpy()[500:700],color='b')
    plt.plot(plot_data.index.to_numpy()[500:700], plot_data['actual value'].to_numpy()[500:700],color='r')
    plt.show()
