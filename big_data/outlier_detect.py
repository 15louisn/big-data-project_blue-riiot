import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random
import scipy.stats as stats
from sklearn import preprocessing

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
    # Implementation of https://www.hindawi.com/journals/mpe/2014/879736/
    event_data = pd.read_csv('reduced_event.csv')

    random.seed(0)

    # Select a pool
    # event_data = event_data.set_index('blue_device_serial')
    # event_data = event_data.loc['000C2925']
    #
    # event_data = event_data.set_index('swimming_pool_id')
    # event_data = event_data.loc['935af500-6b63-49ae-af29-117ad46d20af']

    event_data = event_data.set_index('swimming_pool_id')
    event_data = event_data.loc['80210e99-9886-4281-b857-c3075f827258']

    # Select a variable to study
    # var_of_interest = "data_conductivity"
    var_of_interest = "data_orp"
    event_data = event_data.reset_index(drop=True)
    time_serie = event_data[["created",var_of_interest]]

    # Convert timestamps
    time_serie.loc[:,"created"] = time_serie["created"].apply(pd.to_datetime)
    time_serie.loc[:,"created"] = time_serie["created"].apply(pd.Timestamp.timestamp)
    # time_serie = time_serie.set_index('created')

    # print(time_serie.head())
    # print(time_serie.shape)
    # time_serie.plot(y=var_of_interest)
    # plt.show()

    k = 10
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
    p0 = plt.fill_between(plot_data.index.to_numpy(), plot_data['up'].to_numpy(), plot_data['down'].to_numpy(),
                     color='b', alpha=.5)
    p1 = plt.plot(plot_data.index.to_numpy(), plot_data['prediction'].to_numpy(),color='b')
    p2 = plt.plot(plot_data.index.to_numpy(), plot_data['actual value'].to_numpy(),color='r')
    p3 = plt.fill(np.NaN, np.NaN, 'b', alpha=0.5)
    plt.xlabel('Timestamp')
    plt.ylabel('ORP (mV)')
    plt.legend([(p3[0], p1[0]), p2[0]], ['PCI','Recorded value'], loc='lower right')
    plt.show()


    # plot_data.plot()
    plt.fill_between(plot_data.index.to_numpy()[500:700], plot_data['up'].to_numpy()[500:700], plot_data['down'].to_numpy()[500:700],
                     color='b', alpha=.5)
    plt.plot(plot_data.index.to_numpy()[500:700], plot_data['prediction'].to_numpy()[500:700],color='b')
    plt.plot(plot_data.index.to_numpy()[500:700], plot_data['actual value'].to_numpy()[500:700],color='r')
    plt.xlabel('Timestamp')
    plt.ylabel('ORP (mV)')
    plt.legend([(p3[0], p1[0]), p2[0]], ['PCI','Recorded value'], loc='lower right')
    plt.show()
