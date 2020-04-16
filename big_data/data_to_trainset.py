import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

import outlier_detect_r4 as louis
from sklearn.metrics import accuracy_score, confusion_matrix

data_path = '/Users/gigi/Documents/github/big-data-project_blue-riiot/big_data/labeled_data_v2_pH/'
# data_path = '/labeled_data_v2_pH/'

def export_data():
    events_data = pd.read_csv('reduced_event.csv')

    pool_ranking = events_data['swimming_pool_id'].value_counts().index.values
    # print(pool_ranking)

    events_data = events_data.set_index('swimming_pool_id')
    var_of_interest = "data_ph"

    # for i in range(len(pool_ranking)):
    for i in range(10,15):
        event_data = events_data.loc[pool_ranking[i]].reset_index(drop=True)

        time_serie = event_data[["created",var_of_interest]]

        size = time_serie.shape[0]
        print(size)
        indexes = range(size)
        filename = "pool_" + str(i) + "_" + pool_ranking[i][0:10] + "_" + var_of_interest
        filename_s = pd.Series(filename, index= indexes)
        out = pd.Series(np.zeros(size))

        plot_data = pd.concat([filename_s.rename('filename'),
                                time_serie["created"].rename('timestamp'),
                                time_serie[var_of_interest].rename('value'),
                                out.rename('label')], axis=1)

        plot_data.to_csv(path_or_buf= filename+".csv",index=False)


def test(algo):
    # Algo should take in a time serie and output the
    conf_matrix = np.ndarray(shape=(2,2))
    files = [f for f in glob.glob(data_path + "*.csv")]
    # print(files)
    # for filename in glob.glob(os.path.join(data_path, '*.csv')):
    for filename in files:
       # print(filename)
       labeled_data = pd.read_csv(filename)
      # print(labeled_data.drop.values)
       output = algo(labeled_data[['value']])
       # print(output.value_counts())
       # print(labeled_data['label'].value_counts())

       # missing_rate.append(accuracy_score(labeled_data['label'].to_numpy(), output))
       conf_matrix += confusion_matrix(labeled_data['label'].to_numpy(), output)
    print(conf_matrix)
    print(conf_matrix/np.sum(conf_matrix))
    # print("Average missing rate: ", missing_rate.mean())

def visusalize():
    events_data = pd.read_csv('reduced_event.csv')

    pool_ranking = events_data['swimming_pool_id'].value_counts().index.values
    # print(pool_ranking)

    events_data = events_data.set_index('swimming_pool_id')

    for i in range(10,15):
        event_data = events_data.loc[pool_ranking[i]].reset_index(drop=True)

        data = event_data[["created","data_ph","data_temperature","data_orp","data_conductivity"]]
        datetime_index = pd.DatetimeIndex(data["created"].values)
        data=data.set_index(datetime_index)
        # data = data.drop("created")
        pF = data.plot(style='.-')
        plt.show()
    # (created, plot_data['actual value'].to_numpy(),color='b',zorder=1)


if __name__ == "__main__":
    # export_data()

    # visusalize()

    test(louis.threshhold_test)
