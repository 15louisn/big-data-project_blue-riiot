import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def plotf(event_data, id, var_of_interest, i):
    id = str(id)
    event_data = event_data.set_index('swimming_pool_id')
    event_data = event_data.loc[id]
    # Select a variable to study
    event_data = event_data.reset_index(drop=True)
    time_serie = event_data[["created",var_of_interest]]

    # print(time_serie.head())
    # print(time_serie.shape)
    time_serie.plot(y=var_of_interest)
    # number = (string) i
    name = str(i) +"_plot_"+var_of_interest+"_"+id+".pdf"
    plt.savefig(name)

def is_outlier(pred, perc, interval):
    if ((pred > (perc + interval)) or (pred < (perc - interval))):
        return 1
    else:
        return 0

def threshhold(time_serie, var_of_interest, alpha, use_abs):
    # # Convert timestamps
    time_serie.loc[:,"created"] = time_serie["created"].apply(pd.to_datetime)
    time_serie.loc[:,"created"] = time_serie["created"].apply(pd.Timestamp.timestamp)

    first_el = time_serie.iloc[0][var_of_interest]
    plot_data = pd.DataFrame(columns= ['actual value','prediction','up','down','outlier'])
    init = pd.Series([first_el,first_el,first_el,first_el,0]);
    plot_data = plot_data.append(init, ignore_index=True)

    for i in range(1,time_serie.shape[0]):
        perc = time_serie.loc[i][var_of_interest]
        prev = time_serie.loc[i-1][var_of_interest]

        if(use_abs):
            up = prev + alpha
            down = prev - alpha
            outlier = is_outlier(pred, perc, alpha)
        else:
            up = (1+alpha)*prev
            down = (1-alpha)*prev
            outlier = is_outlier(prev, perc, alpha*prev)

        # Update for plot
        new_s = pd.Series([perc, prev, up, down, outlier],
                         index = ['actual value','prediction','up','down','outlier'])
        plot_data = plot_data.append(new_s, ignore_index=True)

    print(plot_data['outlier'].value_counts())
    # plot_data.plot()
    outlying_points = time_serie.loc[plot_data['outlier']==1]
    # print(outlying_points)
    created = time_serie['created'].to_numpy()
    # print(time_serie['created'].to_numpy().shape, " ", plot_data.index.to_numpy())

    # p0 = plt.fill_between(created, plot_data['up'].to_numpy(), plot_data['down'].to_numpy(),
    #                  color='b', alpha=.5)
    # p4 = plt.scatter(outlying_points['created'].to_numpy(),outlying_points[var_of_interest].to_numpy(), color='g' ,zorder=10)
    #
    # p1 = plt.plot(created, plot_data['prediction'].to_numpy(),color='b',zorder=5)
    pF = plt.plot(created, plot_data['actual value'].to_numpy(),color='b',zorder=1)

    p2 = plt.scatter(created, plot_data['actual value'].to_numpy(),color='r',zorder=5, s=1)
    # p3 = plt.fill(np.NaN, np.NaN, 'b', alpha=0.5)

    plt.xlabel('Timestamp')
    plt.ylabel('Conductivity (mS)')
    plt.ylabel('pH')

    # plt.legend([(p3[0], p1[0]), p2], ['Interval','Recorded value'], loc='upper right')
    plt.show()
    # plt.savefig("r4_baseline2.pdf")
    # # plot_data.plot()
    # plt.fill_between(created[2400:2850], plot_data['up'].to_numpy()[2400:2850], plot_data['down'].to_numpy()[2400:2850],
    #                  color='b', alpha=.5)
    # plt.plot(created[2400:2850], plot_data['prediction'].to_numpy()[2400:2850],color='b')
    # plt.plot(created[2400:2850], plot_data['actual value'].to_numpy()[2400:2850],color='r')
    # plt.xlabel('Timestamp')
    # plt.ylabel('ORP (mV)')
    # plt.legend([(p3[0], p1[0]), p2[0]], ['PCI','Recorded value'], loc='lower right')
    # plt.show()

def threshhold_test(time_serie, alpha=0.2, use_abs=True):
    # # Convert timestamps
    # time_serie.loc[:,"created"] = time_serie["created"].apply(pd.to_datetime)
    # time_serie.loc[:,"created"] = time_serie["created"].apply(pd.Timestamp.timestamp)
    # print(time_serie)
    time_serie = time_serie.to_numpy()
    first_el = time_serie[0]
    # plot_data = pd.DataFrame(columns= ['actual value','prediction','up','down','outlier'])
    # init = pd.Series([first_el,first_el,first_el,first_el,0]);
    # plot_data = plot_data.append(init, ignore_index=True)
    outlier_data = np.zeros(time_serie.shape[0])
    outlier_data[0] = 0
    for i in range(1,time_serie.shape[0]):
        perc = time_serie[i]
        prev = time_serie[i-1]

        if(use_abs):
            up = prev + alpha
            down = prev - alpha
            outlier = is_outlier(prev, perc, alpha)
        else:
            up = (1+alpha)*prev
            down = (1-alpha)*prev
            outlier = is_outlier(prev, perc, alpha*prev)

        # Update for plot
        # new_s = pd.Series([perc, prev, up, down, outlier],
        #                  index = ['actual value','prediction','up','down','outlier'])
        # plot_data = plot_data.append(new_s, ignore_index=True)
        outlier_data[i] = outlier
    # print(plot_data['outlier'].value_counts())
    # plot_data.plot()
    # outlying_points = time_serie.loc[plot_data['outlier']==1]
    # print(outlying_points)
    # created = time_serie['created'].to_numpy()
    # print(time_serie['created'].to_numpy().shape, " ", plot_data.index.to_numpy())
    # return plot_data['outlier']
    return outlier_data

if __name__ == "__main__":
    event_data = pd.read_csv('reduced_event.csv')

    random.seed(0)

    # Pool ranking code for visualisation
    pool_ranking = event_data['swimming_pool_id'].value_counts().index.values
    print(pool_ranking[0:20])

    # for i in range(40,50):
    #     plotf(event_data, pool_ranking[i], "data_conductivity",i)
    #     plotf(event_data, pool_ranking[i], "data_ph",i)
    # plotf(event_data, '009d1793-0dd1-4e47-a256-0f06a485daf0', "data_conductivity",'X')

    event_data = event_data.set_index('swimming_pool_id')
    # event_data = event_data.loc['f20bfa78-3472-436a-911e-f2b965255f2f']
    event_data = event_data.loc['f20bfa78-3472-436a-911e-f2b965255f2f']

    var_of_interest = "data_ph"
    event_data = event_data.reset_index(drop=True)
    time_serie = event_data[["created",var_of_interest]]

    threshhold(time_serie, var_of_interest, 0.02, 0)
