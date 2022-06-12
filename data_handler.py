import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from math import ceil


def data_per_split(data, perc):
    part_1 = data.iloc[:int(len(data)*(perc)), :]
    part_2 = data.iloc[int(len(data)*(perc)):, :]
    return part_1, part_2

class Data:
    def __init__(self, lags, batch_s, data_s=0, features=None):
        if data_s == 0:
            self.data = pd.read_csv("p_apple2.csv")
        elif data_s == 1:
            self.data = pd.read_csv("p_amzn2.csv")
        elif data_s == 2:
            self.data = pd.read_csv("p_tsla2.csv")
        self.data.set_index(['Date'],inplace=True)
        self.lags = lags
        self.batch_s = batch_s        

        if features:
            self.data = self.data.iloc[:,features]
        self.no_features = self.data.shape[1]

        self.train_test = data_per_split(self.data, 0.2)[1]
        self.scaler = MinMaxScaler()
        self.scaler.fit(data_per_split(self.data, 0.8)[0])
        self.scaled_data = self.scaler.transform(self.train_test)

        self.feat = self.scaled_data
        self.target = self.scaled_data[:,0]

        self.XXyy = train_test_split(self.feat, self.target, test_size = 0.2, random_state = 1, shuffle=False)
        self.no_bs = ceil((self.XXyy[0].shape[0]-self.lags)/self.batch_s)
        self.train_generator = TimeseriesGenerator(self.XXyy[0], self.XXyy[2], length=lags, sampling_rate=1, batch_size=batch_s)
        self.test_generator = TimeseriesGenerator(self.XXyy[1], self.XXyy[3], length=lags, sampling_rate=1, batch_size=batch_s)
        
    def get_predictions_df(self, model, train=False):
        if train:
            y_pred = model.predict_generator(self.train_generator)
            df_pred = pd.concat([pd.DataFrame(y_pred), pd.DataFrame(self.XXyy[0][:,1:][self.lags:])], axis=1)
            inv_df_pred = self.scaler.inverse_transform(df_pred)
            df_final = self.data[df_pred.shape[0]*-1:]
            df_final["Close_pred"] = inv_df_pred[:,0]
            return df_final
        else:
            y_pred = model.predict_generator(self.test_generator)
            df_pred = pd.concat([pd.DataFrame(y_pred), pd.DataFrame(self.XXyy[1][:,1:][self.lags:])], axis=1)
            inv_df_pred = self.scaler.inverse_transform(df_pred)
            df_final = self.data[df_pred.shape[0]*-1:]
            df_final["Close_pred"] = inv_df_pred[:,0]
            return df_final

    def plot_test_pred(self, model):
        SMALL_SIZE = 20
        MEDIUM_SIZE = 26
        BIGGER_SIZE = 30
        plt.rc('font', size=SMALL_SIZE)  
        plt.rc('axes', titlesize=BIGGER_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=MEDIUM_SIZE)
        plt.rc('legend', fontsize=BIGGER_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)
        df_final = self.get_predictions_df(model)
        plt.figure(figsize=(16, 9))
        plt.plot(df_final["Close"], marker = "s", linewidth=0.75, color='red')
        plt.plot(df_final["Close_pred"], color="b", marker = "s", linewidth=0.75)
        plt.xticks(df_final.index[::25],  rotation='horizontal')
        plt.legend(["Close Test", "Predicted Test"])
        plt.grid()
    
    def print_test_metrics(self, model):
        df_final = self.get_predictions_df(model)
        mae = metrics.mean_absolute_error(df_final["Close"], df_final["Close_pred"])
        mse = metrics.mean_squared_error(df_final["Close"], df_final["Close_pred"])
        rmse = np.sqrt(mse)
        mape = metrics.mean_absolute_percentage_error(df_final["Close"], df_final["Close_pred"])
        print(f"& \\hfil {mae:.4f} & \\hfil {mse:.4f} & \\hfil {rmse:.4f} & \\hfil {mape:.4f}")
