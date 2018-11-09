#Author: Anubhav Srivastava
#License: MIT License

from iexfinance import get_historical_data
import datetime as dt
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model, save_model
from keras import layers
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import json


class SequenceModel():

    def __init__(self):
        self.model=Sequential()



    def build_model(x_train,y_train,batch_size=32,epochs=20):

        timesteps=x_train.shape[1]
        dim=x_train.shape[2]

        model=Sequential()
        model.add(layers.GRU(32,return_sequences=True,input_shape=(timesteps,dim)))
        model.add(layers.Dense(1))

        print(model.summary())
        model.compile(optimizer='adam',loss='mae')

        history=model.fit(x_train,y_train,batch_size=32,validation_split=0.2,verbose=1,epochs=epochs)

        return model,history.history

    def predict_model (model,x_test):

        y_pred=model.predict(x_test)
        return y_pred

    def modelSave (model,history):

        model.save('rome.h5')
        history_dict=history
        with open('history.json','w') as f:
            json.dump(history_dict,f)

    def modelLoad (model_filename='rome.h5',hist_filename='history.h5'):

        model=load_model(model_filename)
        with open(hist_filename,'r') as f:
            history=json.load(f)

        return model,history

    def plotPerformance(history,y_test,y_pred,y_dates_test,targets_std,window_size,predict_intervals=10):

        loss = history['loss']
        val_loss = history['val_loss']

        print('Training loss (Denormalized)', loss[-1] * targets_std)
        print('Validation loss (Denormalized)', val_loss[-1]*targets_std)

        y_test_timed = y_test[::window_size]
        y_dates_timed= y_dates_test[::window_size]

        print("-----Creating Plotting Vectors-----")
        print("y_test_timed Shape is ", y_test_timed.shape)
        print("y_dates_timed Shape is ", y_dates_timed.shape)

        y_test_timed = y_test_timed.flatten()
        y_dates_timed = y_dates_timed.flatten()
        # Convert y_dates timed to datetime format
        for i in range(0, len(y_dates_timed)):
            y_dates_timed[i] = dt.datetime.strptime(y_dates_timed[i], '%m/%d/%Y')

        print("-----Flattened Plotting Vectors-----")
        print("y_test_timed Shape is ", y_test_timed.shape)
        print("y_dates_timed Shape is ", y_dates_timed.shape)

        print("y_pred Shape is ", y_pred.shape)

        fig, axs =plt.subplots(2,1)

        axs[0].plot(range(1, len(loss) + 1), loss, 'bo', label='Training loss')
        axs[0].plot(range(1, len(loss) + 1), val_loss, 'b', label='Validation loss')
        axs[0].set_title('Training and validation loss')
        axs[0].legend()

        axs[1].plot(y_dates_timed,y_test_timed, label='Actual Prices')

        for p in range(0,len(y_pred),window_size):
            print(p)
            y_pred_timed=y_pred[p,:,:].flatten()
            y_pred_timed = np.lib.pad(y_pred_timed,(p,len(y_dates_timed)-p-window_size), 'constant', constant_values=np.NAN)
            axs[1].plot(y_dates_timed, y_pred_timed,label='Predicted Prices')

        axs[1].set_title('Actuals vs Predicted')
        axs[1].legend()
        axs[1].fmt_xdata=mdates.DateFormatter('%m-%d-%Y')
        fig.autofmt_xdate()

        plt.show()
