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
        self.history=None

    def build_model(self,x_train,y_train,batch_size=32,epochs=20):
        timesteps, dim = x_train.shape[1],x_train.shape[2]
        self.model.add(layers.GRU(32,return_sequences=True,input_shape=(timesteps,dim)))
        self.model.add(layers.Dense(1))
        print(self.model.summary())
        self.model.compile(optimizer='adam',loss='mae')
        self.history=self.model.fit(x_train,y_train,batch_size=32,validation_split=0.2,verbose=1,epochs=epochs)
        return self.model,self.history

    def predict_model (self,x_test):
        y_pred=self.model.predict(x_test)
        return y_pred

    def modelSave (self,fname,hname):
        self.model.save(fname)
        history_dict=self.history.history
        with open(hname,'w') as f:
            json.dump(history_dict,f)

    def modelLoad (self,model_filename,hist_filename):
        self.model=load_model(model_filename)
        with open(hist_filename,'r') as f:
            self.history=json.load(f)


    def plotPerformance(self,targets_std):
        loss = self.history['loss']
        val_loss = self.history['val_loss']
        print('Training loss (Denormalized)', loss[-1] * targets_std)
        print('Validation loss (Denormalized)', val_loss[-1]*targets_std)

        fig, axs =plt.subplots()
        axs.plot(range(1, len(loss) + 1), loss, 'bo', label='Training loss')
        axs.plot(range(1, len(loss) + 1), val_loss, 'b', label='Validation loss')
        axs.set_title('Training and validation loss')
        axs.legend()
        fig.autofmt_xdate()
        plt.show()