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


def main():
    # aapl_data=getData("AAPL",)
    # saveData(aapl_data)
    raw_data=loadData('GE.csv')
    raw_data=raw_data.iloc[::-1]
    features, targets, dates, features_mean, featured_std, targets_mean, targets_std=PrepData(raw_data)
    print('Target Mean: ',targets_mean)
    print('Target Std: ', targets_std)
    x_train, y_train, y_dates_train, x_test, y_test, y_dates_test, window_size=createInputs(features,targets,dates)

    # model,history=build_model(x_train,y_train)
    # modelSave(model,history)
    model,history=modelLoad('rome.h5','history.json')

    y_pred=predict_model(model,x_test)
    y_test=denormalize(y_test,targets_mean,targets_std)
    y_pred = denormalize(y_pred, targets_mean, targets_std)

    plotPerformance(history,y_test,y_pred,y_dates_test,targets_std, window_size)

def getData(ticker):
    start=datetime(2013,2,9)
    end = datetime(2017, 5, 9)
    stock_data = get_historical_data(ticker,start=start,end=end,output_format='pandas')
    print(stock_data.describe)
    return stock_data

def saveData(dataframe):
    dataframe.to_csv('out.csv')
    print("Saved")

def loadData(filename):
    raw_data = pd.read_csv(filename)
    print("------Loaded Raw Data-------")
    return raw_data

def PrepData(df):

    feature_set=['close','volume']
    targets=df['close'].values.reshape(-1,1)
    dates=df['date'].values.reshape(-1,1)
    features=pd.DataFrame(df,columns=feature_set).values

    features,features_mean,featured_std=normalize(features)
    targets, targets_mean, targets_std = normalize(targets)

    print("Feature Shape is ", features.shape)
    print("Target Shape is ", targets.shape)
    print("Dates Shape is ", dates.shape)

    return features,targets,dates,features_mean,featured_std,targets_mean,targets_std

def normalize(raw_input):
    mean=raw_input.mean(axis=0)
    std=raw_input.std(axis=0)
    norm_output=(raw_input-mean)/std
    return norm_output,mean,std

def denormalize(norm_input,mean,std):
    denorm=norm_input*std+mean
    return denorm

def createInputs(features,targets,dates,window_size=60,window_shift=1,predict_delay=1,test_samples=61):
    #window_size = number of days in lookback window. Also same as prediction window
    #window_shift = how many days to shift between each sample
    #predict_delay = how many days in the future to start prediction
    #test_samples = number of test samples to generate. Each sample is 1 window of 30 days

    inputs=[]
    outputs=[]
    sampled_dates=[]
    start=window_size
    end=len(features)-window_size
    print("Start: ",start)
    print("End: ",end)

    for i in range(start,end,window_shift):
        inputs.append(features[i-window_size:i,:])
        outputs.append(targets[i+predict_delay:i+predict_delay+window_size, :])
        sampled_dates.append(dates[i+predict_delay:i+predict_delay+window_size, :])

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    sampled_dates= np.array(sampled_dates)

    test_index = len(inputs) - test_samples

    x_train=inputs[:test_index,:,:]
    y_train = outputs[:test_index,:,:]
    y_dates_train = sampled_dates[:test_index, :,:]
    x_test = inputs[test_index:,:,:]
    y_test= outputs[test_index:,:,:]
    y_dates_test = sampled_dates[test_index:,:,:]

    print("-----Creating Model Inputs-----")
    print("Test Index: ", test_index)
    print("Inputs Shape is ", inputs.shape)
    print("Outputs Shape is ", outputs.shape)
    print("Sampled_Dates Shape is ", sampled_dates.shape)
    print("Test Index: ",test_index)
    print("Inputs Shape is ", inputs.shape)
    print("Outputs Shape is ", outputs.shape)
    print("Sampled_Dates Shape is ", sampled_dates.shape)
    print("x_train Shape is ", x_train.shape)
    print("y_train Shape is ", y_train.shape)
    print("y_dates_train Shape is ", y_dates_train.shape)
    print("x_test Shape is ", x_test.shape)
    print("y_test Shape is ", y_test.shape)
    print("y_dates_test Shape is ", y_dates_test.shape)

    return x_train, y_train, y_dates_train, x_test, y_test, y_dates_test, window_size

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

    # y_pred_timed = y_pred_timed.flatten()
    # y_pred_timed = y_pred[::predict_intervals]
    # y_pred_timed[:,30:,:]=np.NaN
    # print("y_pred_timed Shape is ", y_pred_timed.shape)
    # print("y_pred_timed Shape is ", y_pred_timed.shape)

    fig, axs =plt.subplots(2,1)

    axs[0].plot(range(1, len(loss) + 1), loss, 'bo', label='Training loss')
    axs[0].plot(range(1, len(loss) + 1), val_loss, 'b', label='Validation loss')
    axs[0].set_title('Training and validation loss')
    axs[0].legend()

    axs[1].plot(y_dates_timed,y_test_timed, label='Actual Prices')

    for p in range(0,len(y_pred)-window_size):
        y_pred_timed=y_pred[p,:,:].flatten()
        y_pred_timed = np.lib.pad(y_pred_timed,(p,len(y_pred)-window_size-p), 'constant', constant_values=np.NAN)
        print(y_pred_timed.shape)
        axs[1].plot(y_dates_timed, y_pred_timed,label='Predicted Prices')

    axs[1].set_title('Actuals vs Predicted')
    axs[1].legend()
    axs[1].fmt_xdata=mdates.DateFormatter('%m-%d-%Y')
    fig.autofmt_xdate()

    plt.show()


if __name__ == '__main__':
    main()