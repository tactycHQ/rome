from DataLoader import DataLoader
from SequenceModel import SequenceModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import datetime as dt
from BuildModel import _WINDOW_SHIFT,_WINDOWSIZE

#Test Class

def main():
    ticker='test'
    d=DataLoader(ticker)
    d.loadData()
    d.prepData()
    window_size=_WINDOWSIZE
    window_shift=_WINDOW_SHIFT
    start_index=d.getIndex('11/10/2010')
    end_index=d.getIndex('11/30/2017')
    x_test,y_test,dates_test=createInputs(d.features,
                                           d.targets,
                                           d.dates,
                                           window_size,
                                           window_shift,
                                           start_index,
                                           end_index)


    aapl_model=SequenceModel()
    aapl_model.modelLoad("Data/"+ticker+'.h5',"Data/"+ticker+'_history.json')
    y_pred=aapl_model.predict_model(x_test)

    y_dates = d.dates[start_index+1:end_index]
    y_actuals = d.targets[start_index+1:end_index]
    y_actuals = d.denormalize(y_actuals, d.targets_mean, d.targets_std)

    plotTestPerformance(y_pred,y_actuals,y_dates,aapl_model.history,d.targets_std,window_size=window_size)


def createInputs(features, targets, dates, window_size, window_shift,start_index,end_index):
    """"
    window_size = number of days in lookback window. Also same as prediction window
    window_shift = how many days to shift between each sample
    predict_delay = how many days in the future to start prediction
    test_samples = number of test samples to generate. Each sample is 1 window of 30 days
    """
    inputs=[]
    outputs=[]
    target_dates = []
    print("Start Index: ", start_index)
    print("End Index: ", end_index)
    for i in range(start_index, end_index, window_shift):
        # print(i - window_size)
        # print("x_test in for loop ",features[i - window_size:i, :])
        inputs.append(features[i - window_size:i, :])
        outputs.append(targets[i + 1:i + 1 + window_size, :])
        target_dates.append(dates[i + 1:i + 1 + window_size, :])
    x_test = np.array(inputs)
    y_test = np.array(outputs)
    dates_test = np.array(target_dates)
    print("-----Creating Model Inputs-----")
    print("x_test Shape is ", x_test.shape)
    print("y_test Shape is ", y_test.shape)
    print("y_dates_test Shape is ", dates_test.shape)
    return x_test, y_test, dates_test

def plotPerformance(model,history,targets_std):
    loss = history['loss']
    val_loss = history['val_loss']
    print('Training loss (Denormalized)', loss[-1] * targets_std)
    print('Validation loss (Denormalized)', val_loss[-1]*targets_std)

    fig, axs =plt.subplots()
    axs.plot(range(1, len(loss) + 1), loss, 'bo', label='Training loss')
    axs.plot(range(1, len(loss) + 1), val_loss, 'b', label='Validation loss')
    axs.set_title('Training and validation loss')
    axs.legend()
    fig.autofmt_xdate()
    plt.show()

def plotTestPerformance(y_pred,y_actuals,y_dates,history,targets_std,window_size):

    loss = history['loss']
    val_loss = history['val_loss']
    print('Training loss (Denormalized)', loss[-1] * targets_std)
    print('Validation loss (Denormalized)', val_loss[-1]*targets_std)

    fig, axs =plt.subplots(2,1)
    axs[0].plot(range(1, len(loss) + 1), loss, 'bo', label='Training loss')
    axs[0].plot(range(1, len(loss) + 1), val_loss, 'b', label='Validation loss')
    axs[0].set_title('Training and validation loss')
    axs[0].legend()

    y_dates = y_dates.flatten()
    # Convert y_dates timed to datetime format
    for i in range (0,len(y_dates)):
        y_dates[i] = dt.datetime.strptime(y_dates[i], '%m/%d/%Y')


    axs[1].plot(y_dates, y_actuals, label='Actual Prices')
    for p in range(0, len(y_pred), window_size):
        y_pred_timed = y_pred[p, :, :].flatten()
        y_pred_timed = np.lib.pad(y_pred_timed,
                                  (p, max(0,len(y_dates) - p - window_size)),
                                  'constant',
                                  constant_values=np.NAN)
        y_pred_timed=y_pred_timed[:y_dates.shape[0]]
        axs[1].plot(y_dates, y_pred_timed, label='Predicted Prices')

    axs[1].set_title('Actuals vs Predicted')
    axs[1].legend(loc=(1.04,0))
    axs[1].fmt_xdata = mdates.DateFormatter('%m-%d-%Y')
    fig.autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    main()


