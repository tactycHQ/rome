from DataLoader import DataLoader
from SequenceModel import SequenceModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates

#Build Class

_WINDOWSIZE=60
_WINDOW_SHIFT=1

def main():
    ticker='test'
    d=DataLoader(ticker)
    d.loadData()
    d.prepData()
    start_index=d.getIndex('3/9/2009')
    end_index=d.getIndex('11/10/2015')

    x_train,y_train,dates_train=createInputs(d.features,
                                             d.targets,
                                             d.dates,
                                             _WINDOWSIZE,
                                             _WINDOW_SHIFT,
                                             start_index,
                                             end_index)
    aapl_model=SequenceModel()
    aapl_model.build_model(x_train,y_train)
    aapl_model.modelSave("Data/"+ticker+'.h5',"Data/"+ticker+'_history.json')
    # aapl_model.modelLoad('AAPL.h5', 'AAPL_history.json')
    plotPerformance(aapl_model,aapl_model.history_dict, d.targets_std)



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
        inputs.append(features[i - window_size:i, :])
        outputs.append(targets[i + 1:i + 1 + window_size, :])
        target_dates.append(dates[i + 1:i + 1 + window_size, :])
    x_train = np.array(inputs)
    y_train = np.array(outputs)
    dates_train = np.array(target_dates)
    print("-----Creating Model Inputs-----")
    print("x_train Shape is ", x_train.shape)
    print("y_train Shape is ", y_train.shape)
    print("y_dates_train Shape is ", dates_train.shape)
    return x_train, y_train, dates_train

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


if __name__ == '__main__':
    main()


