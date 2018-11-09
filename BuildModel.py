from DataLoader import DataLoader
from SequenceModel import SequenceModel
import numpy as np

#Build Class

def main():
    ticker='AAPL'
    d=DataLoader(ticker)
    d.loadData()
    d.prepData()
    window_size=60
    window_shift=1
    start_index=d.getIndex('3/10/2009')
    end_index=d.getIndex('11/10/2015')

    x_train,y_train,dates_train=createInputs(d.features,
                                             d.targets,
                                             d.dates,
                                             window_size,
                                             window_shift,
                                             start_index,
                                             end_index)
    aapl_model=SequenceModel()
    # aapl_model.build_model(x_train,y_train)
    # aapl_model.modelSave('AAPL.h5','AAPL_history.json')
    aapl_model.modelLoad('AAPL.h5','AAPL_history.json')
    aapl_model.plotPerformance(d.targets_std)


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


if __name__ == '__main__':
    main()


