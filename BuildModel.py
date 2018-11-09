from DataLoader import DataLoader
import SequenceModel as sq
import numpy as np

#Build Class

def main():
    ticker='AAPL'
    d=DataLoader(ticker)
    d.loadData()
    d.prepData()

    window_size=60
    window_shift=1
    start_date=d.getIndex('11/10/2008')
    end_date=d.getIndex('11/10/2015')

        x_train,y_train,dates_train=createInputs(d.features,
                                             d.targets,
                                             d.dates,
                                             window_size,
                                             window_shift,
                                             start_date,
                                             end_date)

    aapl_model=SequenceModel()
    aapl_model.build_model(x_train,y_train)
    modelSave(aapl_model.model,aapl_model.history)

def modelSave (model,history):
    model.save(ticker+'.h5')
    history_dict=history
    with open(ticker+'history.json','w') as f:
        json.dump(history_dict,f)

def createInputs(features, targets, dates, window_size, window_shift,start_date,end_date):
    # window_size = number of days in lookback window. Also same as prediction window
    # window_shift = how many days to shift between each sample
    # predict_delay = how many days in the future to start prediction
    # test_samples = number of test samples to generate. Each sample is 1 window of 30 days

    inputs = []
    outputs = []
    sampled_dates = []
    start_index = getIndex(start_date)
    end_index = getIndex(end_date)
    print("Start: ", start_index)
    print("End: ", end_index)

    for i in range(start_index, end_index, window_shift):
        inputs.append(features[i - window_size:i, :])
        outputs.append(targets[i + 1:i + 1 + window_size, :])
        sampled_dates.append(dates[i + 1:i + 1 + window_size, :])

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    sampled_dates = np.array(sampled_dates)

    x_train = inputs[:test_index, :, :]
    y_train = outputs[:test_index, :, :]
    dates_train = sampled_dates[:test_index, :, :]

    print("-----Creating Model Inputs-----")
    print("Test Index: ", test_index)
    print("Inputs Shape is ", inputs.shape)
    print("Outputs Shape is ", outputs.shape)
    print("Sampled_Dates Shape is ", sampled_dates.shape)
    print("Test Index: ", test_index)
    print("Inputs Shape is ", inputs.shape)
    print("Outputs Shape is ", outputs.shape)
    print("Sampled_Dates Shape is ", sampled_dates.shape)
    print("x_train Shape is ", x_train.shape)
    print("y_train Shape is ", y_train.shape)
    print("y_dates_train Shape is ", y_dates_train.shape)
    print("x_test Shape is ", x_test.shape)
    print("y_test Shape is ", y_test.shape)
    print("y_dates_test Shape is ", y_dates_test.shape)

    return x_train, y_train, y_dates_train


if __name__ == '__main__':
    main()


