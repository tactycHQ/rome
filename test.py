from DataLoader import DataLoader
import BuildModel

#Test Class

def main()
    ticker='AAPL'
    d=DataLoader(ticker)
    d.loadData()
    d.PrepData()
    createInputs()
    buildModel()





if __name__ = '__main__':
    main()

