from DataLoader import DataLoader

ticker='AAPL'
d=DataLoader(ticker)
# stock_data = d.getData()
d.loadData()
d.PrepData()

