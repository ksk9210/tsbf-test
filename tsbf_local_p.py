# -*- coding: utf-8 -*-
from selenium import webdriver
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import math
import time
import numpy
import pandas
import pybitflyer
import os

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM


class load_bitflyer:
    def gettable(interval,fromDate,toDate):
        name = os.path.dirname(os.path.abspath(__file__))
        joined_path = name + '\PhantomJS\\'+'bin\phantomjs.exe'
        #print(joined_path)
        driver = webdriver.PhantomJS(executable_path=joined_path)
        
#        url_base="https://bitcoincharts.com/charts/bitflyerJPY#rg10zig5-minztgCzm1g10zm2g25zv"

        url_base="https://bitcoincharts.com/charts/bitflyerJPY#rg10zig"+str(interval)+"-minzczsg"+str(fromDate)+"zeg"+str(toDate)+"ztgSzm1g10zm2g25zv"

        #print(url_base)
        driver.get(url_base)
        driver.find_element_by_link_text(u"Load raw data").click()
        time.sleep(5)

        try:
            html=driver.page_source
        except urllib.error.HTTPError:
            print ("e")
        else:
            bsObj = BeautifulSoup(html, "html.parser")
            
            lists = []
            table = bsObj.findAll("table", {"id":"chart_table"})[0]
            rows = table.findAll("tr")

            try:
                for row in rows:
                    listData = []
                    for cell in row.findAll("td"):
                        listData.append(cell.get_text())
#                    print(listData)
                    lists.append(listData)
            finally:
                print("Done scraping")

        driver.quit()
        return lists


class Prediction :

    def __init__(self):
        self.length_of_sequences = 24
        self.in_out_neurons = 1
        self.hidden_neurons = 300


    def load_data(self, data, n_prev):
        X, Y = [], []
        for i in range(len(data) - n_prev):
          X.append(data.iloc[i:(i+n_prev)].as_matrix())
          Y.append(data.iloc[i+n_prev].as_matrix())
        retX = numpy.array(X)
        retY = numpy.array(Y)
        return retX, retY


    def create_model(self) :
        model = Sequential()
        model.add(LSTM(self.hidden_neurons, \
                  batch_input_shape=(None, self.length_of_sequences, self.in_out_neurons), \
                  return_sequences=False))
        model.add(Dense(self.in_out_neurons))
        model.add(Activation("linear"))
        model.compile(loss="mape", optimizer="adam")
        return model

    def train(self, X_train, y_train) :
        model = self.create_model()
        # 学習
        model.fit(X_train, y_train, batch_size=24, nb_epoch=20)
        return model

class buySellBTCJPY :
    def __init__(self):
        self.api = pybitflyer.API(api_key="XXXXXXXXXXXXXXXXXXXX",api_secret="XXXXXXXXXXXXXXXXXXXXXXXX")
        self.refresh()
        
    def judge(self,currentPrice,predictedPrice,actualPrice) :
        if currentPrice > predictedPrice:
            self.sell(actualPrice)
            return "sell"
        
        if currentPrice < predictedPrice:
            self.buy(actualPrice)
            return "buy"
            
    def sell(self, price) :
        amt = round(self.BTCwallet,3)
        print(amt)
        sell_btc = self.api.sendchildorder(product_code="BTC_JPY",
                                            child_order_type="MARKET",
                                            side="SELL",
                                            size=amt,
                                            minute_to_expire=10,
                                            time_in_force="GTC")
        print(sell_btc)
        self.refresh()
        
    def buy(self, price) :
        amt = round(self.JPYwallet/price*0.99,3)
        print(amt)
        buy_btc = self.api.sendchildorder(product_code="BTC_JPY",
                                            child_order_type="MARKET",
                                            side="BUY",
                                            size=amt,
                                            minute_to_expire=10,
                                            time_in_force="GTC")
        print(buy_btc)
        self.refresh()
        
    def refresh(self):
        BTC = self.api.getbalance(product_code="BTC_JPY")
        for btci in BTC:
            if btci['currency_code']=='BTC':
                self.BTCwallet = btci['amount']-0.314
            if btci['currency_code']=='JPY':
                self.JPYwallet = btci['amount']
                
    def exit(self, price):
        self.JPYwallet = self.JPYwallet + self.BTCwallet * price
        self.BTCwallet = 0
        
#def lambda_handler(event, context):
def task():
    prediction = Prediction()
    BTCJPY = buySellBTCJPY()
    print(BTCJPY.BTCwallet)
    print(BTCJPY.JPYwallet)

    # データ準備
    data = None

    #bitflyerのログ読み込み
    interval = 5
    today = datetime.date.today()
    toDate = today + timedelta(days=1)
    fromDate = today + timedelta(days=-13)
    bitflyerLog = load_bitflyer.gettable(5,fromDate,toDate)

    #pandasに変換して進める
    data = pandas.DataFrame(bitflyerLog)
    data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volumeBTC', 'volumeCUR', 'weightedPrice']
    data['datetime'] = pandas.to_datetime(data['datetime'])

    #データクリーニング・不要な行の削除
    data = data.dropna()
    data = data[data['close'] != '—']

    # 終値のデータを標準化
    originalData = pandas.DataFrame(data['close'])
    originalData.columns = ['close']
    data['close'] = preprocessing.scale(data['close'])
    data = data.sort_values(by='datetime')
    data = data.reset_index(drop=True)
    data = data.loc[:, ['datetime', 'close']]

    x_train, y_train = prediction.load_data(data[['close']].iloc[0:len(data)], prediction.length_of_sequences)

    # 学習
    model = prediction.train(x_train, y_train)

    # 将来予測
    future_result = numpy.empty((0)) #将来予測自体を格納する配列
    batch_result = pandas.DataFrame(columns=['datetime', 'close']) #temp用のDataFrameを作成
    batch_result.loc[0,['datetime']]= data[['datetime']].iloc[len(data)-1] #temp用のDataFrameのdatetimeの初期値を入力
    batch_result['datetime'] = pandas.to_datetime(batch_result['datetime']) #temp用のDataFrameのdatetimeの初期値を型変換
    
    predict_length = int(5/5) #予測の長さ(分)/5分
    # 以下、繰り返しで予測
    for step in range(predict_length):
        x_ftest,  y_ftest  = prediction.load_data(data[['close']].iloc[len(data[['close']])-prediction.length_of_sequences*10:], prediction.length_of_sequences)
        
        batch_predict = model.predict(x_ftest)
        batch_result.loc[0,['datetime']] = batch_result.loc[0,['datetime']]+timedelta(minutes=5)
        batch_result.loc[0,['close']] = batch_predict[len(batch_predict)-1]
        data = pandas.concat([data, batch_result])
        #print(batch_result)
        future_result = numpy.append(future_result, batch_predict[len(batch_predict)-1])

    # 財布の状況を再取得
    BTCJPY.refresh()
    # 判断してアクション
    action = BTCJPY.judge(
        float(data[['close']].iloc[len(originalData)-1]),
        float(future_result[predict_length-1]),
        float(originalData[['close']].iloc[len(originalData)-1])) #ジャッジ
    print(action
          +", current="+str(float(data[['close']].iloc[len(originalData)-1]))
          +", predicted="+str(future_result[predict_length-1])
          +", BTCwallet="+str(BTCJPY.BTCwallet)
          +", JPYwallet="+str(BTCJPY.JPYwallet)
          +", Price="+str(float(originalData[['close']].iloc[len(originalData)-1]))
          +", Value" +str(float(BTCJPY.JPYwallet+BTCJPY.BTCwallet*float(originalData[['close']].iloc[len(originalData)-1]))))

if __name__ == "__main__":
    while True:
        task()