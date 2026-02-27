from statsmodels.tsa.arima.model import ARIMA

def arima(order=(1,0,1)):
    def model(y):
        return ARIMA(y, order=order).fit()
    return model