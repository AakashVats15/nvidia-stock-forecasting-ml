from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

def linear():
    return LinearRegression()

def ridge():
    return Ridge(alpha=1.0)

def lasso():
    return Lasso(alpha=0.001)

def elastic():
    return ElasticNet(alpha=0.001, l1_ratio=0.5)