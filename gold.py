from datetime import datetime
from sklearn import linear_model

# Read in prices from file
def getData(datefile, pricefile):
    dfile = open(datefile, 'r')
    dates = dfile.read().split()
    dfile.close()
    pfile = open(pricefile, 'r')
    prices = pfile.read().split()
    pfile.close()
    for i in range(len(dates)):
        dates[i] = datetime.strptime(dates[i], '%m/%d/%y')
        prices[i] = float(prices[i])
    return (dates, prices)

# Calculate trendline - return a from linear equation y = ax + b
def calcTrendline(prices, i, n):
    points = prices[i-n+1:i+1]
    indices = [[index] for index in range(i-n+1, i+1)]

    # Do linear regression
    clf = linear_model.LinearRegression()
    clf.fit(indices, points)
    a = clf.coef_[0]
    b = clf.intercept_
    p = clf.predict([[i+1]])
    return a

# Momentum for day i with period n
def calcMomentum(prices, i, n):
    return prices[i] - prices[i-n]

# Rate of change for day i with period n
def calcROC(prices, i, n):
    return (prices[i] - prices[i-n]) / prices[i-n]

# Calculate %K (helper for stochastic oscillator)
def calcK(prices, i, n):
    high = max(prices[i-n:i+1])
    low = min(prices[i-n:i+1])
    return (prices[i] - low) / (high - low) * 100

# Return stochastic signal on day i with period n
def calcStochastic(prices, i, n):
    curr_K = calcK(prices, i, n)
    prev_K = calcK(prices, i-1, n)
    curr_D = sum([calcK(prices, i-j, n) for j in range(3)])/3
    prev_D = sum([calcK(prices, i-j-1, n) for j in range(3)])/3
    if prev_D < 20 and curr_K < 20 and prev_K < prev_D \
                   and curr_D < curr_K: return "Buy"
    if prev_D > 80 and curr_K > 80 and prev_K > prev_D \
                   and curr_D > curr_K: return "Sell"
    return "Hold"

# Make feature vector for day i
def makeFeatures(prices, i):
    x = []
    y_classify = 1 if prices[i+1] > prices[i] else 0
    y_regression = (prices[i+1] - prices[i]) / prices[i] * 100

    # Add trendlines
    arr = [2, 3, 5, 8, 15, 17]
    for n in arr:
        x = x + [calcTrendline(prices, i, n), calcTrendline(prices, i-1, n)]
    
    # Calculate momentum/ROC
    curr_roc = calcROC(prices, i, 1)
    x = x + [curr_roc]
    for n in arr:
        roc = calcROC(prices, i, n)
        x = x + [roc, curr_roc/roc if roc != 0 else curr_roc]

    # Add stochastic %K
    x = x + [calcK(prices, i, 14)]
    return (x, y_classify, y_regression)
