import gold, stock, numpy as np
from sklearn import linear_model, cross_validation

def fullFeatureSet():
    uv = ['GSPC.csv', 'IXIC.csv', 'DJI.txt', 'N100.csv', 'N225.csv', \
          'HSI.csv', 'FCHI.csv', 'FTSE.csv', 'GSPTSE.csv', 'KS11.csv', \
          'SSMI.csv']
    nv = ['goldFutures', 'silverFutures', 'copperFutures', 'oilFutures', \
          'dollar', 'eurobund', 'US10Y', 'US5Y', 'GDAXI.csv', 'SSEC.csv', \
          'eur-usd', 'gbp-usd', 'usd-jpy', 'usd-cny']
    return (uv, nv)

def bestFeatureSet():
    uv = ['HSI.csv', 'N225.csv']
    nv = ['goldFutures', 'silverFutures', 'copperFutures', 'oilFutures', \
          'eur-usd', 'dollar', 'usd-cny', 'gbp-usd']
    return (uv, nv)

# Read in gold and stock data
dates, prices = gold.getData('data/dates', 'data/prices')
stocks = []
useVolumeNames, noVolumeNames = [], []
# useVolumeNames, noVolumeNames = fullFeatureSet()
# useVolumeNames, noVolumeNames = bestFeatureSet()
for name in useVolumeNames:
    s = stock.Stock('data/' + name, True)
    s.getPrices()
    stocks.append(s)
    print "Added stock:", s.name
for name in noVolumeNames:
    s = stock.Stock('data/' + name, False)
    s.getPrices()
    stocks.append(s)
    print "Added stock:", s.name

# Process data - make X and Y for all examples
X, Y = [], []
currP, nextP = [], []
for i in range(60, len(dates)-1):
    x, y, _ = gold.makeFeatures(prices, i)
    stockFeatures = [s.makeFeatures(dates[i]) for s in stocks]
    noEmpty = True
    for sf in stockFeatures:
        x = x + sf
        if len(sf) == 0: noEmpty = False
    if noEmpty:
        X.append(x)
        Y.append(y)
        currP.append(prices[i])
        nextP.append(prices[i+1])
X = np.array(X)
Y = np.array(Y)
currP = np.array(currP)
nextP = np.array(nextP)

def printErrors(pred, actual):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(pred)):
        if pred[i] == 1:
            if actual[i] == 1: tp = tp + 1
            else: fp = fp + 1
        else:
            if actual[i] == 0: tn = tn + 1
            else: fn = fn + 1
    print "\nTrue positive =", tp
    print "True negative =", tn
    print "False positive =", fp
    print "False negative =", fn
    print "Correct =", tp + tn
    print "Errors =", fp + fn
    precision = float(tp)/(tp + fp) * 100
    recall = float(tp)/(tp + fn) * 100
    accuracy = float(tp + tn)/len(pred) * 100
    print "Precision =", precision
    print "Recall =", recall
    print "Accuracy =", accuracy
    return (precision, recall, accuracy)

# Logistic regression
def logisticRegression():
    kf = cross_validation.KFold(len(Y), n_folds=10, indices=False)
    precision, recall, accuracy = [], [], []
    for train, test in kf:
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        currP_test, nextP_test = currP[test], nextP[test]
        clf = linear_model.LogisticRegression(penalty='l1')
        clf.fit(X_train, Y_train)
	Y_pred = clf.predict(X_test)
        errs = printErrors(Y_pred, Y_test)	
        precision.append(errs[0])
	recall.append(errs[1])
	accuracy.append(errs[2])
    print "\nPrecision =", sum(precision)/len(precision)
    print "Recall =", sum(recall)/len(recall)
    print "Accuracy =", sum(accuracy)/len(accuracy)    

# Linear regression
def linearRegression():
    kf = KFold(len(Y), n_folds=10, indices=False)
    myErrs, zeroErrs = [], []
    errs = []
    correctDir = 0
    total = 0
    for train, test in kf:
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        currP_test, nextP_test = currP[test], nextP[test]
        clf = linear_model.LinearRegression()
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        for i in range(len(Y_pred)):
            nextPrice = currPrice * (1 + 0.01 * Y_test[i])
            predPrice = currPrice * (1 + 0.01 * Y_pred[i])
            err = (predPrice - nextPrice) / nextPrice * 100
            errs.append(abs(err))
            total = total + 1
            if Y_pred[i] * Y_test[i] >= 0: correctDir = correctDir + 1
    print "Correct Direction =", correctDir
    print "Total =", total
    print "Accuracy =", float(correctDir)/total
    print "Average Error =", sum(errs)/len(errs)

logisticRegression()
