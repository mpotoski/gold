import gold, stock, numpy as np
from sklearn import linear_model, svm, feature_selection, cross_validation
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, RFECV

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
# useVolumeNames, noVolumeNames = [], []
# useVolumeNames, noVolumeNames = fullFeatureSet()
useVolumeNames, noVolumeNames = bestFeatureSet()
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
bothP = 0
bothN = 0
firstP = 0
firstN = 0
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
        diff = nextP[-1] - currP[-1]
        if diff > 0 and y == 1: bothP = bothP + 1
        elif diff > 0 and y == 0: firstN = firstN + 1
        elif y == 0: bothN = bothN + 1
        else: firstP = firstP + 1
print "bothP = ", bothP
print "bothN = ", bothN
print "firstP = ", firstP
print "firstN = ", firstN

def normalize():
    m = len(X)
    n = len(X[0])
    totals = [0.0 for j in range(n)]
    mins = [float('+inf') for j in range(n)]
    maxs = [float('-inf') for j in range(n)]
    for i in range(m):
        for j in range(n):
            mins[j] = min(mins[j], X[i][j])
            maxs[j] = max(maxs[j], X[i][j])
            totals[j] = totals[j] + X[i][j]
    mean = [t/m for t in totals]
    for i in range(m):
        for j in range(n):
            X[i][j] = (X[i][j] - mean[j])/(maxs[j] - mins[j])

#normalize()
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
    kf = KFold(len(Y), n_folds=10, indices=False)
    precision, recall, accuracy = [], [], []
    profit = []
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
        for i in range(len(Y_pred)):
            if Y_pred[i] == 1: # Buy 1 oz
                profit.append(nextP_test[i] - currP_test[i])
            else: # Sell 1 oz
                profit.append(currP_test[i] - nextP_test[i])
    print "\nPrecision =", sum(precision)/len(precision)
    print "Recall =", sum(recall)/len(recall)
    print "Accuracy =", sum(accuracy)/len(accuracy)    
    print "Average Daily Profit =", sum(profit)/len(profit)
    pos = 0
    for i in profit:
        if i >= 0: pos = pos + 1 
    print "Pos. profit percentage =", float(pos)/len(profit)
    return sum(accuracy)/len(accuracy)

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
            err = (Y_pred - Y_test)/Y_test * 100
            errs.append(abs(err))
            currPrice = currP_test[i]
            nextPrice = currPrice * (1 + 0.01 * Y_test[i])
            predPrice = currPrice * (1 + 0.01 * Y_pred[i])
            myErr = (predPrice - nextPrice) / nextPrice * 100
            zeroErr = (currPrice - nextPrice) / nextPrice * 100
            print currPrice, nextPrice, predPrice
            myErrs.append((predPrice - nextPrice) * (predPrice - nextPrice))
            zeroErrs.append((currPrice - nextPrice) * (currPrice - nextPrice))
            total = total + 1
            if Y_pred[i] * Y_test[i] >= 0: correctDir = correctDir + 1
    print "myErr average =", sum(myErrs)/len(myErrs)
    print "zeroErr average =", sum(zeroErrs)/len(zeroErrs)
    print "cd =", correctDir
    print "total =", total
    print "acc =", float(correctDir)/total
    print "err average =", sum(errs)/len(errs)

# SVM
def runSVM():
    kf = KFold(len(Y), n_folds=10, indices=False)
    precision, recall, accuracy = [], [], []
    for train, test in kf:
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        print "RADIAL KERNEL SVM:"
        svc = svm.SVC()
        svc.fit(X_train, Y_train)
        Y_pred = svc.predict(X_test)
        errs = printErrors(Y_pred, Y_test)
        precision.append(errs[0])
        recall.append(errs[1])
        accuracy.append(errs[2])
    print "\nPRECISION =", sum(precision)/len(precision)
    print "RECALL =", sum(recall)/len(recall)
    print "ACCURACY =", sum(accuracy)/len(accuracy)

# Linear SVM
def runLinearSVM(lossFunction):
    kf = KFold(len(Y), n_folds=10, indices=False)
    precision, recall, accuracy = [], [], []
    for train, test in kf:
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        print "LINEAR KERNEL SVM: %s" % lossFunction
        svc = svm.LinearSVC(loss=lossFunction)
        svc.fit(X_train, Y_train)
        Y_pred = svc.predict(X_test)
        errs = printErrors(Y_pred, Y_test)
        precision.append(errs[0])
        recall.append(errs[1])
        accuracy.append(errs[2])
    print "\nPRECISION =", sum(precision)/len(precision)
    print "RECALL =", sum(recall)/len(recall)
    print "ACCURACY =", sum(accuracy)/len(accuracy)

# Polynomial SVM
def runPolySVM():
    kf = KFold(len(Y), n_folds=10, indices=False)
    precision, recall, accuracy = [], [], []
    for train, test in kf:
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        print "POLY KERNEL SVM:"
        svc = svm.SVC(kernel='poly')
        svc.fit(X_train, Y_train)
        Y_pred = svc.predict(X_test)
        errs = printErrors(Y_pred, Y_test)
        precision.append(errs[0])
        recall.append(errs[1])
        accuracy.append(errs[2])
    print "\nPRECISION =", sum(precision)/len(precision)
    print "RECALL =", sum(recall)/len(recall)
    print "ACCURACY =", sum(accuracy)/len(accuracy)

#clf = linear_model.LogisticRegression()
#clf.fit(X, Y)
#Y_pred = clf.predict(X)
#printErrors(Y_pred, Y)
logisticRegression()
# linearRegression()
# runSVM()
# runLinearSVM('l1')
# runLinearSVM('l2')
# runPolySVM()
