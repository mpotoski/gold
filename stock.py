from datetime import datetime
from sklearn import linear_model

class Stock:

    def __init__(self, filename, useVolume):
        self.filename = filename
        self.useVolume = useVolume
        self.name = filename.split('.')[0]
        self.dates = []
        self.open_prices = []
        self.high_prices = []
        self.low_prices = []
        self.close_prices = []
        self.volumes = []

    def getPrices(self):
        
        # Comma-separated list
        if self.filename[-3:] == 'csv':
            f = open(self.filename, 'r')
            lines = f.read().split()
            f.close()
            for line in lines:
                items = line.split(',')
                date = datetime.strptime(items[0], '%m/%d/%y')
                self.dates.append(date)
                self.open_prices.append(float(items[1]))
                self.high_prices.append(float(items[2]))
                self.low_prices.append(float(items[3]))
                self.close_prices.append(float(items[4]))
                if self.useVolume: self.volumes.append(int(items[5]))

        # Space-separated list
        elif self.filename[-3:] == 'txt':
            f = open(self.filename, 'r')
            lines = f.read().splitlines()
            f.close()
            for line in lines:
                items = line.split()
                datestr = items[0] + ' ' + items[1] + ' ' + items[2]
                date = datetime.strptime(datestr, '%b %d, %Y')
                self.dates.append(date)
                self.open_prices.append(float(items[3].replace(',', '')))
                self.high_prices.append(float(items[4].replace(',', '')))
                self.low_prices.append(float(items[5].replace(',', '')))
                self.close_prices.append(float(items[6].replace(',', '')))
                if self.useVolume: self.volumes.append(int(items[7].replace(',', '')))

        else:
            f = open(self.filename, 'r')
            lines = f.read().splitlines()
            f.close()
            for line in lines:
                items = line.split()
                datestr = items[0] + ' ' + items[1] + ' ' + items[2]
                date = datetime.strptime(datestr, '%b %d, %Y')
                self.dates.append(date)
                self.close_prices.append(float(items[3]))
                self.open_prices.append(float(items[4]))
                self.high_prices.append(float(items[5]))
                self.low_prices.append(float(items[6]))

    # Calculate trendline - return a, b from linear equation y = ax + b
    def calcTrendline(self, i, n):
        points = [[self.close_prices[j]] for j in range(i-n+1, i+1)]
        indices = [[index] for index in range(i-n+1, i+1)]

        # Do linear regression
        clf = linear_model.LinearRegression()
        clf.fit(indices, points)
        return clf.coef_[0]

    # Rate of change for day i with period n
    def calcROC(self, i, n):
        return (self.close_prices[i] - self.close_prices[i-n]) / self.close_prices[i-n]

    # Calculate %K - stochastic oscillator
    def calcK(self, i, n):
        high = max(self.high_prices[i-n:i+1])
        low = min(self.low_prices[i-n:i+1])
        return (self.close_prices[i] - low) / (high - low) * 100

    # Features for a given date
    def makeFeatures(self, date):
        # print "making features for", date
        if date not in self.dates: return []
        i = self.dates.index(date)
        x = []
        nums = [2, 3, 5, 10]

        # Add trendlines
        for n in nums:
            x = x + [self.calcTrendline(i, n)]

        # Calculate ROC in close prices (over multiple intervals); open to close; volume
        curr_roc = self.calcROC(i, 1)
        x = x + [curr_roc]
        for n in nums:
            roc = self.calcROC(i, n)
            x = x + [roc, curr_roc/roc if roc != 0 else curr_roc]
        x = x + [(self.close_prices[i] - self.open_prices[i])/self.open_prices[i]]
        if self.useVolume: x = x + [(self.volumes[i] - self.volumes[i-1])/self.volumes[i-1]]
        
        return x
