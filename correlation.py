# Used to compute the correlation between each intermarket variable and
# gold prices.  Outputs the variables sorted in descending order of 
# absolute-value correlation coefficient with gold price fix.

import gold, stock, math

def correlation(x_dates, x_prices, y_dates, y_prices):

    # Make clean, matching lists for x, y
    x = []
    y = []
    for i in range(len(x_dates)-1):
        date = x_dates[i]
        next_date = x_dates[i+1]
        if date in y_dates and next_date in y_dates:
            j1 = y_dates.index(date)
            j2 = y_dates.index(next_date)
            x.append((x_prices[i+1] - x_prices[i])/x_prices[i])
            y.append((y_prices[j2] - y_prices[j1])/y_prices[j1])

    # Generate lists xy, x2, y2
    n = len(x)
    xy = [x[i] * y[i] for i in range(n)]
    x2 = [x[i] * x[i] for i in range(n)]
    y2 = [y[i] * y[i] for i in range(n)]

    # Compute formula
    num = n * sum(xy) - sum(x) * sum(y)
    den = (n * sum(x2) - sum(x) * sum(x)) * (n * sum(y2) - sum(y) * sum(y))
    return num / math.sqrt(den)

# Calculate correlation coefficients
dates, prices = gold.getData('data/dates', 'data/prices')
stocks = []
useVolumeNames = ['GSPC.csv', 'IXIC.csv', 'DJI.txt', 'N100.csv', 'N225.csv', \
                  'HSI.csv', 'FCHI.csv', 'FTSE.csv', 'GSPTSE.csv', \
                  'KS11.csv', 'SSMI.csv']
noVolumeNames = ['goldFutures', 'silverFutures', 'eurobund', \
                 'dollar', 'US10Y', 'GDAXI.csv', 'SSEC.csv', \
                 'eur-usd', 'gbp-usd', 'usd-jpy', 'usd-cny', \
                 'copperFutures', 'oilFutures', 'US5Y']
for name in useVolumeNames:
    s = stock.Stock('data/' + name, True)
    s.getPrices()
    stocks.append(s)
for name in noVolumeNames:
    s = stock.Stock('data/' + name, False)
    s.getPrices()
    stocks.append(s)

corrs = []
for s in stocks:
    c = correlation(dates, prices, s.dates, s.close_prices)
    corrs.append((s.name, c))
corrs.sort(key=lambda a : abs(a[1]), reverse=True)
for c in corrs:
    print '%s: %.3f' % (c[0], c[1])
