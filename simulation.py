import gold, stock, util, collections, random, math, numpy as np
from sklearn import linear_model

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
    x, _, y = gold.makeFeatures(prices, i)
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
model = None

class GoldMDP(util.MDP):

    def __init__(self, days):
        self.days = days
        self.stateList = [self.makeState(index) for index in range(len(days))]

    def makeState(self, index):
        day = self.days[index]
        x = currP[day]
        y = nextP[day]
        return (index, day, x, y)

    def startState(self):
        return self.stateList[0]

    def isEnd(self, state):
        return (state[0] == len(self.days)-1)

    # Return set of actions possible from |state|.
    #   positive number = buy a ounces of gold
    #   negative number = sell a ounces of gold
    def actions(self, state):
        actions = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        # actions = [-1, 1]
        return actions

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action):
        if self.isEnd(state): return []
        newIndex = state[0] + 1
        newState = self.stateList[newIndex]
        reward = action * (state[3] - state[2])
        return [(newState, 1.0, reward)]

    def discount(self): return 1.0


# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # Your algorithm will be asked to produce an action given a state.
    # You should use an epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        actions = self.actions(state)
        if random.random() < self.explorationProb:
            return actions[random.randint(0, len(actions)-1)]
        return max(actions, key=lambda a : self.getQ(state, a))

    # Used for outputting the policy learned without exploration.
    def getActionWithoutExploring(self, state):
        actions = self.actions(state)
        return max(actions, key=lambda a : self.getQ(state, a))

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        if newState == None:
            r = reward - self.getQ(state, action)
        else:
            actions = self.actions(newState)
            newAction = max(actions, key=lambda a : self.getQ(newState, a))
            r = reward + self.discount * self.getQ(newState, newAction) - self.getQ(state, action)
        newWeights = self.weights
        for (feature, val) in self.featureExtractor(state, action):
            newWeights[feature] = self.weights[feature] + self.getStepSize() * r * val
        self.weights = newWeights

def predictFeatureExtractor(state, action):
    global model
    index, day, x, y = state
    name = '%d' % action
    features = X[day]
    pred = model.predict([features])
    intercept = 'i%d' % action
    return [(name, pred[0])] #, (intercept, 1.0)]

def printStates(mdp):
    states = sorted(mdp.states, key=lambda s : s[0])
    for state in states:
        print (state[1], state[2], state[3])

def runQLearning():
    global model

    days = list(range(len(X)))
    print len(days)

    randomRewards = 0.0
    testRewards = 0.0

    # Test separately on 100-day periods
    period = 100
    numSets = len(days)/period
    testSets = [n * period for n in range(numSets)]
    for n in testSets:

        print 'Testing on days %d - %d:' % (n, n+period)

        # Test on [n, n + period] examples
        testDays = days[n:n+period]

        # Train on all remaining examples
        trainDays = [d for d in days if d not in testDays]

        # Make train & test MDPs
        trainMDP = GoldMDP(trainDays)
        trainMDP.computeStates()
        testMDP = GoldMDP(testDays)
        testMDP.computeStates()

        # Train linear prediction model on train set
        model = linear_model.LinearRegression()
        model.fit(X[trainDays], Y[trainDays])

        # Measure classification accuracy on test set
        Y_pred = model.predict(X[testDays])
        Y_actual = Y[testDays]
        correct = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] *  Y_actual[i] >= 0: correct = correct + 1
        print "Accuracy = %.2f" % (float(correct)/len(Y_pred))

        # Learn (reinforcement Q-learning) on trainMDP, choosing all random actions
        rl = QLearningAlgorithm(trainMDP.actions, trainMDP.discount(), predictFeatureExtractor, 1.0)
        rewards = util.simulate(trainMDP, rl, 1)
        print rl.weights
        print rewards
        randomRewards = randomRewards + rewards[0]

        # Run with trained RL algorithm on testMDP, choosing all max actions
        rl.explorationProb = 0.0
        rewards = util.simulate(testMDP, rl, 1)
        print rl.weights
        print rewards
        testRewards = testRewards + rewards[0]
        print "Average rewards per day =", rewards[0]/period

    # Output total profit & average daily profit
    print 'Random rewards = %.2f' % randomRewards
    print 'Test rewards = %.2f' % testRewards

    numDays = (len(days)/period) * period
    print 'Avg random = %.2f' % (randomRewards/numDays)
    print 'Avg test = %.2f' % (testRewards/numDays)

runQLearning()
