from random import *
import numpy as np
import time

FILE_NAME = 'karate.paj'
NODES_COUNT = 34
ALPHA = 0.1
LAYERS = [34, 16, 8, 4, 8, 16, 34]


def loadData(fileName):
    graph = [[0 for i in range(NODES_COUNT)] for j in range(NODES_COUNT)]
    f = open(fileName)
    degree = [0 for i in range(NODES_COUNT)]
    for line in f:
        line = line[0:-1]
        line = line.split(' ')
        i = int(line[0]) - 1
        j = int(line[1]) - 1
        graph[i][j] = graph[j][i] = 1
        degree[i] += 1
        degree[j] += 1
    f.close()
    return graph, degree


def sigmoid(z):
    return 1.0 / (1 + np.e ** -z)


def feedForward(X, w, returnAll = False):
    out = [0] * (len(w) + 1)
    x = X.copy()
    for i in range(len(w)):
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        out[i] = x.copy()
        x = sigmoid(x.dot(w[i]))
    out[len(w)] = x
    if returnAll:
        return out
    else:
        return x


def crossEntropyCostFunction(x, w, y):
    j = 0.
    H = feedForward(x, w)
    j = -sum(np.einsum('ij,ji->i', y, np.log(H.transpose())))
    j -= sum(np.einsum('ij,ji->i', (1 - y), np.log(1 - H.transpose())))
    return j / x.shape[0]


def backpropagation(x, w, y):
    sigma = [np.array([])] * (len(w) + 1)
    deltaW = [0] * len(w)
    a = feedForward(x, w, True)
    sigma[len(w)] = a[-1] - y
    for i in range(len(w) - 1, 0, -1):
        deltaW[i] = (a[i].transpose().dot(sigma[i + 1])) / x.shape[0]
        sigma[i] = np.delete(sigma[i + 1].dot(w[i].transpose()) * a[i] * (1 - a[i]), 0, 1)
    deltaW[0] = (a[0].transpose().dot(sigma[1])) / x.shape[0]
    return deltaW


def gradientChecking(x, w, y, e):
    deltaW = [np.array([[0. for k in range(len(w[i][j]))] for j in range(len(w[i]))]) for i in range(len(w))]
    for i in range(len(w)):
        for j in range(len(w[i])):
            for k in range(len(w[i][j])):
                t = w[i][j][k]
                w[i][j][k] = t + e
                j1 = crossEntropyCostFunction(x, w, y)
                w[i][j][k] = t - e
                j2 = crossEntropyCostFunction(x, w, y)
                w[i][j][k] = t
                deltaW[i][j][k] = (j1 - j2) / (2 * e)
    return deltaW


def getSize(w):
    s = 0.
    for i in w:
        for j in i:
            for k in j:
                s += k ** 2
    return s ** 0.5

graph, degree = loadData(FILE_NAME)
x = np.array(graph)
y = x
w = [np.array([[(random() * 2) - 0.5 for i in range(LAYERS[k + 1])] for j in range(LAYERS[k] + 1)]) for k in range(len(LAYERS) - 1)]
for i in range(len(w) / 2):
    #wPartial = [w[i], w[len(w) - i - 1]]
    #xPartial = feedForward(x, w[:i])
    newCost = crossEntropyCostFunction(feedForward(x, w[:i]), [w[i], w[len(w) - i - 1]], feedForward(x, w[:i]))
    prevCost = 1 + newCost
    while prevCost - newCost > 0.0001:
        deltaW = backpropagation(feedForward(x, w[:i]), [w[i], w[len(w) - i - 1]], feedForward(x, w[:i]))
        w[i] -= ALPHA * deltaW[0]
        w[len(w) - i - 1] -= ALPHA * deltaW[1]
        prevCost = newCost
        newCost = crossEntropyCostFunction(feedForward(x, w[:i]), [w[i], w[len(w) - i - 1]], feedForward(x, w[:i]))
    if prevCost < newCost:
        print "Large Alpha"

newCost = crossEntropyCostFunction(x, w, x)
prevCost = 1 + newCost
while prevCost - newCost > 0.0000001:
    deltaW = backpropagation(x, w, x)
    for i in range(len(w)):
        w[i] -= ALPHA * deltaW[i]
    prevCost = newCost
    newCost = crossEntropyCostFunction(x, w, x)
if prevCost < newCost:
    print "Large Alpha"

print crossEntropyCostFunction(x, w, x)
userGroupFuzy = feedForward(x, w[:len(w) / 2])
for i in range(len(userGroupFuzy)):
    maxIndex = 0
    for j in range(len(userGroupFuzy[i])):
        if userGroupFuzy[i][maxIndex] > userGroupFuzy[i][j]:
            maxIndex = j
    print i + 1, ' - ', maxIndex
