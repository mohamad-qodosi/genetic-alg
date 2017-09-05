from random import *
import math

FILE_NAME = 'karate.paj'
NODES_COUNT = 34
COMMIUNITIES_COUNT = 5
INITIAL_T = 0
EDGE_COUNT = 0
INITIAL_POPULATION = 5
INITIAL_GENERATION = 200

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

def fitnessFunction(people, graph, degree):
    Q = 0.
    for i in range(NODES_COUNT):
        for j in range(i):
            if people[i] == people[j]:
                Q += graph[i][j] - (degree[i] * degree[j] * 1.0) / EDGE_COUNT
    Q = Q * 2
    for i in range(NODES_COUNT):
        Q += 1 - (degree[i] ** 2 * 1.0) / EDGE_COUNT
    Q = Q / EDGE_COUNT
    return Q

def findPerson(population, p):
    for i in range(len(population)):
        if p < population[i][1]:
            return i
        else:
            p -= population[i][1]

def selectParents(population):
    sumQ = 0.
    for people in population:
        sumQ += people[1]
    p1 = random() * sumQ
    dadIndex = findPerson(population, p1)
    p2 = random() * sumQ
    momIndex = findPerson(population, p2)
    return momIndex, dadIndex, (sumQ * 1.0 / len(population))

def generateChild(mom, dad):
    child = dad[:]
    crossOver = randrange(NODES_COUNT)
    for i in range(NODES_COUNT):
        if mom[i] == mom[crossOver]:
            child[i] = mom[crossOver]
    return child

def immutate(genome):
    place = randrange(NODES_COUNT)
    immutatedValue = randrange(COMMIUNITIES_COUNT)
    genome[place] = immutatedValue

def geneticAlg(mom , dad):
    firstChild = generateChild(mom, dad)
    secondChild = generateChild(dad, mom)
    immutate(firstChild)
    immutate(secondChild)
    return firstChild, secondChild


def semulatedAnnealing(old1, old2, new1, new2, T):
    if old1[1] < old2[1]:
        old1, old2 = old2, old1
    if new1[1] < new2[1]:
        new1, new2 = new2, new1
    # old1 is better than old2 and new1 is better than new2
    if new2[1] >= old1[1]:
        # new1 and new2 are better than old1 and old2
        # N1>N2>O1>O2
        return new1, new2
    elif new1[1] >= old2[1]:
        # N1>O1>N2>O2 or N1>O1>O2>N2 or O1>N1>O2>N2 or O1>N1>N2>O2
        deltaE = new2[1] - old1[1]
        r = random()
        if T == 0:
            prob = 0
        else:
            prob = math.exp(deltaE / T)
        if r < prob:
            return new1, new2
        else:
            return new1, old1
    elif new1[1] < old2[1]:
        # O1>O2>N1>N2
        deltaE = new2[1] + new1[1] - old1[1] - old2[1]
        r = random()
        if T == 0:
            prob = 0
        else:
            prob = math.exp(deltaE / T)
        if r < prob:
            return new1, new2
        else:
            deltaE = new1[1]- old2[1]
            r = random()
            if T == 0:
                prob = 0
            else:
                prob = math.exp(deltaE / T)
            if r < prob:
                return old1, new1
            else:
                return old1, old2

def generateNewPop(population, T):
    report = [0., 0.]
    population = sorted(population, key=lambda x: x[1])
    report[0] = population[-1][1]
    newPop = []
    while len(newPop) < INITIAL_POPULATION:
        momIndex , dadIndex, report[1] = selectParents(population)
        mom = population[momIndex]
        dad = population[dadIndex]
        firstChild, secondChild = geneticAlg(mom[0], dad[0])
        firstChild = firstChild, fitnessFunction(firstChild, graph, degree)
        secondChild = secondChild, fitnessFunction(secondChild, graph, degree)

        firstChild, secondChild = semulatedAnnealing(mom, dad, firstChild, secondChild, T)
        newPop.append(firstChild)
        newPop.append(secondChild)
    if (len(newPop) > INITIAL_POPULATION):
        newPop = newPop[:-1]
    return newPop, report

def findClusters(population, graph, degree):
    T = INITIAL_T * 1.0
    report = [[0., 0.] for i in range(INITIAL_GENERATION)]
    for i in range(INITIAL_GENERATION):
        population, report[i] = generateNewPop(population, T)
        T /= 2
    sorted(population, key=lambda x: x[1])
    return population[-1], report

graph, degree = loadData(FILE_NAME)
for i in range(NODES_COUNT):
    EDGE_COUNT += degree[i]
report = [[0., 0.] for i in range(INITIAL_GENERATION)]
for k in range(200):
    population = [[randrange(1, COMMIUNITIES_COUNT + 1) for i in range(NODES_COUNT)] for j in range(INITIAL_POPULATION)]
    for i in range(len(population)):
        population[i] = population[i], fitnessFunction(population[i], graph, degree)
    clusters, rep = findClusters(population, graph, degree)
    for i in range(INITIAL_GENERATION):
        report[i][0] += rep[i][0]
        report[i][1] += rep[i][1]
for t in report:
    print t[0] / 200, '\t', t[1] / 200
'''
population = [[randrange(1, COMMIUNITIES_COUNT + 1) for i in range(NODES_COUNT)] for j in range(INITIAL_POPULATION)]
for i in range(len(population)):
    population[i] = population[i], fitnessFunction(population[i], graph, degree)
'''
#clusters = findClusters(population, graph, degree)
#print clusters[0]