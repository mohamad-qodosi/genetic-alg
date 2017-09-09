from random import *
import math

FILE_NAME = 'karate.paj'
NODES_COUNT = 34
COMMIUNITIES_COUNT = 2
INITIAL_T = 10000
EDGE_COUNT = 0
INITIAL_POPULATION = 10
INITIAL_GENERATION = 200
RIGHT_ANSWER = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


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


def NMI(clustering1, clustering2):
    c1Max = max(clustering1)
    c2Max = max(clustering2)
    c1Min = min(clustering1)
    c2Min = min(clustering2)
    clustering1Count = [0] * (c1Max - c1Min + 1)
    clustering2Count = [0] * (c2Max - c2Min + 1)
    for i in range(NODES_COUNT):
        clustering1Count[clustering1[i] - c1Min] += 1
        clustering2Count[clustering2[i] - c2Min] += 1

    z = 0.
    for i in range(c1Min, c1Max + 1):
        for j in range(c2Min, c2Max + 1):
            sameClusters = 0
            for k in range(NODES_COUNT):
                if clustering1[k] == i and clustering2[k] == j:
                    sameClusters += 1
            if sameClusters != 0:
                z += (sameClusters * 1.0 / NODES_COUNT) * math.log((NODES_COUNT * sameClusters * 1.0) / (clustering1Count[i - c1Min] * clustering2Count[j - c2Min]))
    return z


def findPerson(population, p):
    for i in range(len(population)):
        if p < population[i][1]:
            return i
        else:
            p -= population[i][1]


def selectParents(population):
    sumQ = 0.
    sumNMI = 0.
    for people in population:
        sumQ += people[1]
        sumNMI += NMI(people[0], RIGHT_ANSWER)
    p1 = random() * sumQ
    dadIndex = findPerson(population, p1)
    p2 = random() * sumQ
    momIndex = findPerson(population, p2)
    return momIndex, dadIndex, (sumNMI * 1.0 / len(population))


def generateChilds(mom, dad, crossOverPattern):
    child1 = dad[:]
    child2 = mom[:]
    for i in range(NODES_COUNT):
        if crossOverPattern[i] == 0:
            child1[i] = mom[i]
            child2[i] = dad[i]
    return child1, child2


def immutate(genome):
    place = randrange(NODES_COUNT)
    immutatedValue = randrange(1, COMMIUNITIES_COUNT + 1)
    genome[place] = immutatedValue


def geneticAlg(mom , dad):
    crossOverPattern = [randrange(2) for i in range(NODES_COUNT)]
    firstChild, secondChild = generateChilds(mom, dad, crossOverPattern)
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
    report[0] = NMI(population[-1][0], RIGHT_ANSWER)
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
    return population[-1],report

graph, degree = loadData(FILE_NAME)
for i in range(NODES_COUNT):
    EDGE_COUNT += degree[i]
report = [[0., 0.] for i in range(INITIAL_GENERATION)]
for k in range(200):
    population = [[randrange(1, COMMIUNITIES_COUNT + 1) for i in range(NODES_COUNT)] for j in range(INITIAL_POPULATION)]
    for i in range(len(population)):
        population[i] = population[i], fitnessFunction(population[i], graph, degree)
    clusters,rep = findClusters(population, graph, degree)
    for i in range(INITIAL_GENERATION):
        report[i][0] += rep[i][0]
        report[i][1] += rep[i][1]
for t in report:
    print t[0] / 200, '\t', t[1] / 200

'''
population = [[randrange(1, COMMIUNITIES_COUNT + 1) for i in range(NODES_COUNT)] for j in range(INITIAL_POPULATION)]
for i in range(len(population)):
    population[i] = population[i], fitnessFunction(population[i], graph, degree)
clusters = findClusters(population, graph, degree)
print clusters
print (RIGHT_ANSWER, fitnessFunction(RIGHT_ANSWER, graph, degree))'''