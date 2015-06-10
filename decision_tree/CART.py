'''
Created on Sun May 24 15:16:30 CST 2015
Refactoring Decision Tree Source Code for Machine Learning in Action - ch9 CART
@author: Michael Xie
'''
from kernel.StringPlus import *

from numpy import *


# general function to parse tab -delimited floats
def loadDataSet(fileName):
    # assume last column is target value
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')

        # map all elements to float()
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    mat1 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    return mat0, mat1


def regLeaf(dataSet):  # returns the value used for each leaf
    return mean(dataSet[:, -1])


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestCARTSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    # if all the target variables are the same value: quit and return value
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # exit cond 1
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    print '%d x %d' % (m, n)
    # the choice of the best feature is driven by Reduction in RSS error from
    # mean
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:
        return None, leafType(dataSet)  # exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # exit cond 3
        return None, leafType(dataSet)
    return bestIndex, bestValue
    # returns the best feature to split on
    # and the value used for that split


# assume dataSet is NumPy Mat so we can array filtering
def createCARTTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # choose the best split
    feature, value = chooseBestCARTSplit(dataSet, leafType, errType, ops)
    if feature is None:
        return value  # if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feature
    retTree['spVal'] = value
    lSet, rSet = binSplitDataSet(dataSet, feature, value)
    retTree['left'] = createCARTTree(lSet, leafType, errType, ops)
    retTree['right'] = createCARTTree(rSet, leafType, errType, ops)
    return retTree


def linearSolve(dataSet):  # helper function used in two places
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]  # and strip out Y
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):  # create linear model and return coeficients
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)  # if we have no test data collapse the tree
    # if the branches are not trees try to prune them
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) +\
            sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


# def mmTree(tree):
# tree = "{'spInd': 1, 'spVal': matrix([[ 0.39435]]), 'right': {'spInd':
# 1, 'spVal': matrix([[ 0.582002]]), 'right': {'spInd': 1, 'spVal':
# matrix([[ 0.797583]]), 'right': 3.9871631999999999, 'left':
# 2.9836209534883724}, 'left': 1.980035071428571}, 'left': {'spInd': 1,
# 'spVal': matrix([[ 0.197834]]), 'right': 1.0289583666666666, 'left':
# -0.023838155555555553}}"

#        if shape(testData)[0] == 0:
# return getMean(tree)  # if we have no test data collapse the tree
# if the branches are not trees try to prune them
#     if (isTree(tree['right']) or isTree(tree['left'])):
#         lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
#     if isTree(tree['left']):
#         tree['left'] = prune(tree['left'], lSet)
#     if isTree(tree['right']):
#         tree['right'] = prune(tree['right'], rSet)
# if they are now both leafs, see if we can merge them
#     if not isTree(tree['left']) and not isTree(tree['right']):
#         lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
#         errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) +\
#             sum(power(rSet[:, -1] - tree['right'], 2))
#         treeMean = (tree['left'] + tree['right']) / 2.0
#         errorMerge = sum(power(testData[:, -1] - treeMean, 2))
#         if errorMerge < errorNoMerge:
#             print "merging"
#             return treeMean
#         else:
#             return tree
#     else:
#         return tree


# def planting(dic, bossTree, lr, treeDicts):
#     treeDict = {}
#     treeDict['value'] = dic['spVal'][0, 0]
#     treeDict['bossID'] = bossTree['selfID']
#     treeDict['selfID'] = treeDict['bossID'] + lr
#     treeDicts.append(treeDict)
#     if isTree(dic['left']):
#         planting(dic['left'], treeDict, 'l', treeDicts)
#     else:
#         nodeDict = {}
#         nodeDict['value'] = dic['left']
#         nodeDict['selfID'] = treeDict['selfID'] + 'l'
#         nodeDict['bossID'] = treeDict['selfID']
#         # nodeDict['terminating'] = '('
#         treeDicts.append(nodeDict)
#     if isTree(dic['right']):
#         planting(dic['right'], treeDict, 'r', treeDicts)
#     else:
#         nodeDict = {}
#         nodeDict['value'] = dic['right']
#         nodeDict['selfID'] = treeDict['selfID'] + 'r'
#         nodeDict['bossID'] = treeDict['selfID']
#         treeDicts.append(nodeDict)


def plantTree(tree):
    treeDicts = []
    tree['bossID'] = 'root'
    tree['selfID'] = 'Tree'
    planting(tree, tree, '', treeDicts)
    return treeDicts


def harvestTree(treeDicts):
    for treeDict in treeDicts:
        treeString = '%s-->%s[%.2f]' % (treeDict['bossID'],
                                        treeDict['selfID'], treeDict['value'])
        print treeString


def newBranch(value, bossTree, lr, treeDicts):
    treeDict = {}
    treeDict['value'] = value
    treeDict['bossID'] = bossTree['selfID']
    treeDict['selfID'] = treeDict['bossID'] + lr
    treeDicts.append(treeDict)


def planting(dic, bossTree, lr, treeDicts):
    treeDict = {}
    treeDict['value'] = dic['spVal'][0, 0]
    treeDict['bossID'] = bossTree['selfID']
    treeDict['selfID'] = treeDict['bossID'] + lr
    treeDicts.append(treeDict)
    if isTree(dic['left']):
        planting(dic['left'], treeDict, 'l', treeDicts)
    else:
        newBranch(dic['left'], treeDict, 'l', treeDicts)
    if isTree(dic['right']):
        planting(dic['right'], treeDict, 'r', treeDicts)
    else:
        newBranch(dic['right'], treeDict, 'r', treeDicts)
