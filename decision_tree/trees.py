'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from kernel.StringPlus import *

from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['water', 'flippers']
    # change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    shannonEnt = 0.0

    numEntries = len(dataSet)
    labelCountDict = {}

    # the the number of unique elements and their occurance
    for item in dataSet:
        # item : [1, 1, 'yes']
        currentLabel = item[-1]     # 'Yes'

        if currentLabel not in labelCountDict.keys():
            labelCountDict[currentLabel] = 0
        labelCountDict[currentLabel] += 1

    # print labelCountDict
    for key in labelCountDict:
        prob = float(labelCountDict[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)       # log base 2
        print 'Key:[%s] Count:[%s] prob:[%f] shannonEnt:[%s]' \
            % (key, labelCountDict[key], prob, -prob * log(prob, 2))
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # chop out axis used for splitting
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def filterDataSet(dataSetIn, column, value):
    filteredSet = []
    for item in dataSetIn:
        if item[column] == value:
            # chop out axis used for splitting
            itemToFilter = item[:]
            itemToFilter.pop(column)
            # print 'Filtering on column [%s] by value[%s]' % (column, value)
            # retDataSet.append(reducedFeatVec)
            filteredSet.append(itemToFilter)
    return filteredSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]
        # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            # subDataSet = splitDataSet(dataSet, i, value)
            subDataSet = filterDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            # print newEntropy
        # calculate the info gain; ie reduction in entropy
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


# sp = StringPlus()
def test():
    ds, labels = createDataSet()
    printlist(ds)
    print calcShannonEnt(ds)

    print "split with first column = 0"
    printlist(splitDataSet(ds, 0, 0))
    print '-'*20
    printlist(splitDataSet(ds, 0, 1))
    print "split with second column = 1"
    printlist(splitDataSet(ds, 1, 0))
    print '-'*20
    printlist(splitDataSet(ds, 1, 1))

    print chooseBestFeatureToSplit(ds)
# test()


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(
        classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    dlog(classList, 'classList')

    # stop splitting when all of the classes are equal
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # stop splitting when there are no more features in dataSet
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # copy all of labels, so trees don't mess up existing labels
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels
        )
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
