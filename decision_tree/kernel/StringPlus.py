'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
# import sys 
def printlist(a):
    for i in a:
        print i

class stringPlus:

    def printlist(self,a):
        for i in a:
            print (i)


def test():
    sp = stringPlus()
    sp.printlist([1,2,3])

def dlog(var, varName):
	print '======%s:======\n %s \n======' % (varName, var)

# test()



# print linuxany100.value

# print names.keys()[1]


# printlist(list)
