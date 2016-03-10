import re
import sys
import os

def getMeasurementList(file):

    with open(file, 'r') as f:
        content = f.read()
    pattern = 'Overall:(.*)'

    x = re.search(pattern,content)
    overallLine = x.group(0)
    overallLine = overallLine.split('\t')[1:]
    measurementLine = []
    for each in overallLine:
        if(each.strip() != ''):
            measurementLine.append(each.strip())
    measurementLine.insert(0, sys.argv[3])
    return measurementLine

def getMeasureNamesList(file):
    with open(evalFile, 'r') as f:
        content = f.read()
    pattern = '(Recall.*)\n'
    measureNames = re.search(pattern,content).group(0)
    measureNames = measureNames.split('\t')
    measureList = []
    for each in measureNames:
        if(each.strip() != ''):
            measureList.append(each.strip())
    measureList.insert(0, sys.argv[2])
    return measureList

def writeHeader(evalFile):
    measureList = getMeasureNamesList(evalFile)
    with open('../../data/' + sys.argv[2] + '_eval_csv'+'.dat', 'w') as f:
        f.write(getLineFromeList(measureList))

def getLineFromeList(list):
    line = list[0]
    for each in list[1:]:
        line = line + ', ' + each
    line = line + '\n'
    return line

def appendLine(file):
    mList = getMeasurementList(file)
    f = open('../../data/' + sys.argv[2] + '_eval_csv'+'.dat', 'a')
    f.write(getLineFromeList(mList))
    f.close()
  

"""
assume command line arguments are: 
    1) evalutaion file from evaluation code
    2) Parameter Name
    3) Parameter Value
    4) write header flag run flag

script stores csv file in '../../data' folder
script should be located in 'REPOSITORY/evaluation/scripts/Python folder'
"""

evalFile = sys.argv[1]
if sys.argv[-1] == "1":
    writeHeader(evalFile)

appendLine(evalFile)

