import sys
import re

filedata = None

str=sys.argv[2]
str = str.split(',')
pattern =   '\(' + str[0] + ',' + str[1] + ', .*\))'

f = open(sys.argv[1],'r')
filedata = f.read()
f.close()

m = re.search(pattern, filedata)
newdata = filedata.replace(m.group(0),sys.argv[2])

f = open(sys.argv[1],'w')
f.write(newdata)
f.close()


