fileReadObject = open('Result2.txt','r')
fileWriteObject = open('Result2Overload.txt','w')
try:
	lineNum = 0
	for line in fileReadObject:
		words = line.split(' ')
		wordsNew = words[1:len(words)-2]
		for i in range(0,len(wordsNew)):
			if i == 0:
				fileWriteObject.write( wordsNew[i] )
			else :
				fileWriteObject.write( ' ' )
				fileWriteObject.write( wordsNew[i] )
		fileWriteObject.write( '\n' )
finally:
	fileReadObject.close()
	fileWriteObject.close()