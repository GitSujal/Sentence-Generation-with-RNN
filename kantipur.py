import os
import collections
#import numpy as np 


with open("kantipur_samachar_valid.txt", 'r') as f:
	document = f.read()
stopWord = '†'
startWord = 'Å'

expungeString = startWord + stopWord + '\n'
cleanDocument = ''.join( c for c in document if  c not in expungeString)
#cleanDocument = startWord + cleanDocument.replace('\n', stopWord + startWord)
cleanDocument = startWord + cleanDocument.replace('।', stopWord + startWord)
with open("kantipur_samachar_valid_clean.txt", 'w') as f:
	f.write(cleanDocument)