"""
	Dependencies: Python3
	Unresolved Issues:
		1. Check when Corpus fails to complete processing
			ERROR:filename not found breaks the code further down the line
			Safely exit

"""


import os
import tensorflow as tf 
import numpy as np 
import collections



class Corpus():
	def __init__(self, preprocessed=True):
		train_filename = "data/kantipur_samachar_train_clean.txt"
		vaild_filename = "data/kantipur_samachar_valid_clean.txt"
		test_filename = "data/kantipur_samachar_test_clean.txt"

		clean_train_data = self._readCorpus(filename = train_filename, preprocessed = preprocessed)
		clean_vaild_data = self._readCorpus(filename = vaild_filename, preprocessed = preprocessed)
		clean_test_data = self._readCorpus(filename = test_filename, preprocessed = preprocessed)

		char_to_id = self._build_vocab(clean_train_data)

		self._train_set = self._file_to_ids(clean_train_data, char_to_id)
		self._validataion_set= self._file_to_ids(clean_vaild_data, char_to_id)
		self._test_set = self._file_to_ids(clean_test_data, char_to_id)

		self._vocab_size = len(char_to_id)
		self._char_to_id = char_to_id




	def _readCorpus(self, filename, preprocessed):
		"""
		1. Read corpus file and do the following preprocessing tasks
		1.1 If the document is preprocessed i.e. start and stop Char has already been added,
			no changes need to be made on the document itself. Else, follow the folowing steps
		1.2 Chosen Start and Stop Character, which marks range of document with single context,
			must be chosen such that it is infrequent 
		1.3 Remove all previous instances of startChar and stopChar from the document
		1.4 It is assumed that each story or context sections within the document should be separated by newline character.
			If it's not replace the 'separatorChar' variable, but it should be a character
		1.5 Replace 'sepratorChar' with startChar and stopChar to mark range of each story within the document
		1.6 If custom preprocessing needs to be done, add it in the _customPreprocess(_) function
		"""
		
		self._startChar = startChar ='Å'
		self._stopChar = stopChar = '†'
		separatorChar = '\n'

		cleanDocument = ""
		if preprocessed == False:
			if os.path.isfile(filename):
				with open(filename, 'r') as f:
					document = f.read()
				cleanDocument = self._customPreprocess(cleanDocument)
				expungeString = startChar + stopChar
				cleanDocument = ''.join( c for c in document if  c not in expungeString)
				cleanDocument = startChar + cleanDocument.replace(separatorChar, stopChar + startChar)
			else:
				print("ERROR: ", filename, " File doesn't exist")
				return 
		else:
			if os.path.isfile(filename):
				with open(filename, 'r') as f:
					cleanDocument = f.read()
				#cleanDocument = self._customPreprocess(cleanDocument)
			else:
				print("ERROR: ", filename, " File doesn't exist")
				return 
		return cleanDocument


	def _build_vocab(self, cleanDocument):
		"""
		2. Read cleanDocument and do the following processing tasks
		2.1 Prepare list of distinct characters 
		2.2 Map each character to its ID
		"""

		#Get Frequency Distribution of characters in the document and sort them in descending order of Frequency
		counter = collections.Counter(cleanDocument)
		count_pairs = sorted(counter.items(), key = lambda x: (-x[1], x[0]))

		#Get list of characters
		characters, _ = list(zip(*count_pairs))
		self._id_to_char = characters
		#Get Mapping from character to ID
		char_to_id = dict(zip(characters, range(len(characters))))
		return char_to_id


	def _file_to_ids(self, cleanDocument, char_to_id):
		#Get Mapping from character to ID and assign integer ID to each character
		return [char_to_id[c] for c in cleanDocument if c in char_to_id]



	def _customPreprocess(self, cleanDoc):
		""" 
		Add additional code to change cleanDoc Here 
		Following changes are made prior to adding start and stop delimiters
		"""
		return cleanDoc


	@property
	def startChar(self):
		return self._startChar

	@property
	def stopChar(self):
		return self._stopChar

	@property
	def vocab_size(self):
		return self._vocab_size

	@property
	def char_to_id(self):
		return self._char_to_id

	@property
	def id_to_char(self):
		return self._id_to_char

	@property
	def train_set(self):
		return self._train_set

	@property
	def valid_set(self):
		return self._validataion_set

	@property
	def test_set(self):
		return self._test_set



def inputProducer(raw_data, batch_size, num_steps, name = None):
	"""
		Take corpus, divide it into batches and dequeue it
		raw_data: Clean Corpus with character replaced by ID
		batch_size: Size of batch of input processed at once
		num_steps: Number of timesteps over which backprop is done for RNN
		On epoch_size iterations, a single epoch is finished i.e all inputs are passed through,
			where each iteration feeds a matrix of shape [batch_size X num_steps]
		x,y : input and target pairs, target is input displaced to the right by a single timesteps
				i.e. one character in the future
	"""

	with tf.name_scope(name, "inputProducer", [raw_data, batch_size, num_steps]):
		#total count of characters represented as integers in range [0, vocab_size)
		data_len = len(raw_data)
		#Divide total data in batch_size counts, where each division has batch_len number of characters
		#i.e. batch_size rows and batch_len columns, using integer division to remove leftovers
		batch_len = data_len // batch_size
		data = np.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])
		#Convert to tensor
		label_data = tf.convert_to_tensor(data, name= "raw_data", dtype = tf.int32)

		epoch_size = (batch_len -1) // num_steps
		assertion = tf.assert_positive(epoch_size, message = "epoch_size == 0")
		with tf.control_dependencies([assertion]):
			epoch_size = tf.identity(epoch_size, name = "epoch_size")

		i = tf.train.range_input_producer(epoch_size, shuffle = False).dequeue()

		x = tf.strided_slice(label_data, [0, i*num_steps], [batch_size, (i+1)*num_steps])
		x.set_shape([batch_size, num_steps])
		y = tf.strided_slice(label_data, [0, i*num_steps +1], [batch_size, (i+1)*num_steps +1])
		y.set_shape([batch_size, num_steps])

	return x, y

# corpus = Corpus()
# _batch_size = 20
# input_data, targets = inputProducer(raw_data = corpus.train_set, batch_size = _batch_size, num_steps = 15)
# comp_num = tf.constant(value = corpus.char_to_id[corpus.startChar] ,dtype=tf.int32, shape = [_batch_size])
# vec_p = tf.reshape(tf.cast(tf.equal(comp_num, input_data[:,0]), tf.float32),shape = [_batch_size, 1])

# print(vec_p)
# with tf.Session() as sess:
# 	print("Input Data")
# 	print(sess.run(vec_p))
# 	print("StartCharID")
# 	print(sess.run(comp_num))






