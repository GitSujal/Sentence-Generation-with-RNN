import tensorflow as tf 
import numpy as np 
import reader
import time
import random



class PTBInput(object):
	"""Input Data"""
	def __init__(self, config, raw_data = None, name = None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		if raw_data is not None:
			self.epoch_size = ((len(raw_data) // batch_size) -1)	// num_steps
			self.input_data, self.targets = reader.inputProducer(raw_data, batch_size, num_steps, name = name)

class PTBModel(object):
	"""PTB Model"""
	def __init__(self, is_training, config, input_):
		self._is_training = is_training
		self._input = input_
		self.batch_size = input_.batch_size
		self.num_steps = input_.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size

		#Initialize one-hot encoding matrix
		embedding = tf.get_variable("embedding", [vocab_size, size], dtype = tf.float32)
		#input_data is batch_size X num_steps per iteration till epoch_size
		#inputs is of size batch_size X num_steps X hidden_size 
		inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

		if is_training and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		#Ouput is of shape [batch_size X size]
		output, state = self._build_rnn_graph_lstm(inputs, config, is_training)

		softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
		softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
		logits = tf.matmul(output, softmax_w) + softmax_b
     
		# Reshape logits to be a 3-D tensor for sequence loss
		logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

		self._logits = logits
		self._output_probs = tf.nn.softmax(logits) 

		loss = tf.contrib.seq2seq.sequence_loss(logits,input_.targets,tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),average_across_timesteps=False,average_across_batch=True)

		# Update the cost
		self._cost = cost = tf.reduce_sum(loss)
		self._final_state = state

		if not is_training:
			return

		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm)

		optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
		self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step = tf.train.get_or_create_global_step())


	def _get_lstm_cell(self, config, is_training):
		return tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True,reuse=not is_training)

	def _build_rnn_graph_lstm(self, inputs, config, is_training):
		def make_cell():
			cell = self._get_lstm_cell(config, is_training)
			#Using dropout
			if is_training and config.keep_prob < 1:
				cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = config.keep_prob)
			return cell

		#Multilayer RNN
		cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(config.num_layers)], state_is_tuple = True)

		#With state_is_tuple set to True, hidden layer consisting of cell and hidden-to-output states is represented by tuple (c,h)
		#So initial state has size num_layers X [batch_size X (h,c))*size]
		#With state_is_tuple set to false, initial state is represented by 
		#a concatenated matrix of shape [batch_size, num_layers * (h,c) * size]

		self._initial_state = cell.zero_state(config.batch_size, tf.float32)
		self._prime_initial_state = cell.zero_state(config.batch_size, tf.float32)	

		state = self._initial_state
		startCharTensor = tf.constant(value = config.startCharID, dtype = tf.int32, shape = [config.batch_size])

		#Outputs is a tensor of shape [batch_size X num_steps X size]
		#state is LSTM Tuple of shape [batch_size X size] for a sequence of hidden layers
		#Output is of shape [batch_size X num_steps] rows and [size] columns
		#Weight shared across all time steps (softmax) is operated on batch_size *num_steps character vectors
		#logits is of shape [batch_size * num_steps   vocab_size]
		initMatrix = tf.constant(value = 0.05, shape = [config.batch_size, config.hidden_size], dtype = tf.float32)
		initCell = tf.contrib.rnn.LSTMStateTuple(c = initMatrix, h = initMatrix)
		initMultiCell = tuple(initCell for i in range(config.num_layers))
		self._prime_initial_state = initMultiCell

		outputs = []
		with tf.variable_scope("RNN"):
			for time_step in range(self.num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()

				startCharMatchTensor = tf.reshape(tf.cast(tf.equal(startCharTensor, self._input.input_data[:, time_step]), tf.float32), shape = [config.batch_size, 1])
				startCharMismatchTensor = tf.reshape(tf.cast(tf.not_equal(startCharTensor, self._input.input_data[:, time_step]), tf.float32), shape = [config.batch_size, 1])
				state = tuple((tf.add(tf.multiply(self._prime_initial_state[i].c, startCharMatchTensor), tf.multiply(state[i].c, startCharMismatchTensor)), tf.add(tf.multiply(self._prime_initial_state[i].h, startCharMatchTensor), tf.multiply(state[i].h, startCharMismatchTensor))) for i in range(config.num_layers))

				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)
		output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
		return output, state


		"""
		#  Simplified Version
		inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
		outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,initial_state=self._initial_state)
		output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
		return output, state
		"""

	@property
	def input(self):
		return self._input

	@property
	def logits(self):
		return self._logits

	@property
	def train_op(self):
		return self._train_op

	@property
	def cost(self):
		return self._cost

	@property
	def output_probs(self):
		return self._output_probs

	@property
	def final_state(self):
		return self._final_state

	@property
	def initial_state(self):
		return self._initial_state


def run_epoch(session, model, generate_model, corpus, eval_op=None, verbose = False):
	"""
	Runs the model on the given data
	"""
	start_time = time.time()
	costs = 0.0
	iters = 0

	state = session.run(model.initial_state)
	for step in range(model.input.epoch_size):
		cost, state, _ = session.run([model.cost, model.final_state, model.train_op], {model.initial_state: state})

		costs += cost
		iters += model.input.num_steps

		if verbose and step % (model.input.epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" % ( step * 1.0 / model.input.epoch_size, np.exp(costs / iters), iters * model.input.batch_size / (time.time() - start_time)))
			print(GenerateSentence(session, generate_model, corpus))

	return np.exp(costs/iters)

def sample(a, temperature=1.0):
	a = np.log(a) / temperature
	a = np.exp(a) / np.sum(np.exp(a))
	r = random.random() # range: [0,1)
	total = 0.0
	for i in range(len(a)):
		total += a[i]
		if total>r:
			return i
	return len(a)-1 

def GenerateSentence(session, model, corpus, verbose = False):
	id_to_char = corpus.id_to_char
	startCharID = corpus.char_to_id[corpus.startChar]
	stopCharID = corpus.char_to_id[corpus.stopChar]

	state = session.run(model.initial_state)
	_input = np.matrix([[startCharID]])
	batchItr = 0
	batchSize = 500
	text = ""
	
	while batchItr < batchSize:
		output_probs, state = session.run([model.output_probs, model.final_state], {model.input.input_data : _input, model.initial_state:state})
		#primaryIndex = np.argpartition(output_probs[0][0], -10)[-10:]
		#x = random.choice(primaryIndex)
		x =  sample(output_probs[0][0], 0.8)		
		_input = np.matrix([[x]])
		if x == stopCharID:
			text += '\n'
		else:
			text += id_to_char[x] + ' '
		batchItr += 1
	return text


class TrainConfig(object):
	init_scale = 0.01
	learning_rate = 0.50
	vocab_size = 214
	max_grad_norm = 5
	hidden_size = 250
	keep_prob = 0.5
	batch_size = 20
	num_steps = 40
	num_layers = 2
	max_max_epoch = 2
	startCharID = 0
	stopCharID = 0

class GenerateConfig(object):
	init_scale = 0.01
	learning_rate = 0.50
	max_grad_norm = 5
	vocab_size = 214
	keep_prob = 1.0
	hidden_size = 250
	batch_size = 1
	num_steps = 1
	num_layers = 2
	startCharID = 0
	stopCharID = 0


def main(_):
	print("Start")
	print("Preparing Corpus")
	corpus = reader.Corpus()
	startCharID = corpus.char_to_id[corpus.startChar]
	stopCharID = corpus.char_to_id[corpus.stopChar]

	print("Getting Configurations")
	train_config = TrainConfig()
	train_config.vocab_size = corpus.vocab_size
	train_config.startCharID = startCharID
	train_config.stopCharID = stopCharID
	generate_config = GenerateConfig()
	generate_config.vocab_size = corpus.vocab_size
	generate_config.startCharID = startCharID
	generate_config.stopCharID = stopCharID

	print(train_config.vocab_size)
	print(train_config.startCharID)
	print(train_config.stopCharID)


	print("Setting up Graph")
	with tf.Graph().as_default():
	 	#initializer = tf.random_uniform_initializer(-train_config.init_scale, train_config.init_scale)
	 	initializer = tf.contrib.layers.xavier_initializer()
	 	print("Train")
	 	with tf.name_scope("Train"):
	 		train_input = PTBInput(config = train_config, raw_data = corpus.train_set, name = "TrainInput")
	 		with tf.variable_scope("Model", reuse = None, initializer= initializer):
	 			train_model = PTBModel(is_training = True, config = train_config, input_=train_input)
	 		tf.summary.scalar("Training Loss", train_model.cost)

	 	with tf.name_scope("Valid"):
	 		valid_input = PTBInput(config = train_config, raw_data = corpus.valid_set, name = "ValidInput")
	 		with tf.variable_scope("Model", reuse = True, initializer = initializer):
	 			valid_model = PTBModel(is_training = False, config = train_config, input_=valid_input)
	 		tf.summary.scalar("Validation Loss", valid_model.cost)

	 	with tf.name_scope("Test"):
	 		test_input = PTBInput(config = generate_config, raw_data = corpus.test_set, name = "TestInput")
	 		with tf.variable_scope("Model", reuse = True, initializer = initializer):
	 			test_model = PTBModel(is_training = False, config = generate_config, input_ = test_input)

	 	with tf.name_scope("Generate"):
	 		generate_input = PTBInput(config = generate_config, raw_data = corpus.test_set, name = "GenerateInput")
	 		with tf.variable_scope("Model", reuse = True, initializer = initializer):
	 			generate_model = PTBModel(is_training = False, config = generate_config, input_ = generate_input) 

	 	models = {"Train":train_model, "Valid":valid_model, "Test":test_model, "Generate":generate_model}
	 	print("Executing Graph")
	 	with tf.Session() as sess:
	 		saver = tf.train.Saver()
	 		coord = tf.train.Coordinator()
	 		threads = tf.train.start_queue_runners(sess = sess, coord = coord)
	 		sess.run(tf.global_variables_initializer())
	 		saver.restore(sess, 'model/savedModelValidL2H250N40-1000')
	 		for i in range(train_config.max_max_epoch):
	 			train_perplexity = run_epoch(session = sess, model = train_model, generate_model = generate_model, corpus = corpus, eval_op = train_model.train_op, verbose = True)
	 			print("Epoch %d Train perplexity %.3f" % (i+1, train_perplexity))
	 			genDoc = GenerateSentence(session=sess, model=generate_model, corpus=corpus, verbose = False)
		 		print(genDoc)
		 		saver.save(sess, "model/savedModelValidL2H250N40", global_step = 1000) 
		 		
	 		coord.request_stop()
	 		coord.join(threads)

if __name__ == "__main__":
	tf.app.run()








