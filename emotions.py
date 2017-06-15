import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import gensim, logging, time, lasagne, theano
from re import findall
import mnist
import sys, os
import theano.tensor as T

start = time.time()

theano.config.floatX = 'float32'
#theano.config.exception_verbosity = "high"

def vectorize(words):
	if len(words) > 28:
		words = words[:28]
	if len(words) < 28:
		words = words + [''] * (28 - len(words))
	result = []
	for word in words:
		if word in model.wv.vocab:
			result.append(model[word].astype(np.float32))
		else:
			result.append(np.zeros(28, dtype = np.float32))
	return np.array([result], dtype = np.float32)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]	

X_train_not_vectorized = []
y_train = []
X_test_not_vectorized = []
y_test = []
types = ['sberbank', 'vtb', 'gazprom', 'alfabank', 'bankmoskvy', 'raiffeisen', 'uralsib']
train_address = 'database/bank_train_2016.xml'
test_address = 'database/banks_test_etalon.xml'

for type in types:
	tree = ET.parse(train_address)
	root = tree.getroot()
	tables = root.find('./database')
	for bank_tables in tables.findall('table'):
		if bank_tables.find("./*[@name='%s']" % type).text != "NULL":
			X_train_not_vectorized.append(bank_tables.find('./column[4]').text)
			y_train.append(bank_tables.find("./*[@name='%s']" % type).text)

for type in types:
	tree = ET.parse(test_address)
	root = tree.getroot()
	tables = root.find('./database')
	for bank_tables in tables.findall('table'):
		if bank_tables.find("./*[@name='%s']" % type).text != "NULL":
			X_test_not_vectorized.append(bank_tables.find('./column[4]').text)
			y_test.append(bank_tables.find("./*[@name='%s']" % type).text)					

X_train_not_vectorized = [findall('[А-Яа-яЁё]+', x) for x in X_train_not_vectorized]
X_test_not_vectorized = [findall('[А-Яа-яЁё]+', x) for x in X_test_not_vectorized]
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
if 'save' in sys.argv:
	model = gensim.models.Word2Vec(X_train_not_vectorized, min_count = 1, workers = 4, size = 28)
	model.save(os.getcwd() + '/tmp/mymodel')
else:
	model = gensim.models.Word2Vec.load(os.getcwd() + '/tmp/mymodel')

model.init_sims(replace=True)

X_train = np.array([vectorize(x) for x in X_train_not_vectorized], dtype = np.float32)
y_train = np.array([int(x) for x in y_train], dtype = np.int32) + 1
X_test = np.array([vectorize(x) for x in X_test_not_vectorized], dtype = np.float32)
y_test = np.array([int(x) for x in y_test], dtype = np.int32) + 1
X_train, X_val = X_train[:-1600], X_train[-1600:]
y_train, y_val = y_train[:-1600], y_train[-1600:]

#построил сеть, но пока не обучаю её
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var = input_var)
network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify)
network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify)
network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=3, nonlinearity=lasagne.nonlinearities.softmax)
#дальше переделать, т. к. это остатки старого перцептрона
# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
num_epochs = 100
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
# We could add some weight decay as well here, see lasagne.regularization.

# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates, on_unused_input = 'warn')

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc], on_unused_input = 'warn')

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
for epoch in range(num_epochs):
	# In each epoch, we do a full pass over the training data:
	train_err = 0
	train_batches = 0
	start_time = time.time()
	for batch in iterate_minibatches(X_train, y_train, 500, shuffle=False):
		inputs, targets = batch
		train_err += train_fn(inputs, targets)
		train_batches += 1
	# And a full pass over the validation data:
	val_err = 0
	val_acc = 0
	val_batches = 0
	for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
		inputs, targets = batch
		err, acc = val_fn(inputs, targets)
		val_err += err
		val_acc += acc
		val_batches += 1
	# Then we print the results for this epoch:
	print("Epoch {} of {} took {:.3f}s".format(
		epoch + 1, num_epochs, time.time() - start_time))
	print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
	print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
	print("  validation accuracy:\t\t{:.6f} %".format(
		val_acc / val_batches * 100))

# After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
	inputs, targets = batch
	err, acc = val_fn(inputs, targets)
	test_err += err
	test_acc += acc
	test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
	test_acc / test_batches * 100))

end = time.time()
duration = end - start
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = int(duration % 60)
print("The whole process took %d hours, %d minutes and %d seconds" % (hours, minutes, seconds))
