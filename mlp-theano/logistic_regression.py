import pickle
import gzip
import numpy as np
import theano
import theano.tensor as tt
import matplotlib
import matplotlib.pyplot as plt

# load MNIST
########################################################################

with gzip.open('mnist.pkl.gz', 'rb') as f:
	try:
		train_xy, valid_xy, test_xy = pickle.load(f, encoding='latin1')
	except:
		train_xy, valid_xy, test_xy = pickle.load(f)

def shared_dataset(data_xy):
	data_x, data_y = data_xy
	shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
	shared_y = theano.shared(np.asarray(data_y, dtype='int32'), borrow=True)
	return shared_x, shared_y

train_x, train_y = shared_dataset(train_xy)
valid_x, valid_y = shared_dataset(valid_xy)
test_x, test_y = shared_dataset(test_xy)

# logistic regression
########################################################################

rng=np.random.RandomState(1337)

# x and y (symbolic)
x = tt.tensor(name='x', dtype=theano.config.floatX, broadcastable=(False,False,))
y = tt.tensor(name='y', dtype='int32', broadcastable=(False,))

params = []
L1 = 0.0
L2 = 0.0

n_in = 28*28
n_out = 100

# weights (shared)
w = theano.shared(name='w', borrow=True,
	value=np.asarray(rng.uniform(
		low=-np.sqrt(6. / (n_in + n_out)), # Glorot initialisation
		high=np.sqrt(6. / (n_in + n_out)),
		size=(n_in, n_out)
	), dtype=theano.config.floatX))

# bias (shared)
b = theano.shared(name='b', borrow=True,
	value=np.zeros(shape=(n_out, ), dtype=theano.config.floatX))

# model parameters (symbolic)
params = params + [w, b]
L1 += tt.abs_(w).sum()
L2 += tt.power(w, 2).sum()

# softmax output (symbolic)
h = tt.nnet.sigmoid(tt.dot(x, w) + b)

n_in = 100
n_out = 10

# weights (shared)
w = theano.shared(name='w2', borrow=True,
	value=np.asarray(rng.uniform(
		low=-np.sqrt(6. / (n_in + n_out)), # Glorot initialisation
		high=np.sqrt(6. / (n_in + n_out)),
		size=(n_in, n_out)
	), dtype=theano.config.floatX))

# bias (shared)
b = theano.shared(name='b2', borrow=True,
	value=np.zeros(shape=(n_out, ), dtype=theano.config.floatX))

# model parameters (symbolic)
params = params + [w, b]
L1 += tt.abs_(w).sum()
L2 += tt.power(w, 2).sum()

# softmax output (symbolic)
p_y_given_x = tt.nnet.softmax(tt.dot(h, w) + b)
y_pred = tt.argmax(p_y_given_x, axis=1)

# negative log-likelihood (symbolic)
lambda_1 = theano.shared(value=0.000)
lambda_2 = theano.shared(value=0.0001)

cost_negll = -tt.mean(tt.log(p_y_given_x)[tt.arange(y.shape[0]), y])

# Hamming loss (symbolic)
cost_hamming = tt.mean(tt.neq(y_pred, y))

# parameter gradients (symbolic)
grads = tt.grad(cost=cost_negll + lambda_1 * L1 + lambda_2 * L2, wrt=params)

# learning rate (symbolic)
lr = tt.scalar(name='lr', dtype=theano.config.floatX)

momentum = 0.9

# weights (shared)
velocities = [theano.shared(name='v', borrow=True,
	value=np.zeros(shape=p.get_value().shape, dtype=theano.config.floatX)) for p in params]

# SGD parameter updates (symbolic)
updates = []
for p, g, vel in zip(params, grads, velocities):	
	updates.append((vel, vel*momentum + (1-momentum) * g))
	updates.append((p, p - lr * vel))

	

# compilation
########################################################################

predict_y = theano.function(
	inputs=[x],
	outputs=y_pred
)

# mini-batch indexes (symbolic)
indexes = tt.vector(name='indexes', dtype='int64')

train_batch = theano.function(
	inputs=[indexes, lr],
	outputs=[cost_negll, cost_hamming],
	updates=updates,
	givens={
		x: train_x[indexes],
		y: train_y[indexes]
	}
)

eval_valid_batch = theano.function(
	inputs=[indexes],
	outputs=[cost_negll, cost_hamming],
	givens={
		x: valid_x[indexes],
		y: valid_y[indexes]
	}
)

eval_test_batch = theano.function(
	inputs=[indexes],
	outputs=[cost_negll, cost_hamming],
	givens={
		x: test_x[indexes],
		y: test_y[indexes]
	}
)

# training
########################################################################

n_train = train_x.get_value(borrow=True).shape[0]
n_valid = valid_x.get_value(borrow=True).shape[0]
n_test = test_x.get_value(borrow=True).shape[0]

train_indexes = np.arange(n_train)
valid_indexes = np.arange(n_valid)
test_indexes = np.arange(n_test)

batch_size = 50
n_train_batches = n_train // batch_size
n_valid_batches = n_valid // batch_size
n_test_batches = n_test // batch_size

n_epochs = 100
learning_rate = 0.1
for epoch in range(n_epochs):

	# shuffle minibatches
	rng.shuffle(train_indexes)

	# train / compute training cost
	train_cost = np.mean([
		train_batch(train_indexes[i * batch_size: (i + 1) * batch_size], learning_rate)
		for i in range(n_train_batches)
	], axis=0)
	
	# compute validation cost
	valid_cost = np.mean([
		eval_valid_batch(valid_indexes[i * batch_size: (i + 1) * batch_size])
		for i in range(n_valid_batches)
	], axis=0)

	print('epoch %i, training cost %f/%f, validation cost %f/%f' %
		  (epoch + 1, train_cost[0], train_cost[1], valid_cost[0], valid_cost[1]))

# testing
########################################################################
	
# compute test cost
test_cost = np.mean([
	eval_test_batch(test_indexes[i * batch_size: (i + 1) * batch_size])
	for i in range(n_test_batches)
], axis=0)

print('Test set cost %f/%f' % (test_cost[0], test_cost[1]))


# plot some mistalen digits
test_x_val = test_x.get_value(borrow=True)
test_y_val = test_y.get_value(borrow=True)
pred_y_val = predict_y(test_x_val)
wrong_idx = [i for i, x in enumerate(test_y_val != pred_y_val) if x]

n_plot = 50
n_rows = 10
n_cols = n_plot // 10

fig, sub = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(1.5 * n_rows, 2.0 * n_cols))

for i in range(n_plot):

    j = wrong_idx[i]

    title = "y=%i h(x)=%i" % (test_y_val[j], pred_y_val[j])
    img = test_x_val[j].reshape(28, 28)
    
    c = i // n_rows
    r = i - c * n_rows
    sub[r, c].axis("off")
    sub[r, c].imshow(img, extent=[0,100,0,1], aspect=100 , cmap=plt.cm.gray, vmin=0., vmax=1.)
    sub[r, c].set_title(title)

plt.suptitle('MNIST test set error: %f' % (test_cost[1]))
plt.subplots_adjust(left=0.02, right=.98, top=0.92, bottom=0.02, wspace = 0.05, hspace = 0.4)
plt.show()
