import utils_multiMNIST as U

import numpy as np
import h5py
np.random.seed(12321)  # for reproducibility
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import backend as K

path_to_data_dir = '../Datasets/'
use_mini_dataset = True
K.set_image_dim_ordering('th')

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28
# input image dimensions
X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)


#print ("X_train first element", )




def main():
	X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)
    #Investigate the test set, training set, and the labels to get intuition about the representation of the data
	
	#=================== Model ======================#
	# TO DO: Define a model with input the first image and outputs the two digits. The model variable should be called model and should be based on 
	# a multilayer fully connected network. Consult https://keras.io/getting-started/functional-api-guide/ for the relevant API.

	# model = Model(input=digit_input, output=[prediction1, prediction2])

	inputs = Input(shape=(1,42,28))
	flat = Flatten()(inputs)
	x = Dense(64, activation='relu')(flat)
	prediction1 = Dense(10, activation='softmax')(x)
	prediction2 = Dense(10, activation='softmax')(x)

	model = Model(input=inputs, output=[prediction1,prediction2])
	


	# input -> FC(64) -> ReLU -> 2*(FC(10) -> SoftMax)


	model.compile(loss='categorical_crossentropy',optimizer='sgd',  metrics=['accuracy'], loss_weights=[0.5, 0.5])
	model.fit(X_train, [y_train[0], y_train[1]], nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
	objective_score = model.evaluate(X_test, y_test, batch_size=batch_size) #TO  BE COMPLETED.
	print('Evaluation on test set:', dict(zip(model.metrics_names, objective_score)))
	
	#Uncomment the following line if you would like to save your trained model
	#model.save('./current_model_conv.h5')
	if K.backend()== 'tensorflow':
		K.clear_session()

if __name__ == '__main__':
	main()
