# Preamble 

import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras

trainfile = 'training.csv'
testfile = 'test.csv'

# Create pandas dataframes from csv files 
dftrain = pd.read_csv(trainfile)
dftest = pd.read_csv(testfile)

# Create a shuffled dataframe 
dftrain_shuf = dftrain.sample(frac=1)

#                      Feature Selection 
#--------------------------------------------------------------

# Declare variables to be trained based on which variables have clear differences in topology 
# between continuum and non-continuum events

features = ['B_R2', 'B_p', 'B_thrustAxisCosTheta', 'B_CC9', 'B_CC8', 'B_CC7', 'B_CC6', 'B_CC4',
            'B_CC3', 'B_CC2', 'B_CC1', 'B_KSFWV_hso14', 'B_KSFWV_hso12']
target = ['B_isContinuumEvent']

training = dftrain_shuf[features]
target = dftrain_shuf[target]

#                       Create Training Set                        
#--------------------------------------------------------------
# NumPy arrays fit naturally into tensorflow/keras ML models

xtrain = training.to_numpy()
ytrain = target.to_numpy()

#                       The Model
#--------------------------------------------------------------

def buildmodel(n_hidden = 2, n_neurons = 64, learning_rate = 0.01, input_shape = [13]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))

    model.add(keras.layers.Dense(1,activation='sigmoid'))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss= 'binary_crossentropy')

    return model      

# create training model instance and graph it 

model = buildmodel()

#                       Training
#--------------------------------------------------------------

# train model 

# ReduceLROnPlateau is a callback that reduces the learning rate when
# the model doesnt find an improvement in loss after 5 epochs, reducing
# learning rate by a factor of 0.2. Can help avoid getting stuck in local mins

# EarlyStopping stops training after (patience) epochs, reducing
# overfitting

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
my_callbacks = [keras.callbacks.EarlyStopping(patience=10), reduce_lr]
training_history = model.fit(xtrain, ytrain, epochs=100, validation_split=0.2, callbacks=[my_callbacks], verbose=2)

# Save the model 
model.save("continuum_model")


# Plot the training history 

pd.DataFrame(training_history.history).plot(figsize=(20, 15))
plt.grid(True)
plt.ylim(top=1.2)
plt.ylim(bottom=0)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Training History", fontsize=16)
plt.xlabel('Epoch', fontsize=16)
fname = "continuum_model_traininghistory.png"
plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', transparent=False, bbox_inches=None, pad_inches=0.1)
plt.show()

# save training history 
df_training_history = pd.DataFrame(training_history.history)
filename_csv = "continuum_model_traininghistory.csv"
df_training_history.to_csv(filename_csv)

# Prepare test data for evaluation

xtest = dftest[features].to_numpy()

# Evaluate model based on test data
y_predict_test = model.predict(xtest)

# Append evaluation results to dataframe with test data 
dftest['B_isContinuumEvent'] = y_predict_test

# Save only needed columns
vars = ['Id', 'B_isContinuumEvent']
df_final = dftest[vars]

# Save to csv 
df_final.to_csv('final.csv', index=False)




