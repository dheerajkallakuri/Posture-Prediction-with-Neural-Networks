import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# Reading thee csv file
gdata = pd.read_csv('sensor_data/data.csv', header= 0)
# Shuffling the data rows of csv
shuffled_data = gdata.sample(frac=1).reset_index(drop=True)

# Assign the first three columns to x
x = shuffled_data.iloc[:, :3]

# Assign the fourth column to y
y = shuffled_data.iloc[:, 3]

SAMPLEDATA=len(shuffled_data)

# Splitting data as 60% training, 20% each for test and validation 
Train_split=int(0.6*SAMPLEDATA)
Test_split=int(0.2*SAMPLEDATA+Train_split)

x_train,x_validate,x_test=np.split(x,[Train_split,Test_split])
y_train,y_validate,y_test=np.split(y,[Train_split,Test_split])
# Declaring the model
model_1=Sequential()
model_1.add(Dense(6,activation='relu',input_shape=(3,))) # Input layer and first layer
model_1.add(Dense(16,activation='relu')) # Input layer and first layer
model_1.add(Dense(32,activation='relu')) # Input layer and first layer
model_1.add(Dense(16,activation='relu')) # Input layer and first layer
model_1.add(Dense(5)) # Output layer
model_1.compile(optimizer='adam',loss='mse',metrics=['mae']) # Declaring the data model paramaters
model_1.summary()

# Model training on training data ans validating on validation data
history_1= model_1.fit(x_train,y_train ,epochs=200,batch_size=10,validation_data=(x_validate,y_validate))



# Extract the loss values for training and validation data from the history
training_loss = history_1.history['loss']
validation_loss = history_1.history['val_loss']

# Creating a plot to visualize training and validation loss
epochs_loss = range(1, len(training_loss) + 1)
plt.plot(epochs_loss, training_loss, 'r.',alpha=0.2, label='Training Loss')
plt.plot(epochs_loss, validation_loss, 'b.',alpha=0.5, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Extract the loss values for training and validation data from the history
mae=history_1.history['mae']
val_mae=history_1.history['val_mae']

# Creating a plot to visualize training and validation loss
epochs_mae = range(1, len(mae) + 1)

epochs_mae = range(1, len(mae) + 1)
plt.plot(epochs_mae, mae, 'r.',alpha=0.2, label='Training MAE')
plt.plot(epochs_mae, val_mae, 'b.',alpha=0.5, label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()
# Prediction on test data
predictions=model_1.predict(x_test)
y_pred_classes=list()
for i in range(len(predictions)):
    rval=int(np.round(max(predictions[i])))
    if rval == 0:rval=1
    if rval == 6:rval=5
    y_pred_classes.append(rval)

# Accuracy calcuation of the the test model
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Test Accuracy: {accuracy}')
# Plotting the prediction vs y_test values for x_test data
plt.clf()
plt.title('Predict vs Actual')
plt.plot(x_test, y_test,'b*',alpha=0.2,label='Actual')
plt.plot(x_test, y_pred_classes,'r.',alpha=0.5,label='Predict')
plt.legend()
plt.show()
converter= tf.lite.TFLiteConverter.from_keras_model(model_1)
tflite_model=converter.convert()

open("model_pred.tflite","wb").write(tflite_model)
converter=tf.lite.TFLiteConverter.from_keras_model(model_1)
converter.optimizations=[tf.lite.Optimize.DEFAULT]

def representative_dataset_generator():
    for value in x_test.values:
         yield [np.array(value, dtype=np.float32).reshape(1, -1)]
converter.representative_dataset=representative_dataset_generator

tflite_model=converter.convert()

open("model_pred_quant.tflite","wb").write(tflite_model)

# Instantiate an interpreter for each model
model_pre = tf.lite.Interpreter('model_pred.tflite')
model_quantized = tf.lite.Interpreter('model_pred_quant.tflite')

# Allocate memory for each model
model_pre.allocate_tensors()
model_quantized.allocate_tensors()

# Get indexes of the input and output tensors
model_input_index = model_pre.get_input_details()[0]["index"]
model_output_index = model_pre.get_output_details()[0]["index"]
model_quantized_input_index = model_quantized.get_input_details()[0]["index"]
model_quantized_output_index = model_quantized.get_output_details()[0]["index"]

# Create arrays to store the results
model_predictions = []
model_quantized_predictions = []

# Run each model's interpreter for each value and store the results in arrays
y_lite_pred=list()
y_quant_lite_pred=list()
for x_value in x_test.values:
    # Create a 2D tensor wrapping the current x value
    x_value_tensor = tf.convert_to_tensor([x_value], dtype=np.float32)
    # Write the value to the input tensor
    model_pre.set_tensor(model_input_index, x_value_tensor)
    # Run inference
    model_pre.invoke()
    # Read the prediction from the output tensor
    templist=model_pre.get_tensor(model_output_index)[0].tolist()
    rval=int(np.round(max(templist)))
    if rval == 0:rval=1
    if rval == 6:rval=5
    y_lite_pred.append(rval)

    # Do the same for the quantized model
    model_quantized.set_tensor(model_quantized_input_index, x_value_tensor)
    model_quantized.invoke()
    templist1=model_quantized.get_tensor(model_quantized_output_index)[0].tolist()
    rval1=int(np.round(max(templist1)))
    if rval1 == 0:rval1=1
    if rval1 == 6:rval1=5
    y_quant_lite_pred.append(rval1)

model_predictions=y_lite_pred
model_quantized_predictions=y_quant_lite_pred

# See how they line up with the data
plt.clf()
plt.title('Comparison of various models against actual values')
plt.plot(x_test, y_test,'b*',alpha=0.2,label='Actual')
plt.plot(x_test, y_pred_classes,'r.',alpha=0.5,label='Original predictions')
plt.plot(x_test, model_predictions, 'ro', label='Lite predictions')
plt.plot(x_test, model_quantized_predictions, 'gx', label='Lite quantized predictions')
plt.legend()
plt.show()
