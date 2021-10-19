import os
import tensorflow as tf
import numpy as np
import subscripts as sc



os.environ["CUDA_VISIBLE_DEVICES"] = "-2"
# 4 neurony na vstupnej vrstve
# 2 neurony na skrytej vrstve - full connected
# vystupna vrstva 4 neurony - full connected

# Set up layers
input_layer = tf.keras.Input(shape=(4,))
hidden = tf.keras.layers.Dense(2, activation= tf.nn.sigmoid)(input_layer)
output_layer = tf.keras.layers.Dense(4, activation=tf.nn.sigmoid)(hidden)

# Set up the model
model = tf.keras.Model(input_layer, output_layer)

# Set up the optimizer and loss
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.3, momentum=0.01),
    loss=tf.keras.losses.MeanSquaredError()
)

# Print model summary
model.summary()

# Train data
# Add sample [1,1,1,1] -> convergation to 0
train_x = [
    [1,1,1,1],
    [1,1,0,0],
    [0,0,1,1],
    [1,0,1,0],
    [0,1,0,1],
    [0,0,0,0]
]

# y for train samples
train_y = [
    [1,1,1,1],
    [1,1,0,0],
    [0,0,1,1],
    [1,0,1,0],
    [0,1,0,1],
    [0,0,0,0]
]

# Train the model
model.fit(x=train_x, y=train_y, batch_size=1,epochs=10_000, verbose=1)

# Test data
test_x = [

    [1,1,0,0],
    [0,0,1,1],
    [1,0,1,0],
    [0,1,0,1],
    [0,0,0,0],

    # [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    # [0,0,1,1],
    [0,1,0,0],
    # [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    # [1,0,1,0],
    [1,0,1,1],
    # [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0]
    # [1,1,1,1]
]


test_y = [
    # [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    # [0,0,1,1],
    [0,1,0,0],
    # [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    # [1,0,1,0],
    [1,0,1,1],
    # [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0]
    # [1,1,1,1]
]

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


# Test the model
print('Test the model')
print('Train samples')
print("Word\t\t\tExpected\t\tPredicted\t\tError\tAccuracy\tReliability")
for word, test_input, test_output in zip(train_x, train_x, train_y):
    prediction = model.predict([test_input], batch_size=1, verbose=0)
    loss = model.evaluate(x=[test_input], y=[test_output], batch_size=1, verbose=0)
    print(f"{word}\t\t{test_output}\t{prediction[0]}\t\t{float_formatter(loss)}\t{sc.accuracy(test_output, prediction[0])}\t\t{sc.reliability(prediction[0])}")


# Test the model
print('Test the model')
print('Test samples')
print("Word\t\t\tExpected\t\tPredicted\t\tError\tAccuracy\tReliability")
for word, test_input, test_output in zip(test_x, test_x, test_y):
    prediction = model.predict([test_input], batch_size=1, verbose=0)
    loss = model.evaluate(x=[test_input], y=[test_output], batch_size=1, verbose=0)
    print(f"{word}\t\t{test_output}\t{prediction[0]}\t\t{float_formatter(loss)}\t{sc.accuracy(test_output, prediction[0])}\t\t{sc.reliability(prediction[0])}")

