import os
import tensorflow as tf
import numpy as np
import random
import string
import accuracy as acc
import reliability as re
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.") 
parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.") 

parser.add_argument("--learning_rate", default=0.5, type=int, help="Learning_rate.") 
parser.add_argument("--momentum", default=0.01, type=int, help="Momentum.") 

# Hidden layer
parser.add_argument("--hidden_layers", default=1, type=int, help="Number of hidden layers.") 
parser.add_argument("--hidden_layer_size", default=30, type=int, help="Number of neurons in hidden layer.") 

# Word parameters
parser.add_argument("--max_word_length", default=8, type=int, help="Maximum word length.") 
parser.add_argument("--one_char_length", default=5, type=int, help="One character length.")

# Word weight
parser.add_argument("--positive_weight", default=4, type=int, help="Weight of positive sample.")
parser.add_argument("--negative_weight", default=1, type=int, help="Weight of negative sample")

# Negative word
parser.add_argument("--neg_add_1", default=3, type=int, help="Number of negative words: One letter add.")
parser.add_argument("--neg_modifications_1", default=5, type=int, help="Number of negative words: One letter changed.")
parser.add_argument("--neg_modifications_2", default=5, type=int, help="Number of negative words: Two letters changed.")


# Encode a string into a one dimensional array of ones and zeroes, where every letter is described as a one hot vector of size 5
def encode_string(args, word):

    word = word.upper()

    # sample length 
    result = [0] * args.max_word_length * args.one_char_length
    cur_idx = -1
    
    for character in word:
        number_value = ord(character) - ord('A') + 1
        binary_value = bin(number_value)
        
        for i in range(args.one_char_length, 0, -1):
            cur_idx += 1
            bin_idx = len(binary_value) - i
            
            if bin_idx < 2:
                continue
            
            result[cur_idx] = int(binary_value[bin_idx])
    
    return result

# Randomly generate negative samples: Try adding a random letter, removing the last letter, randomly modifying 1 and 2 letters
def generate_negative_samples(args, word):

    negative_words = []

    # removing the last letter
    negative_words.append(word[:-1])   

    # try adding a random letter
    if len(word) < args.max_word_length:    
      
        for _ in range(args.neg_add_1):
            negative_words.append(word + random.choice(string.ascii_lowercase))  

    # randomly modifying 1 letter
    modify_1_words = negative_word_modify_word(
        word= word,
        number_of_neg_word= args.neg_modifications_1, 
        number_of_modifications = 1,
    )
    
    # randomly modifying 2 letters
    modify_2_words = negative_word_modify_word(
        word= word,
        number_of_neg_word= args.neg_modifications_2, 
        number_of_modifications = 2,
    )

    negative_words.extend(modify_1_words)
    negative_words.extend(modify_2_words)

    return negative_words


def negative_word_modify_word(word, number_of_neg_word, number_of_modifications):
    
    # output negative words
    modify_words = []
    
    for _ in range(number_of_neg_word): 

        modify_word = list(word)

        # randomly modifying number_of_mofications letters
        for _ in range(number_of_modifications):
            
            rand_idx = random.randint(0, len(word) - 1)
            rand_letter = word[rand_idx]       

            while rand_letter == word[rand_idx]:
                rand_letter = random.choice(string.ascii_lowercase)  

            modify_word[rand_idx] = rand_letter

        modify_words.append("".join(modify_word)) 
    
    return modify_words
        
def generate_target_y(number_of_class):
    
    train_y = []
    
    for i in range(number_of_class):

        y = [0] * number_of_class
        y[i] = 1
        train_y.append(y)
    
    return train_y

def add_negative_samples(args, words, number_of_class):
    
    train_neg_x = []
    train_neg_y = []
    train_neg_weights = []

    for word in words:

        negative_words = generate_negative_samples(args, word)
        train_neg_x = train_neg_x + [encode_string(args, x) for x in negative_words]

        for _ in range(len(negative_words)):

            train_neg_weights.append(args.negative_weight)
            train_neg_y.append([0] * number_of_class)   
    
    return train_neg_x, train_neg_y, train_neg_weights
    
def main(args: argparse.Namespace) -> np.ndarray:
 
    # Train data
    words = [
        "hello",
        "world",
        "neurons",
        "apple",
        "biology",
        "yes",
    ]

    number_of_class = len(words)
    
    train_x = [encode_string(args, x) for x in words]

    # Positive samples have 4 times the weight of negative samples
    train_weights = [args.positive_weight] * len(words)

    # y for positive samples
    train_y = generate_target_y(number_of_class)

    # Add negative training data to training
    train_neg_x, train_neg_y, train_neg_weights = add_negative_samples(args, words, number_of_class)
    
    train_x.extend(train_neg_x)
    train_y.extend(train_neg_y)
    train_weights.extend(train_neg_weights)
    
    # Model
    # Set up layers
    hidden = input_layer = tf.keras.Input(shape=(args.max_word_length * args.one_char_length,))

    for _ in range(args.hidden_layers):
        hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.sigmoid)(hidden)
    
    output_layer = tf.keras.layers.Dense(number_of_class, activation=tf.nn.sigmoid)(hidden)

    # Set up the model
    model = tf.keras.Model(input_layer, output_layer)

    # Set up the optimizer and loss
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum),
        loss=tf.keras.losses.MeanSquaredError()
        )

    # Print model summary
    model.summary()

    # Train the model
    model.fit(x=train_x, y=train_y, sample_weight=train_weights, batch_size=args.batch_size, epochs=args.epochs, verbose=1, shuffle=True)

    # Test data
    test_x = train_x
    test_y = train_y

    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    # Test the model
    # Positive samples
    print("Positive samples:")
    print("Word\t\tExpected\t\tPredicted\t\t\t\tError\tAccuracy\tReliability")
    for word, test_input, test_output in zip(words, test_x, test_y):
        prediction = model.predict([test_input], batch_size=args.batch_size, verbose=0)
        loss = model.evaluate(x=[test_input], y=[test_output], batch_size=args.batch_size, verbose=0)

        if len(word) < args.max_word_length:
            print(f"{word}\t\t{test_output}\t{prediction[0]}\t\t{float_formatter(loss)}\t{acc.accuracy(test_output, prediction[0])}\t\t{re.reliability(prediction[0])}")
        else:
            print(f"{word}\t{test_output}\t{prediction[0]}\t\t{float_formatter(loss)}\t{acc.accuracy(test_output, prediction[0])}\t\t{re.reliability(prediction[0])}")

    # Negative samples
    print("Negative Samples:")
    print("Word\t\tPredicted")

    negative_words = [
        "network", 
        "python", 
        "nytrons", 
        "no", 
        "slovak",
        "yess", 
        "europe",
        "addle", 
        "biologi", 
        "worad",
        "helkos", 
        "yello", 
        "anqle"
    ]

    negative_samples = [encode_string(args, x) for x in negative_words]

    for word, negative_sample in zip(negative_words, negative_samples):
        prediction = model.predict([negative_sample], batch_size=args.batch_size, verbose=0)

        if len(word) < args.max_word_length:
            print(f"{word}\t\t{prediction[0]}")
        else:
            print(f"{word}\t{prediction[0]}")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    value_function = main(args)