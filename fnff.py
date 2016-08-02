__author__ = "hmartinez"

"""
A first version of a simple feedforward NN for FN classification
Heavily based on Barbara Plank's example FF nn
"""
import numpy as np
np.random.seed(113) #set seed before any keras import
import argparse
from keras.utils import np_utils
#from keras.models import Sequential
#from keras.layers.core import Dense, Activation, Dropout

from keras.layers import Input, Embedding, LSTM, Dense, merge, Reshape, Flatten
from keras.models import Model



def get_index(word, index_from, word2idx, freeze=False):
    if word in word2idx:
        return word2idx[word]
    else:
        if not freeze:
            word2idx[word]=len(word2idx) #new index
            return word2idx[word]
        else:
            return word2idx["_UNK"]

def load_data(trainfile, devfile, testfile):
    ### load data
    train_sents, train_y = load_animacy_sentences_and_labels(trainfile)
    dev_sents, dev_y = load_animacy_sentences_and_labels(devfile)
    test_sents, test_y = load_animacy_sentences_and_labels(testfile)

    ### create mapping word to indices
    word2idx = {"_UNK": 0}  # reserve 0 for OOV
    index_from = len(word2idx)

    ### convert training etc data to indices
    X_train = [[get_index(w, index_from,word2idx) for w in x] for x in train_sents]
    freeze=True
    X_dev = [[get_index(w, index_from,word2idx,freeze) for w in x] for x in dev_sents]
    X_test = [[get_index(w, index_from,word2idx,freeze) for w in x] for x in test_sents]


    ### convert labels to one-hot
    label2idx = {label: i for i, label in enumerate(set(train_y+test_y+dev_y))}
    num_labels = len(label2idx.keys())
    train_y = np_utils.to_categorical([label2idx[label] for label in train_y], nb_classes=num_labels)
    dev_y = np_utils.to_categorical([label2idx[label] for label in dev_y], nb_classes=num_labels)
    test_y = np_utils.to_categorical([label2idx[label] for label in test_y], nb_classes=num_labels)

    return X_train, train_y, X_dev, dev_y, X_test, test_y, word2idx, label2idx

def load_animacy_sentences_and_labels(datafile):
    """
    loads the data set
    """
    input = [line.strip().split("\t") for line in open(datafile)]
    sentences = [[w] for w,label in input]
    labels = [label for sentence, label in input]
    return sentences, labels

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default="data/fn_train.tsv")
    parser.add_argument('--dev', default="data/fn_dev.tsv")
    parser.add_argument('--test', help="",default='data/fn_test.tsv')
    parser.add_argument('--iters', help="epochs (iterations)", type=int, default=10)
    parser.add_argument('--featcolums',default=1)
    args = parser.parse_args()
    
    ## read input data
    print("load data..")
    X_train, y_train, X_dev, y_dev, X_test, y_test, word2idx, tag2idx = load_data(args.train, args.dev, args.test)
    print(X_train[:3])

    print("#train instances: {}\n#test instances: {}\n#dev instances: {}".format(len(X_train),len(X_test), len(X_dev)))
    assert(len(X_train)==len(y_train))
    assert(len(X_test)==len(y_test))
    assert(len(X_dev)==len(y_dev))

    vocabulary_size=len(word2idx.keys())
    num_labels = len(tag2idx)
    input_size = len(X_train[0])

    print("build model")
    print(input_size)

    ### now we'll use the functional API:
    # each layer is a function and can be applied to another layer
    # you can name each layer: name='main_input'

    # input: a sequence  of 5 integers, each representing a word (index between 0 and vocab_size).
    main_input = Input(shape=(args.featcolums,), dtype='int32', name='main_input')

    # now the embedding layer will encode the input sequence
    # into a sequence of dense 128-dimensional vectors.
    embeds = Embedding(output_dim=128, input_dim=vocabulary_size, input_length=args.featcolums)(main_input)
    flatten = Flatten()(embeds) # we flatten it as Dense expects a 2D input
    dense = Dense(64, activation='tanh')(flatten)

    # finally the softmax (logistic) output layer
    main_loss = Dense(num_labels, activation='softmax', name='main_output')(dense)

    # the model is specified by connecting input and output
    model = Model(input=[main_input], output=[main_loss])

    print("compile model..")
    model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])
    print("fit model..")
    #model.fit(X_train, y_train,nb_epoch=args.iters,batch_size=100) #, validation_data=(X_dev, y_dev))

    score = model.evaluate(X_test, y_test)
    print("evaluate model..")
    score, acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', acc)
    
if __name__ == "__main__":
    main()
