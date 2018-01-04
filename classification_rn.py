"""This module builds the neural networks aimed at desambiguating
occurences of the verbs in context."""

import read_data
import word2vec
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

import parse_syntax


def make_dataset(gold, vocab_code, train_p=0.8, mode="linear", pos_ignored=['PUNCT'], ctx_size=2):
    """
    Creates the datasets for training the classifier on linear context.
    """
    train_data, test_data = read_data.divide_data_in_train_test(gold, train_p)
    if mode == "linear":
        train, test = {}, {}

        for verb in train_data:
            x_train, y_train = [], []
            for occ in train_data[verb]:
                linear_ctx = read_data.get_linear_ctx(train_data[verb][occ]["conll"],
                                                      pos_ignored)
                coded_ctx = read_data.linear_ctx_2_cbow(linear_ctx,
                                                        vocab_code, ctx_size)
                x_train.append(coded_ctx)
                y_train.append(train_data[verb][occ]["classe"])
            train[verb] = [np.array(x_train), np.array(y_train)]

        for verb in test_data:
            x_test, y_test = [], []
            for occ in test_data[verb]:
                linear_ctx = read_data.get_linear_ctx(test_data[verb][occ]["conll"],
                                                      pos_ignored)
                coded_ctx = read_data.linear_ctx_2_cbow(linear_ctx,
                                                        vocab_code, ctx_size)
                x_test.append(coded_ctx)
                y_test.append(test_data[verb][occ]["classe"])
            test[verb] = [np.array(x_test), np.array(y_test)]

    return train, test

def code_embeddings(model_file, vocab):
    """
    Extracts a weight matrix from the pre-trained word embeddings making
    up our vocabulary.
    """
    model_embeddings = word2vec.load(model_file, kind="txt")
    list_arrays = []
    for word in vocab:
        try:
            list_arrays.append(model_embeddings[word])
        except KeyError:
            list_arrays.append(np.zeros(100))
    embeddings = np.array(list_arrays, dtype=object)
    return embeddings

if __name__ == "__main__":

    GOLD_DIR = "../data_WSD_VS"
    CONLL_FILE = "../data_WSD_VS/abattre.deep_and_surf.sensetagged.conll"
    MODEL_FILE = "../vecs100-linear-frwiki/data"
    CLASSES = {"aborder":4, "affecter":4, "abattre":5}
    RESULTS = {}

    TRAIN_PERCENTAGE = 0.8
    POS_IGNORED = ['PUNCT']
    SIZE_VOCAB = 400
    update_weights = True # Fine-uning the weights for the word embeddings
    use_word_embeddings = True # pre-trained embeddings
    NB_EPOCHS = 15

    gold_data = read_data.load_gold(GOLD_DIR, CLASSES)
    train_mfs, test_mfs = read_data.divide_data_in_train_test(gold_data)
    MFS = read_data.most_frequent_sense(train_mfs, test_mfs)

    dico_code, dico_code_reverse = read_data.make_vocab_dico(GOLD_DIR, SIZE_VOCAB)
    # LINEAR
    # train, test = make_dataset(gold_data, dico_code)
    # print(train)

    # SYNTACTIC # TODO : check why the format isn't working with the classifier
    train, test = read_data.divide_data_in_train_test(gold_data)
    train, dico_code= parse_syntax.make_dataset_syntax(train, dico_code)
    test = parse_syntax.make_dataset_syntax(test, dico_code, False)
    train = parse_syntax.normalise_dataset_syntactic_contexte(train, dico_code)
    test = parse_syntax.normalise_dataset_syntactic_contexte(test, dico_code)

    SIZE_VOCAB = len(dico_code) # updating for syntactic mode

    # EMBEDDINGS
    embeddings = code_embeddings(MODEL_FILE, dico_code)

    for verb in ["abattre", "affecter", "aborder"]:

        RESULTS[verb] = {}
        nb_neuron_output = CLASSES.get(verb)

        x_train, y_train = train[verb]
        x_test, y_test = test[verb]


        # TEST avec CBOW
        model = Sequential()
        if use_word_embeddings:
            model.add(Embedding(len(dico_code), 100, input_shape=(SIZE_VOCAB,),
                                weights=[embeddings], trainable=update_weights))
        else:
            model.add(Embedding(len(dico_code), 100, input_shape=(SIZE_VOCAB,),
                                trainable=update_weights))
        model.add(Flatten())
        model.add(Dense(140, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(nb_neuron_output, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        checkpoint = ModelCheckpoint("best_weights.hdf5", monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train, epochs=NB_EPOCHS, callbacks=callbacks_list)

        score = model.evaluate(x_test, y_test,
                               batch_size=1, verbose=1)

        RESULTS[verb]["loss"] = score[0]
        RESULTS[verb]["accuracy"] = score[1]
        RESULTS[verb]["mfs"] = MFS.get(verb)

    print(RESULTS)
