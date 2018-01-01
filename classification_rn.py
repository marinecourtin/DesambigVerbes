import read_data, word2vec
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Activation, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

def make_dataset(gold_data, dico_code, train_percentage=0.8, mode="linear", pos_ignored=['PUNCT'], ctx_size=2):
    """
    Creates the datasets for training the classifier on linear context.
    """
    train_data, test_data = read_data.divide_data_in_train_test(gold_data, train_percentage)
    if mode == "linear":
        train, test = {}, {}

        for verb in train_data:
            x_train, y_train = [], []
            for occ in train_data[verb]:
                linear_context = read_data.get_linear_context(train_data[verb][occ]["conll"], pos_ignored)
                coded_context = read_data.linear_ctx_2_cbow(linear_context, dico_code, ctx_size)
                x_train.append(coded_context)
                y_train.append(train_data[verb][occ]["classe"])
            train[verb]=[np.array(x_train), np.array(y_train)]

        for verb in test_data:
            x_test, y_test = [], []
            for occ in test_data[verb]:
                linear_context = read_data.get_linear_context(test_data[verb][occ]["conll"], pos_ignored)
                coded_context = read_data.linear_ctx_2_cbow(linear_context, dico_code, ctx_size)
                x_test.append(coded_context)
                y_test.append(test_data[verb][occ]["classe"])
            test[verb]=[np.array(x_test), np.array(y_test)]

    return train, test

def code_embeddings(MODEL_FILE, vocab):
    model = word2vec.load(MODEL_FILE, kind="txt")
    list_arrays = []
    for word in vocab:
        try:
            list_arrays.append(model[word])
        except:
            list_arrays.append(np.zeros(100))
    embeddings = np.array(list_arrays, dtype=object)
    return embeddings

if __name__ =="__main__":

    GOLD_DIR = "../data_WSD_VS"
    train_percentage = 0.8
    CONLL_FILE = "../data_WSD_VS/abattre.deep_and_surf.sensetagged.conll"
    pos_ignored = ['PUNCT']
    MODEL_FILE = "../vecs100-linear-frwiki/data"
    SIZE_VOCAB=400

    gold_data = read_data.load_gold(GOLD_DIR)
    dico_code, dico_code_reverse = read_data.make_vocab_dico(GOLD_DIR, SIZE_VOCAB)
    train, test = make_dataset(gold_data, dico_code)
    embeddings = code_embeddings(MODEL_FILE, dico_code)

    x_train, y_train = train["aborder"]
    x_test, y_test = test["aborder"]

    # read_data.linear_ctx_2_cbow(linear_context, model)
    # TEST avec CBOW
    model = Sequential()
    model.add(Embedding(len(dico_code), 100, input_shape=(SIZE_VOCAB,), weights=[embeddings]))
    model.add(Flatten())
    model.add(Dense(140, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint("best_weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='model.png')
    model.fit(x_train, y_train, epochs=10, callbacks=callbacks_list)

    score = model.evaluate(x_test, y_test,
                           batch_size=1, verbose=1)
    print("\n")
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
