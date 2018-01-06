"""This module builds the neural networks aimed at desambiguating
occurences of the verbs in context."""

import read_data
import word2vec
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout, Merge
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
# from keras.layers.merge

import parse_syntax


# TODO : virer ça et corriger les trucs qui provoquent des warnings
import warnings
warnings.filterwarnings("ignore")

class LackRessource(Exception):
    pass

class TrainingSession(object):

    dir = "../data_WSD_VS"
    model_file = "../vecs100-linear-frwiki/data"
    classes = {"aborder":4, "affecter":4, "abattre":5}
    pos_ignored = ['PUNCT']
    model = Sequential()

    def __init__(self, mode, train_percentage, nb_epochs, size_vocab, use_embeddings, update_weights, ctx_size):
        self.mode = mode
        self.train_p = train_percentage
        self.nb_epochs = nb_epochs
        self.size_vocab = size_vocab
        self.use_embeddings = use_embeddings
        self.update_weights = update_weights
        self.gold_data = {}
        self.vocab = {}
        self.results = dict([(key, {}) for key in self.classes])
        self.ctx_size = ctx_size

    def add_gold(self):
        """
        Updates the gold_data attribute.
        """
        self.gold_data = read_data.load_gold(self.dir, self.classes)

    def find_mfs(self):
        """
        Computes the most_frequent_sense.
        """
        if not self.gold_data:
            self.add_gold()
        train_mfs, test_mfs = read_data.divide_data_in_train_test(self.gold_data)
        MFS = read_data.most_frequent_sense(train_mfs, test_mfs)
        return MFS

    def add_vocab(self):
        """
        Updates the vocab attribute of the object.
        """
        vocab, vocab_reverse = read_data.make_vocab_dico(self.dir, self.size_vocab)
        self.vocab = vocab

    def code_dico_linear_context(self, dico):
        """
        Creates a coded version of the context and updates the dictionary with its value.
        """
        result = {}
        for key in dico:
            x, y = [], []
            for occ in dico[key]:
                linear_ctx = read_data.get_linear_ctx(dico[key][occ]["conll"],
                                                      self.pos_ignored)
                coded_ctx = read_data.linear_ctx_2_cbow(linear_ctx,
                                                        self.vocab, self.ctx_size)
                x.append(coded_ctx)
                y.append(dico[key][occ]["classe"])
            result[key] = [np.array(x), np.array(y)]
        return result

    def get_linear_dataset(self):
        """
        Creates the datasets for training the classifier on linear context.
        """
        if not self.gold_data:
            raise LackRessource("The gold data hasn't been updated yet. You must call add_gold")
        if not self.vocab:
            raise LackRessource("The gold data hasn't been updated yet. You must call add_vocab")

        train_data, test_data = read_data.divide_data_in_train_test(self.gold_data, self.train_p)
        train, test = {}, {}

        train = self.code_dico_linear_context(train_data)
        test = self.code_dico_linear_context(test_data)

        return train, test

    def get_syntactic_dataset(self):
        """
        Creates the dataset for training the network on syntactic data.
        """
        divided_data = read_data.divide_data_in_train_test(self.gold_data)
        train_data, features_dico = parse_syntax.make_dataset_syntax(divided_data[0], self.vocab)
        self.nb_features = len(features_dico)
        test_data = parse_syntax.make_dataset_syntax(divided_data[1], self.vocab, False) # TODO : verifier que c'est bien self.vocab
        train_normalised_dataset = parse_syntax.normalise_dataset_syntactic_contexte(train_data, features_dico)
        test_normalised_dataset = parse_syntax.normalise_dataset_syntactic_contexte(test_data, features_dico)
        return train_normalised_dataset, test_normalised_dataset

    def code_embeddings(self):
        """
        Extracts a weight matrix from the pre-trained word embeddings making
        up our vocabulary.
        """
        model_embeddings = word2vec.load(self.model_file, kind="txt")
        list_arrays = []

        for word in self.vocab:
            try:
                list_arrays.append(model_embeddings[word])
            except KeyError:
                list_arrays.append(np.zeros(100))
        embeddings = np.array(list_arrays, dtype=object)

        return embeddings

    def run_one_session(self):
        """
        Given a set of parameters, runs a session with training, prediction and evaluation.
        """
        self.add_gold()
        self.add_vocab()

        gold_data = self.gold_data
        MFS = self.find_mfs()
        size_vocab = len(self.vocab)
        train, test = self.get_linear_dataset()
        embeddings = self.code_embeddings()

        for verb in self.classes:

            nb_neuron_output = self.classes.get(verb)
            x_linear_train, y_linear_train = train[verb]
            x_linear_test, y_linear_test = test[verb]


            left_branch, model = Sequential(), Sequential()

            if self.use_embeddings: # we use the linear context in both modes
                left_branch.add(Embedding(size_vocab, 100, input_shape=(size_vocab,),
                                          weights=[embeddings], trainable=self.update_weights))
            else:
                left_branch.add(Embedding(size_vocab, 100, input_shape=(size_vocab,),
                                                trainable=self.update_weights))
            left_branch.add(Flatten())
            left_branch.add(Dense(140, activation='tanh'))
            left_branch.add(Dropout(0.2))

            if self.mode == "linear": # if the mode is linear, there is no other input
                model = left_branch

            elif self.mode == "syntactic":
                syntactic_dataset = self.get_syntactic_dataset()
                x_syntactic_train, y_syntactic_train = syntactic_dataset[0][verb]
                x_syntactic_test, y_syntactic_test = syntactic_dataset[1][verb]

                # TODO : virer ça quand je saurais d'où vient le problème
                x_linear_train = x_linear_train[:len(x_syntactic_train)]
                y_linear_train = y_linear_train[:len(y_syntactic_train)]
                x_linear_test = x_linear_test[:len(x_syntactic_test)]
                y_linear_test = y_linear_test[:len(y_syntactic_test)]


                # print(len(x_linear_train), len(y_linear_train), len(x_linear_test), len(y_linear_test))
                # print(len(x_syntactic_train), len(y_syntactic_train), len(x_syntactic_test), len(y_syntactic_test))
                right_branch = Sequential()

                # adding 2nd input based on syntactic features
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # print(self.nb_features)
                right_branch.add(Embedding(self.nb_features, 100, input_shape=(self.nb_features,)))
                right_branch.add(Flatten())
                right_branch.add(Dense(80, activation="tanh"))
                print(left_branch.output_shape, right_branch.output_shape)
                merged = Merge([left_branch, right_branch], mode="concat")
                model.add(merged)

            model.add(Dense(nb_neuron_output, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['accuracy'])
            checkpoint = ModelCheckpoint("best_weights.hdf5", monitor='val_acc',
                                         verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]
            model.summary()

            if self.mode == "linear":
                model.fit(x_linear_train, y_linear_train, epochs=self.nb_epochs, callbacks=callbacks_list)
                score = model.evaluate(x_linear_test, y_linear_test,
                                       batch_size=1, verbose=1)
            elif self.mode == "syntactic":
                model.fit([x_linear_train, x_syntactic_train], y_linear_train, epochs=self.nb_epochs, callbacks=callbacks_list)
                score = model.evaluate([x_linear_test, x_syntactic_test], y_linear_test,
                                       batch_size=1, verbose=1)

            self.model = model
            self.results[verb]["loss"], self.results[verb]["accuracy"] = score
            self.results[verb]["mfs"] = MFS.get(verb)

            plot_model(self.model, to_file='model.png')

# TODO : faire en sorte qu'il y ait un partage des arguments de la session avec read_data et parse_syntax pour chaque fonction qui le requière (use attrbutes ? import class ?)

if __name__ == "__main__":

    T_1 = TrainingSession("syntactic", 0.8, 15, 400, True, True, 2)
    T_1.run_one_session()
