"""This module builds the neural networks aimed at desambiguating
occurences of the verbs in context."""

import re
import read_data
import word2vec
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout, Merge
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

import parse_syntax


# TODO : virer ça et corriger les trucs qui provoquent des warnings
import warnings
warnings.filterwarnings("ignore")

class LackRessource(Exception):
    pass

class TrainingSession(object):

    dir = "../data_WSD_VS" # TODO : remettre vrai dossier
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

    def make_dataset_syntax(self, dico, training):
        """
        Transforms a gold dictionary (train or test) into the dataset to train the NN with syntactic contexts (Surface and Deep).
        Also updates the dictionaries to encode PoS, syntactic relations, voice(diathèse) as well as lemmas.

        input :
            - dictionary with gold data
            - dictionaries containing the correspondencies feature <=> code

        output :
            - dataset for each verb [[voice, syn_level_1, syn_rel_1, lemma_1, pos_1, syn_level_2...], [classe]] (1+4 features by dependency)
        """
        datasets, info_to_encode = {}, {}
        vocab = self.vocab
        for verb in dico:
            count = 0
            datasets[verb] = []

            for bloc in dico[verb]: # 177

                results = []
                bloc_sentence = dico[verb][bloc]["conll"]
                classe = dico[verb][bloc]["classe"]
                motif = re.compile("^(?:(\d+)\t)(?:.+?(?:diat=(.+?)\|[^\t]*)?sense=(?:.+?)\|)", re.MULTILINE)

                try:
                    str_index_verb = re.search(motif, bloc_sentence).group(1)
                except AttributeError:
                    motif_2 = re.compile("^(\d+)\t[^\t]+\t(affecter|aborder|abattre|affeterai)", re.MULTILINE)
                    str_index_verb = re.search(motif_2, bloc_sentence).group(1)

                motif_diat = re.compile("^%s(?:\t[^\t]+){4}\tdiat=([^\t\|]+)" % str_index_verb, re.MULTILINE)
                diat = re.search(motif_diat, bloc_sentence)

                try:
                    diathese = "diathese="+diat.group(1)
                except AttributeError:
                    diathese = "diathese=False"
                results.append(diathese)


                info_to_encode[diathese]=info_to_encode.get(diathese, 0)+1
                motif_dep = re.compile("^(\d+)\t[^\t]+\t([^\t]+)\t[^\t]+\t([^\t]+)\t[^\t]+\t(?:\d+\|)*%s(?:\|\d+)*\t([^\t]+)\t[^\t]" % str_index_verb, re.MULTILINE)

                try:
                    dep = re.search(motif_dep, bloc_sentence).groups()
                except AttributeError: # no dependants for this verb
                    results.append("dep=False")
                    info_to_encode["dep=False"] = info_to_encode.get("dep=False", 0)+1

                index, lemma_dep, pos_dep, rel_dep = dep
                motif_sub_dep = re.compile("^(\d+)\t[^\t]+\t([^\t]+)\t[^\t]+\t([^\t]+)\t[^\t]+\t(?:\d+\|)*%s(?:\|\d+)*\t([^\t]+)\t[^\t]" % index, re.MULTILINE)

                # TODO : si je veux filtrer sur les pos
                for elt in rel_dep.split("|"):

                    niveau = elt[0]
                    if niveau == "I": continue # arg et comp on s'en occupe pas
                    if niveau != "S" and niveau != "D": niveau = "S&D"
                    rel_synt = elt.split(":")[-1] # relation canonique TODO verifier

                    if self.mode == "surface_s" and (niveau == "S&D" or niveau == "S"):
                        results.extend([niveau, rel_synt, lemma_dep, pos_dep])

                    elif self.mode == "deep_s" and (niveau == "S&D" or niveau == "D"):
                        results.extend([niveau, rel_synt, lemma_dep, pos_dep])

                    for info in [niveau, rel_synt, pos_dep]:
                        info_to_encode[info]=info_to_encode.get(info, 0)+1

                    if "mod" in rel_synt and pos_dep == "P": # going further down the tree for modifiers which are prep
                        index_mod, lemma_mod, pos_mod, rel_mod = re.search(motif_sub_dep, bloc_sentence).groups()

                        rel_mod = rel_mod.split("|")

                        for elt in rel_mod:
                            niveau = elt[0]
                            if niveau == "I": continue
                            if niveau != "S" and niveau != "D": niveau = "S&D"
                            rel_synt = elt.split(":")[-1] # relation canonique TODO verifier

                            results.extend([niveau, rel_synt, lemma_mod, pos_mod])

                            for info in [niveau, rel_synt, pos_mod]:
                                info_to_encode[info] = info_to_encode.get(info, 0)+1

                datasets[verb].append([results, classe])

        if training: # create a feature dictionary
            begin = len(self.vocab)
            self.features = dict(self.vocab) # we don't want our vocab to be updated
            feats_freq = sorted(info_to_encode, key=info_to_encode.get, reverse=True)
            self.features.update(dict([(tok, idx+begin) for idx, tok in enumerate(feats_freq)]))

        return datasets

    def normalise_dataset_syntactic_contexte(self, dataset):
        """
        Replaces the string values of features with integers in a fixed hashing space.

        input:
            - the dataset w
        """

        nb_dimensions = len(self.features)
        output = {}
        for verb in dataset:
            x_data, y_data = [], []

            for i in range(len(dataset[verb])):

                length = len(dataset[verb][i][0]) # nb of features for a given occurence
                rep_vec = np.zeros(nb_dimensions)

                for j in range(length):
                    feature = dataset[verb][i][0][j]
                    index = self.features.get(feature, 0)
                    rep_vec[index] += 1

                y_data.append(dataset[verb][i][1])
                x_data.append(rep_vec)

            output[verb]=[np.array(x_data), np.array(y_data)]

        return output

    def get_syntactic_dataset(self):
        """
        Creates the dataset for training the network on syntactic data.
        """
        train, test= read_data.divide_data_in_train_test(self.gold_data)

        train_data = self.make_dataset_syntax(train, True)
        test_data = self.make_dataset_syntax(test, False) # TODO : verifier que c'est bien self.vocab

        train_normalised_dataset = self.normalise_dataset_syntactic_contexte(train_data)
        test_normalised_dataset = self.normalise_dataset_syntactic_contexte(test_data)

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


            self.left_branch, self.model = Sequential(), Sequential()

            if self.use_embeddings: # we use the linear context in both modes
                self.left_branch.add(Embedding(size_vocab, 100, input_shape=(size_vocab,),
                                          weights=[embeddings], trainable=self.update_weights))
            else:
                self.left_branch.add(Embedding(size_vocab, 100, input_shape=(size_vocab,),
                                                trainable=self.update_weights))
            self.left_branch.add(Flatten())
            self.left_branch.add(Dense(140, activation='tanh'))
            dropout=0.2
            self.left_branch.add(Dropout(dropout))

            if self.mode == "linear": # if the mode is linear, there is no other input
                self.model = self.left_branch

            elif self.mode == "deep_s" or self.mode == "surface_s":
                syntactic_dataset = self.get_syntactic_dataset()
                nb_features = len(self.features)
                x_syntactic_train, y_syntactic_train = syntactic_dataset[0][verb]
                x_syntactic_test, y_syntactic_test = syntactic_dataset[1][verb]

                self.right_branch = Sequential()

                # adding 2nd input based on syntactic features
                self.right_branch.add(Embedding(nb_features, 100, input_shape=(nb_features,)))
                self.right_branch.add(Flatten())
                self.right_branch.add(Dense(80, activation="tanh"))
                self.merged = Merge([self.left_branch, self.right_branch], mode="concat")
                self.model.add(self.merged)

            self.model.add(Dense(nb_neuron_output, activation='softmax'))
            self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['accuracy'])
            checkpoint = ModelCheckpoint("best_weights.hdf5", monitor='val_loss',
                                         verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            self.model.summary()

            if self.mode == "linear":
                self.model.fit(x_linear_train, y_linear_train, epochs=self.nb_epochs, callbacks=callbacks_list)
                score = self.model.evaluate(x_linear_test, y_linear_test,
                                       batch_size=1, verbose=1)
            elif self.mode == "deep_s" or self.mode == "surface_s":
                self.model.fit([x_linear_train, x_syntactic_train], y_linear_train, epochs=self.nb_epochs, callbacks=callbacks_list)
                score = self.model.evaluate([x_linear_test, x_syntactic_test], y_linear_test,
                                       batch_size=1, verbose=1)

            self.results[verb]["loss"], self.results[verb]["accuracy"] = score
            self.results[verb]["mfs"] = MFS.get(verb)

if __name__ == "__main__":

    T_1 = TrainingSession("deep_s", 0.8, 15, 400, True, True, 2)
    T_1.run_one_session()
    plot_model(T_1.model, to_file='model.png', show_shapes=True)
    print(T_1.mode, T_1.train_p, T_1.nb_epochs, T_1.size_vocab, T_1.use_embeddings, T_1.update_weights, T_1.ctx_size, "dropout=0.2")
    print(T_1.results)
