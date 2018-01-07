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



# TODO : virer ça et corriger les trucs qui provoquent des warnings
import warnings
warnings.filterwarnings("ignore")

class TrainingSession(object):

    dir = "../data_WSD_VS"
    model_file = "../vecs100-linear-frwiki/data"
    classes = {"aborder":4, "affecter":4, "abattre":5}
    pos_ignored = r"^(NONE)$"
    model = Sequential()

    def __init__(self, **kwargs):
        self.mode = kwargs["mode"]
        self.train_p = kwargs["train_percentage"]
        self.nb_epochs = kwargs["nb_epochs"]
        self.size_vocab = kwargs["size_vocab"]
        self.use_embeddings = kwargs["use_embeddings"]
        self.update_weights = kwargs["update_embeddings"]
        self.ctx_size = kwargs["ctx_size"]

        self.train = dict([(key, {}) for key in self.classes])
        self.test = dict(self.train)
        self.gold_data = {}
        self.vocab = None
        self.models =dict(self.train)
        self.features = None


    def get_linear_ctx_dataset(self, dico):
        """
        Creates a coded version of the context and updates the dictionary with its value.

        input :
            - the TrainingSession object
            - TrainingSession.train or TrainingSession.test

        output :
            - the dataset for training the classifier on linear context
        """
        linear_ctx_dataset = {}
        for key in dico:
            x, y = [], []
            for occ in dico[key]:
                linear_ctx = read_data.get_linear_ctx(self, dico[key][occ]["conll"])
                coded_ctx = read_data.linear_ctx_2_cbow(self, linear_ctx)
                x.append(coded_ctx)
                y.append(dico[key][occ]["classe"])
            linear_ctx_dataset[key] = [np.array(x), np.array(y)]

        return linear_ctx_dataset


    def parse_syntax_dataset(self, training):
        """
        Transforms a gold dictionary (train or test) into the dataset to train the NN with syntactic contexts (Surface or Deep).
        Also updates the dictionaries to encode PoS, syntactic relations, voice(diathèse) as well as lemmas.

        input :
            - dictionary with gold data
            - dictionaries containing the correspondencies feature <=> code

        output :
            - dataset for each verb [[voice, syn_level_1, syn_rel_1, lemma_1, pos_1, syn_level_2...], [classe]] (1+4 features by dependency)
        """
        datasets, info_to_encode = dict([(verb, []) for verb in self.classes]), {}

        if training:
            dico = self.train
        else:
            dico = self.test

        for verb in dico:
            for occ in dico[verb].keys():

                syntactic_context = []
                bloc_sentence = dico[verb][occ]["conll"]
                motif = re.compile(r"^(?:(\d+)\t)(?:.+?(?:diat=(.+?)\|[^\t]*)?sense=(?:.+?)\|)", re.MULTILINE)

                try:
                    str_index_verb = re.search(motif, bloc_sentence).group(1)
                except AttributeError:
                    motif_2 = re.compile(r"^(\d+)\t[^\t]+\t(affecter|aborder|abattre|affeterai)", re.MULTILINE)
                    str_index_verb = re.search(motif_2, bloc_sentence).group(1)

                motif_diat = re.compile(r"^%s(?:\t[^\t]+){4}\tdiat=([^\t\|]+)" % str_index_verb, re.MULTILINE)
                diat = re.search(motif_diat, bloc_sentence)

                try:
                    diathese = "diathese="+diat.group(1)
                except AttributeError:
                    diathese = "diathese=False"
                syntactic_context.append(diathese)


                info_to_encode[diathese] = info_to_encode.get(diathese, 0)+1
                motif_dep = re.compile(r"^(\d+)\t[^\t]+\t([^\t]+)\t[^\t]+\t([^\t]+)\t[^\t]+\t(?:\d+\|)*%s(?:\|\d+)*\t([^\t]+)\t[^\t]" % str_index_verb, re.MULTILINE)

                try:
                    dep = re.search(motif_dep, bloc_sentence).groups()
                except AttributeError: # no dependants for this verb
                    syntactic_context.append("dep=False")
                    info_to_encode["dep=False"] = info_to_encode.get("dep=False", 0)+1

                index, lemma_dep, pos_dep, rel_dep = dep
                motif_sub_dep = re.compile(r"^(?:\d+)\t[^\t]+\t([^\t]+)\t[^\t]+\t([^\t]+)\t[^\t]+\t(?:\d+\|)*%s(?:\|\d+)*\t([^\t]+)\t[^\t]" % index, re.MULTILINE)

                for elt in rel_dep.split("|"):

                    niveau = elt[0]
                    if niveau == "I": continue # arg et comp on s'en occupe pas
                    if niveau != "S" and niveau != "D": niveau = "S&D"
                    rel_synt = elt.split(":")[-1] # relation canonique TODO verifier

                    if not re.search(self.pos_ignored, pos_dep):
                        if self.mode == "surface_s" and (niveau == "S&D" or niveau == "S"):
                            syntactic_context.extend([niveau, rel_synt, lemma_dep, pos_dep])

                        elif self.mode == "deep_s" and (niveau == "S&D" or niveau == "D"):
                            syntactic_context.extend([niveau, rel_synt, lemma_dep, pos_dep])

                        for info in [niveau, rel_synt, pos_dep]:
                            info_to_encode[info] = info_to_encode.get(info, 0)+1

                    if "mod" in rel_synt and pos_dep == "P": # going further down the tree for modifiers which are prep
                        lemma_mod, pos_mod, rel_mod = re.search(motif_sub_dep, bloc_sentence).groups()

                        rel_mod = rel_mod.split("|")

                        for elt in rel_mod:
                            niveau = elt[0]
                            if niveau == "I": continue
                            if niveau != "S" and niveau != "D": niveau = "S&D"
                            rel_synt = elt.split(":")[-1] # relation canonique TODO verifier

                            if not re.search(self.pos_ignored, pos_mod):
                                syntactic_context.extend([niveau, rel_synt, lemma_mod, pos_mod])

                                for info in [niveau, rel_synt, pos_mod]:
                                    info_to_encode[info] = info_to_encode.get(info, 0)+1

                datasets[verb].append(syntactic_context)

        if training: # create a feature dictionary
            begin = len(self.vocab)
            self.features = dict(self.vocab) # we don't want our vocab to be updated
            feats_freq = sorted(info_to_encode, key=info_to_encode.get, reverse=True)
            self.features.update(dict([(tok, idx+begin) for idx, tok in enumerate(feats_freq)]))

        return datasets

    def normalise_syntactic_dataset(self, training):
        """
        Replaces the string values of features with integers in a fixed hashing space.

        input:
            - the TrainingSession object
            - a boolean True for training, False for test
        """

        dataset = self.parse_syntax_dataset(training)
        nb_dimensions = len(self.features)
        output = {}
        for verb in dataset:
            x_data = []

            for i in range(len(dataset[verb])):

                length = len(dataset[verb][i]) # nb of features for a given occurence
                rep_vec = np.zeros(nb_dimensions)

                for j in range(length):
                    feature = dataset[verb][i][j]
                    index = self.features.get(feature, 0)
                    rep_vec[index] += 1
                x_data.append(rep_vec)

            output[verb] = np.array(x_data)

        return output

    def code_embeddings(self):
        """
        Extracts a weight matrix from the pre-trained word embeddings making
        up our vocabulary.
        """
        model_embeddings = word2vec.load(self.model_file, kind="txt")
        list_arrays = []

        for word in self.vocab.keys():
            try:
                list_arrays.append(model_embeddings[word])
            except KeyError:
                list_arrays.append(np.zeros(100))
        embeddings = np.array(list_arrays, dtype=object)

        return embeddings

    def run_one_session(self):
        """
        Runs an entire session : creates the appropriate datasets, trains a model for each verb,
        predicts classes for the test data and gives back an evaluation.

        Remark : there is one model per verb and for each verb the final model
                  minimizes the loss.

        input :
            - the TrainingSession object

        output :
            - un dico TrainingSession.models :
            {verb : {model, results : {accuracy, mfs, loss}}}
        """
        read_data.load_gold(self)
        read_data.make_vocab_dico(self)
        read_data.divide_data_in_train_test(self)

        mfs = read_data.most_frequent_sense(self)
        size_vocab = len(self.vocab)
        train_linear = self.get_linear_ctx_dataset(self.train)
        test_linear = self.get_linear_ctx_dataset(self.test)

        if self.mode in ["deep_s", "surface_s"]:
            x_syntactic_train = self.normalise_syntactic_dataset(True)
            x_syntactic_test = self.normalise_syntactic_dataset(False)

        for verb in self.classes:
            self.models[verb] = {"model":None, "results":{}}

            nb_neuron_output = self.classes.get(verb)
            x_linear_train, y_linear_train = train_linear[verb]
            x_linear_test, y_linear_test = test_linear[verb]
            left_branch, model = Sequential(), Sequential()

            if self.use_embeddings: # we use the linear context in both modes
                embeddings = self.code_embeddings()
                left_branch.add(Embedding(size_vocab, 100, input_shape=(size_vocab,),
                                          weights=[embeddings], trainable=self.update_weights))
            else:
                left_branch.add(Embedding(size_vocab, 100, input_shape=(size_vocab,),
                                          trainable=self.update_weights))

            left_branch.add(Flatten())
            left_branch.add(Dense(140, activation='tanh'))
            left_branch.add(Dropout(0.2))

            if self.mode == "linear": # there is only 1 input
                model = left_branch

            elif self.mode in ["deep_s", "surface_s"]: # 2nd input based on syntactic features
                nb_features = len(self.features)
                right_branch = Sequential()
                right_branch.add(Embedding(nb_features, 100, input_shape=(nb_features,)))
                right_branch.add(Flatten())
                right_branch.add(Dense(80, activation="tanh"))
                merged = Merge([left_branch, right_branch], mode="concat")
                model.add(merged)

            model.add(Dense(nb_neuron_output, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['accuracy'])
            callbacks_list = [ModelCheckpoint("best_weights.hdf5", monitor='val_loss',
                                         verbose=0, save_best_only=True, mode='min')]

            if self.mode == "linear":
                model.fit(x_linear_train, y_linear_train, epochs=self.nb_epochs, callbacks=callbacks_list)
                score = model.evaluate(x_linear_test, y_linear_test,
                                       batch_size=1, verbose=1)

            elif self.mode == "deep_s" or self.mode == "surface_s":
                model.fit([x_linear_train, x_syntactic_train[verb]], y_linear_train, epochs=self.nb_epochs, callbacks=callbacks_list)
                score = model.evaluate([x_linear_test, x_syntactic_test[verb]], y_linear_test,
                                       batch_size=1, verbose=1)

            self.models[verb]["model"] = model
            self.models[verb]["results"]["loss"], self.models[verb]["results"]["accuracy"] = score
            self.models[verb]["results"]["mfs"] = mfs.get(verb)

        plot_model(model, to_file=verb+"model.png", show_shapes=True) # only for last verb

if __name__ == "__main__":

    T_1 = TrainingSession(mode="deep_s", train_percentage=0.8, nb_epochs=15, size_vocab=400, use_embeddings=True, update_embeddings=True, ctx_size=2)
    T_1.run_one_session()
    print(T_1.mode, T_1.train_p, T_1.nb_epochs, T_1.size_vocab, T_1.use_embeddings, T_1.update_weights, T_1.ctx_size)
    for verb in T_1.classes:
        print(T_1.models[verb]["results"])
    with open("./results_classification.txt", "w") as outf:
        outf.write("\t".join(["verb", "mfs", "accuracy", "loss"])+"\n")
        for verb in T_1.classes:
            results = T_1.models[verb]["results"]
            outf.write("\t".join([verb, str(results["mfs"]), str(results["accuracy"]), str(results["loss"])])+"\n")
