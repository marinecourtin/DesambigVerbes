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


class TrainingSession(object):

    dir = "../data_WSD_VS"
    model_file = "../vecs100-linear-frwiki/data"
    classes = {"aborder":4, "affecter":4, "abattre":5}
    pos_ignored = r"^(PONCT|)$"
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
        self.models = dict(self.train)
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
        Transforms a gold dictionary (train or test) into the dataset to train the NN with
        syntactic contexts (Surface or Surface & Deep). Also creates a dictionary to encode
        PoS, syntactic relations, morphosyntactic features...

        Remark : from the point of view of syntax, we should probably represent a dependency
                by a tuple(syntactic_level, relation, pos), which would be our feature to encode.
                However due to the very limited dataset, I did not encode them as a unit, but
                instead encoded each one separately.

        input :
            - the TrainingSession object
            - a boolean indicating whether we're training or testing

        output :
            - a dataset w/ an entry for each verb : [[features_occ1, lemmas_dependents_occ1], ...]
            - TrainingSession.features which encodes syntactic and morphosyntactic features as weel as lemmas
        """
        datasets, info_to_encode = dict([(verb, []) for verb in self.classes]), {}

        interesting_syn_rel = re.compile(r"obj|suj|ccomp|xcomp|mod|dep|aux\.pass|argc") # can act as a filter
        not_used_feature = re.compile(r"(sense=|dl=)") # features we are NOT using

        if training:
            dico = self.train
        else:
            dico = self.test

        for verb in dico:
            for occ in dico[verb]:

                syntactic_context, dep_visited, lemmas_argument = [], [], []
                bloc_sentence = dico[verb][occ]["conll"]
                motif = re.compile(r"^(\d+)\t(?:[^\t]+\t){4}([^\t]*sense=[^\t]*)\t[^\t]+\t([^\t]+)\t[^\t]+\t[^\t]+$", re.MULTILINE)

                try:
                    str_index_verb, feats, role = re.search(motif, bloc_sentence).groups()
                except AttributeError:
                    motif_2 = re.compile(r"^(\d+)\t[^\t]+\t(?:affecter|aborder|abattre|affeterai)\t(?:[^\t]+\t){2}([^\t]+)\t[^\t]+\t([^\t]+)\t[^\t]+\t[^\t]+$", re.MULTILINE)
                    str_index_verb, feats, role = re.search(motif_2, bloc_sentence).groups()

                if feats:
                    feats = feats.split("|")
                else:
                    feats = ["feats=None"]

                for feat in feats:
                    if re.search(not_used_feature, feat): continue
                    syntactic_context.append(feat)
                    info_to_encode[feat] = info_to_encode.get(feat, 0)+1

                motif_dep = re.compile(r"^(\d+)\t[^\t]+\t([^\t]+)\t[^\t]+\t([^\t]+)\t[^\t]+\t(?:\d+\|)*%s(?:\|\d+)*\t([^\t]+)\t[^\t]" % str_index_verb, re.MULTILINE)

                try:
                    dep = re.search(motif_dep, bloc_sentence).groups()
                except AttributeError: # no dependents for this verb
                    syntactic_context.append("dep=False")
                    info_to_encode["dep=False"] = info_to_encode.get("dep=False", 0)+1

                index, lemma_dep, pos_dep, rel_dep = dep
                motif_dep_of_dep = re.compile(r"^(?:\d+)\t[^\t]+\t([^\t]+)\t[^\t]+\t([^((\t|(DET))]+)\t[^\t]+\t(?:\d+\|)*%s(?:\|\d+)*\t([^\t]+)(?:\t[^\t]){2}$" % index, re.MULTILINE)

                for elt in rel_dep.split("|"):
                    niveau = elt[0]
                    if niveau == "I": continue # doesn't concern us
                    if niveau != "S" and niveau != "D": niveau = "S&D"
                    rel_synt = elt.split(":")[-1]

                    if not re.search(self.pos_ignored, pos_dep):

                        if not re.search(interesting_syn_rel, rel_synt): continue

                        if self.mode == "surface_s" and niveau == "S&D":
                            niveau = "S" # we shouldn't have any info about deep relations in this mode

                        # info = tuple([niveau, rel_synt, pos_dep])
                        info = [niveau, rel_synt, pos_dep]
                        if niveau == "S&D":
                            # syntactic_context.append(info)
                            syntactic_context.extend(info)
                            dep_visited.append([rel_synt, pos_dep])
                            lemmas_argument.append(lemma_dep)

                        elif self.mode == "surface_s" and niveau == "S" or self.mode == "deep_s":
                            if [rel_synt, pos_dep] in dep_visited: continue
                            # syntactic_context.append(info)
                            syntactic_context.extend(info)
                            lemmas_argument.append(lemma_dep)

                        for info in [niveau, rel_synt, pos_dep]:
                            info_to_encode[info] = info_to_encode.get(info, 0)+1

                    if "mod" in rel_synt and pos_dep == "P": # going further down the tree for modifiers which are prep
                        try:
                            lemma_mod, pos_mod, rel_mod = re.search(motif_dep_of_dep, bloc_sentence).groups()
                        except AttributeError:
                            continue

                        rel_mod = rel_mod.split("|")

                        for elt in rel_mod:
                            niveau = elt[0]
                            if niveau == "I": continue
                            if niveau != "S" and niveau != "D": niveau = "S&D"
                            rel_synt_mod = elt.split(":")[-1]

                            if not re.search(self.pos_ignored, pos_mod):

                                if self.mode == "surface_s" and niveau == "S&D":
                                    niveau="S"
                                # info = tuple([niveau, rel_synt_mod, pos_mod])
                                info = [niveau, rel_synt_mod, pos_mod]

                                if niveau == "S&D":
                                    # syntactic_context.append(info)
                                    syntactic_context.extend(info)
                                    dep_visited.append([rel_synt_mod, pos_mod])
                                    lemmas_argument.append(lemma_dep)

                                elif self.mode == "surface_s" and niveau == "S" or self.mode == "deep_s":
                                    if [rel_synt_mod, pos_mod] in dep_visited: continue
                                    # syntactic_context.append(info)
                                    syntactic_context.extand(info)
                                    lemmas_argument.append(lemma_dep)

                                info_to_encode[info] = info_to_encode.get(info, 0)+1
                # print(syntactic_context) # uncomment to observe the elements which will be encoded
                datasets[verb].append([syntactic_context, lemmas_argument])

        if training: # create a feature dictionary
            begin = len(self.vocab)
            self.features = dict(self.vocab) # deep copy as we don't want our vocab to be updated
            feats_freq = sorted(info_to_encode, key=info_to_encode.get, reverse=True)
            self.features.update(dict([(tok, idx+begin) for idx, tok in enumerate(feats_freq)]))
            print(self.features)

        return datasets

    def normalise_syntactic_dataset(self, training):
        """
        Replaces the string values of features with integers in a fixed hashing space.

        input:
            - the TrainingSession object
            - a boolean True for training, False for test

        output:
            - the dataset for training on syntax. Each verb is associated to a list
            of 2 arrays, 1st one for syntactic features (pos, relations...), the 2nd
            for vectorized rep. of the lemmas involved in dependency relations
        """

        dataset = self.parse_syntax_dataset(training)
        nb_dimensions = len(self.features)
        output = {}
        for verb in self.classes:
            x_synt, x_lemma = [], []

            for i in range(len(dataset[verb])):

                syntactic_context, lemmas_argument = dataset[verb][i]
                length = len(syntactic_context)
                rep_vec = np.zeros(nb_dimensions)

                for j in range(length):
                    feature = syntactic_context[j]
                    index = self.features.get(feature, 0)
                    rep_vec[index] += 1
                x_synt.append(rep_vec)

                length_lemma = len(lemmas_argument)
                rep_vec = np.zeros(self.size_vocab)

                for k in range(length_lemma):
                    lemma = lemmas_argument[k]
                    index = self.vocab.get(lemma, 0)
                    rep_vec[index] += 1
                x_lemma.append(rep_vec)

            output[verb] = [np.array(x_synt), np.array(x_lemma)]

        return output

    def code_embeddings(self):
        """
        Extracts a weight matrix from the pre-trained word embeddings making
        up our vocabulary.

        input :
            - the TrainingSession object

        output :
            - an array which we will use to provide weights for our Embedding layer
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
        Runs an entire session : creates the appropriate datasets, trains a model for each verb,
        predicts classes for the test data and gives back an evaluation.

        Remark : there is one model per verb and for each verb the final model
                  maximizes the accuracy.

        input :
            - the TrainingSession object

        output :
            - TrainingSession.models :
            {verb : {model, results : {accuracy, mfs, loss}}}
        """
        read_data.load_gold(self)
        read_data.make_vocab_dico(self)
        read_data.divide_data_in_train_test(self)

        mfs = read_data.most_frequent_sense(self)
        size_vocab = len(self.vocab)
        train_linear = self.get_linear_ctx_dataset(self.train)
        test_linear = self.get_linear_ctx_dataset(self.test)
        embeddings = self.code_embeddings()

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
                left_branch.add(Embedding(size_vocab, 100, input_shape=(size_vocab,),
                                          weights=[embeddings], trainable=self.update_weights,  name="linear_embeddings"))
            else:
                left_branch.add(Embedding(size_vocab, 100, input_shape=(size_vocab,),
                                          trainable=self.update_weights, name="linear_embeddings"))

            left_branch.add(Flatten(name="flat_second_layer_linear"))
            left_branch.add(Dense(140, activation='tanh', name="third_layer_linear"))
            left_branch.add(Dropout(0.2, name="fourth_layer_after_dropout"))

            if self.mode == "linear": # there is only 1 input
                model = left_branch

            elif self.mode in ["deep_s", "surface_s"]: # 2 other inputs based on syntactic features
                nb_features = len(self.features)

                middle_branch = Sequential() # word embeddings of the dependents
                if self.use_embeddings:
                    middle_branch.add(Embedding(size_vocab, 100, input_shape=(size_vocab,),
                                                weights=[embeddings], trainable=self.update_weights,  name="dep_embeddings"))
                else:
                    middle_branch.add(Embedding(size_vocab, 100, input_shape=(size_vocab,),
                                              trainable=self.update_weights, name="dep_embeddings"))
                middle_branch.add(Flatten())
                middle_branch.add(Dense(140, activation='tanh'))
                middle_branch.add(Dropout(0.2))

                right_branch = Sequential() # syntactic features (rel, pos..)
                right_branch.add(Dense(80, input_shape=(nb_features,), activation="tanh", name="third_layer_syntactic"))
                right_branch.add(Dense(25, activation="tanh"))

                merged = Merge([left_branch, middle_branch, right_branch], mode="concat", name="concatenated_layer")
                model.add(merged)

            model.add(Dense(nb_neuron_output, activation='softmax', name="last_layer"))
            model.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['accuracy'])
            callbacks_list = [ModelCheckpoint("best_weights.hdf5", monitor='val_acc',
                                              verbose=0, save_best_only=True, mode='max')]

            if self.mode == "linear":
                model.fit(x_linear_train, y_linear_train, epochs=self.nb_epochs, callbacks=callbacks_list)
                score = model.evaluate(x_linear_test, y_linear_test,
                                       batch_size=1, verbose=1)

            elif self.mode == "deep_s" or self.mode == "surface_s":
                model.fit([x_linear_train, x_syntactic_train[verb][1],  x_syntactic_train[verb][0]], y_linear_train, epochs=self.nb_epochs, callbacks=callbacks_list)
                score = model.evaluate([x_linear_test, x_syntactic_test[verb][1],  x_syntactic_test[verb][0]], y_linear_test,
                                       batch_size=1, verbose=1)

            self.models[verb]["model"] = model
            self.models[verb]["results"]["loss"], self.models[verb]["results"]["accuracy"] = score
            self.models[verb]["results"]["mfs"] = mfs.get(verb)

        plot_model(model, to_file=verb+"model.png", show_shapes=True) # plot only for last verb

if __name__ == "__main__":

    T_1 = TrainingSession(mode="deep_s", train_percentage=0.8, nb_epochs=15, size_vocab=400, use_embeddings=True, update_embeddings=True, ctx_size=2)
    T_1.run_one_session()
    for verb in T_1.classes:
        print(T_1.models[verb]["results"])

    # T_2 = TrainingSession(mode="linear", train_percentage=0.8, nb_epochs=1, size_vocab=400, use_embeddings=True, update_embeddings=True, ctx_size=2)
    # T_3 = TrainingSession(mode="surface_s", train_percentage=0.8, nb_epochs=1, size_vocab=400, use_embeddings=True, update_embeddings=True, ctx_size=2)
    # T_2.run_one_session()
    # T_3.run_one_session()
    #
    #
    # with open("./results_classification.txt", "w") as outf:
    #     outf.write("\t".join(["verb", "mfs", "accuracy", "loss", "mode", "train_percentage", "nb_epochs", "use_embeddings", "update_weights", "ctx_size"])+"\n")
    # for session in [T_1, T_2, T_3]:
    #     print(session.mode, session.train_p, session.nb_epochs, session.use_embeddings, session.update_weights, session.ctx_size)
    #     for verb in session.classes:
    #         print(session.models[verb]["results"])
    #     with open("./results_classification.txt", "a") as outf:
    #         for verb in session.classes:
    #             results = session.models[verb]["results"]
    #             outf.write("\t".join([verb, str(results["mfs"]), str(results["accuracy"]),
    #                        str(results["loss"]), session.mode, str(session.train_p), str(session.nb_epochs),
    #                        str(session.use_embeddings), str(session.update_weights), str(session.ctx_size)])+"\n")
    # print(T_1.mode, T_1.train_p, T_1.nb_epochs, T_1.size_vocab, T_1.use_embeddings, T_1.update_weights, T_1.ctx_size)
