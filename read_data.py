"""This modules reads and prepares the dataset that will be used to train
and test the classifier"""

import os
import glob
import re
import numpy as np


def make_vocab_dico(training_session):
    """
    Creates a hashing dictionary token:code and a reverse one code:token
    Used to vectorize the linear context
    """
    conll_files = [fichier for fichier in glob.glob(os.path.join(training_session.dir, "*.conll"))]
    vocab = {"UKNOWN":99999} # fake token allows us to make predictions for never-encountered context tokens

    for fichier in conll_files:

        for line in open(os.path.join(training_session.dir, fichier)):

            motif = re.compile(r"^(?:\d+)\t([^\t]+)\t[^\t]+\t([^\t]+)\t.+$") # pattern for tokens
            the_match = re.match(motif, line)

            try:
                token = the_match.group(1)
                pos = the_match.group(2)
                if not re.search(training_session.pos_ignored, pos):
                    vocab[token] = vocab.get(token, 0)+1

            except AttributeError: continue

    vocab_freq = sorted(vocab, key=vocab.get, reverse=True)[:training_session.size_vocab]
    training_session.vocab = dict([(tok, idx) for idx, tok in enumerate(vocab_freq)])


def load_gold(training_session):
    """
    Loads the 3 gold data files and updates the gold_data attribute of the training_session object
    with a dico associating occurences of a verb with the appropriate class.

    input :
        - the training_session object

    output :
        - training_session.gold_data : {verb : {occ : {classe, phrase, conll}}}
    """
    gold_files = [fichier for fichier in glob.glob(os.path.join(training_session.dir, "*.tab"))]

    for fichier in gold_files:
        num_data, index_conll = 0, 0
        verb = fichier.split("/")[-1][:-4]
        training_session.gold_data[verb] = {}
        last_identifiant = None

        conll_verb = open("../data_WSD_VS/"+verb+".deep_and_surf.sensetagged.conll").read().split("\n\n")

        for line in open(os.path.join(training_session.dir, fichier)):

            motif = re.compile(r"^([\w]+#\d#)\t(\d+)\t(.+)$")
            the_match = re.match(motif, line)

            try:
                classe_gold = int(the_match.group(1).split("#")[1])
            except AttributeError: continue # comments

            classe_gold_one_hot = np.zeros(training_session.classes[verb])
            classe_gold_one_hot[classe_gold-1] = 1 # one-hot rep of the class because Keras says so
            identifiant = int(the_match.group(2))
            phrase = the_match.group(3)

            if identifiant == last_identifiant: # for some sentences there are several occurences of the verb to disambiguate
                index_conll -= 1 # + directly changed 2 blocs of the conll when the sentences with several occurences didn't follow each other

            training_session.gold_data[verb][num_data] = {"classe": classe_gold_one_hot,
                                                          "phrase":phrase,
                                                          "conll":conll_verb[index_conll]}
            num_data += 1
            index_conll += 1
            last_identifiant = identifiant


def divide_data_in_train_test(training_session):
    """
    Divides randomly (but reproducibly) the gold data into a train set and a test set

    intput:
        - the training_session object

    output:
        - training_session.train and training_session.test : {verb : {occ : {classe, phrase, conll}}}
    """
    for verb in training_session.gold_data:

        nb_data = len(training_session.gold_data[verb])-1
        index_train_data = np.linspace(0, nb_data, int(training_session.train_p*nb_data), dtype=int).tolist()
        training_session.train[verb] = dict([item for item in training_session.gold_data[verb].items() if item[0] in index_train_data])
        training_session.test[verb] = dict([item for item in training_session.gold_data[verb].items() if item[0] not in index_train_data])


def get_linear_ctx(training_session, bloc_sentence):
    """
    Gets a linear context (liste of lemmas) for the verb to disambiguate.

    input :
        - the TrainingSession object
        - bloc_sentence : bloc of text corresponding to a sentence in the conll

    output :
        - list of lemmas present in the context window
    """
    motif = re.compile(r"^(?:(\d+)\t)(?:.+?sense=(?:.+?)\|)", re.MULTILINE)
    infos_token = re.compile(r"(?:[^\t]+\t){2}([^\t]+)\t(?:[^\t]+)\t([^\t]+)\t(?:[^\t]+\t){4}[^\t]+")

    try:
        index_verb_to_deambiguate = int(re.search(motif, bloc_sentence).group(1))-1
    except AttributeError: # not all occurences are id by sense= ...
        # 2 cases : word not id by sense= but has a correct lemma, or has incorrect lemma
        motif_2 = re.compile(r"^(\d+)\t[^\t]+\t(affecter|aborder|abattre|affeterai)", re.MULTILINE)
        index_verb_to_deambiguate = int(re.search(motif_2, bloc_sentence).group(1))-1

    linear_context = []

    for line in bloc_sentence.split("\n"):

        try:
            lemme, upos = re.match(infos_token, line).groups()
            if not re.search(training_session.pos_ignored, upos):
                linear_context.append((lemme, upos))
        except AttributeError: # empty line
            continue

    linear_context_filtered = [] # filtered according to the size of ctx window

    # contexte gauche
    count = 1
    while count <= training_session.ctx_size:
        try:
            if linear_context[index_verb_to_deambiguate-count][1]:
                linear_context_filtered.append(linear_context[index_verb_to_deambiguate-count][0])
                count += 1
        except IndexError: break
    linear_context_filtered = linear_context_filtered[::-1] # get right order back

    # contexte droit
    count = 1
    while count <= training_session.ctx_size:
        try:
            if linear_context[index_verb_to_deambiguate+count][1]:
                linear_context_filtered.append(linear_context[index_verb_to_deambiguate+count][0])
                count += 1
        except IndexError: break

    return linear_context_filtered


def linear_ctx_2_cbow(training_session, linear_context):
    """
    Creates a vectorial representation of the linear context.
    The token is represented by one vector (sum of the one hot vectors of the tokens)

    input :
        - linear_context : list of the lemmas in the context window
        - dico_code : a dictionary associating each token of the vocabulary to its code
                      (UKNOWN words are 0)

    output :
        - a numpy array where the context is coded by an array of size vocab
    """
    rep_ctx = np.zeros(len(training_session.vocab))

    for lemma in linear_context:

        index = training_session.vocab.get(lemma, 0) # word not in the dico is coded at index 0
        rep_ctx[index] += 1

    return rep_ctx


def most_frequent_sense(training_session):
    """
    Computes the baseline results of the MFS for each verb

    input :
        - the TrainingSession object

    output :
        - dico w/ the accuracy of MFS for each verb
    """
    MFS, results = {}, {}

    for verb in training_session.classes:
        MFS[verb] = {}

        for occ in training_session.train[verb]:
            classe_vec = training_session.train[verb][occ]["classe"] # 0100
            sense = np.argmax(classe_vec)+1 # sense n°2
            MFS[verb][sense] = MFS[verb].get(sense, 0)+1

    MFS = dict([(verb, sorted(MFS[verb], key=MFS[verb].get, reverse=True)[0]) for verb in MFS])

    for verb in training_session.classes:
        accuracy = []

        for occ in training_session.test[verb]:
            classe_vec = training_session.test[verb][occ]["classe"]
            sense = np.argmax(classe_vec)+1
            accuracy.append(sense == MFS.get(verb))

        accuracy = sum(accuracy)/len(accuracy)
        results[verb] = accuracy

    return results
