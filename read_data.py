import numpy as np
import argparse, os, glob, re, word2vec


def load_gold(directory):
    """
    Load the 3 gold data files and creates a dictionnary associating occurences of a verb with the appropriate class.

    input :
        - path to the directory containing the data files

    output :
        - dictionnary with gold output, 3 verbs as keys. For each verb, you can loop through the ids to get
        to the gold class, sentence, and fragment of conll associated with it.
    """
    gold_files = [fichier for fichier in glob.glob(os.path.join(directory, "*.tab"))]
    gold_results = {}

    for fichier in gold_files:
        num_data, index_conll=0, 0
        verb = fichier.split("/")[-1][:-4]
        gold_results[verb] = {}
        last_identifiant = None

        conll_verb = open("../data_WSD_VS/"+verb+".deep_and_surf.sensetagged.conll").read().split("\n\n")

        for line in open(os.path.join(directory, fichier)):

            motif = re.compile("^([\w]+#\d#)\t(\d+)\t(.+)$")
            mappingMatch = re.match(motif, line)

            try:
                classe_gold = mappingMatch.group(1)
                identifiant = int(mappingMatch.group(2))
                phrase = mappingMatch.group(3)
                if identifiant == last_identifiant: # some sentences have several occurences of the verb to disambiguate
                    index_conll-=1
                gold_results[verb][num_data]={"classe": classe_gold, "phrase":phrase, "conll":conll_verb[index_conll]}
                num_data+=1
                index_conll+=1
                last_identifiant = identifiant

            except AttributeError: continue

    return gold_results


def divide_data_in_train_test(gold_data, percentage_train=0.8):
    """
    Divides randomly (but reproducibly) the gold data into a train set and a test set

    intput:
        - a dictionary containing the gold data
        - the percentage of train data

    output:
        2 dictionaries : 1 with train data and the other with test data
    """
    train, test = {}, {}

    for verb in gold_data:

        nb_data = len(gold_data[verb])-1
        index_train_data = np.linspace(0, nb_data, int(percentage_train*nb_data), dtype=int).tolist()
        train[verb] = dict([item for item in gold_data[verb].items() if item[0] in index_train_data])
        test[verb] = dict([item for item in gold_data[verb].items() if item[0] not in index_train_data])

    return train, test

def get_linear_context(bloc_sentence, pos_ignored, ctx_size=2, ):
    """
    Gets a linear context (liste of lemmas) for the verb to disambiguate.

    input :
        - bloc_sentence : bloc of text corresponding to a sentence in the conll
        - pos_ignored : liste of PoS which aren't taken into account
        - ctx_size : size of the context window

    output :
        - list of lemmas present in the context window
    """
    motif = re.compile("^(?:(\d+)\t)(?:.+?sense=(?:.+?)\|)", re.MULTILINE)
    index_verb_to_deambiguate = int(re.search(motif, bloc_sentence).group(1))-1

    linear_context = []
    for line in bloc_sentence.split("\n"):
        index, forme, lemme, upos, xpos, features, idgov, func, misc1, misc2 = line.split("\t")
        linear_context.append((lemme, upos))
    linear_context_filtered = [] # context is filtered according to the size we passed in the args


    # contexte gauche
    count = 1
    while count <= ctx_size:
        if linear_context[index_verb_to_deambiguate-count][1] not in pos_ignored:
            linear_context_filtered.append(linear_context[index_verb_to_deambiguate-count][0])
            count+=1
    linear_context_filtered=linear_context_filtered[::-1] # dans l'ordre c'est mieux

    # contexte droit
    count = 1
    while count <= ctx_size:
        if linear_context[index_verb_to_deambiguate+count][1] not in pos_ignored:
            linear_context_filtered.append(linear_context[index_verb_to_deambiguate+count][0])
            count+=1

    return linear_context_filtered


def linear_ctx_2_cbow(linear_context, model):
    """
    Creates a vectorial representation of the linear context based on a word2vec model

    input :
        - linear_context : list of the lemmas in the context window
        - model : the word2vec model used the get word embeddings

    output :
        - the cbow representing the linear context (numpy array)
    """

    rep_vec = None

    for lemma in linear_context:

        if isinstance(rep_vec, np.ndarray):
            rep_vec += model[lemma] # le vecteur a déjà été initialisé
        else:
            rep_vec=model[lemma] # on initialise et on type

    return rep_vec




if __name__ == "__main__":
    gold_affecter = load_gold("../data_WSD_VS")
    divide_data_in_train_test(gold_affecter, 0.8)
    bloc_sentence = open("../data_WSD_VS/abattre.deep_and_surf.sensetagged.conll").read().split("\n\n")[3]
    pos_ignored = ['PUNCT']
    linear_context = get_linear_context(bloc_sentence, pos_ignored)
    model = word2vec.load("../vecs100-linear-frwiki/data", kind="txt")
    linear_ctx_2_cbow(linear_context, model)
