import numpy as np
import argparse, os, glob, re


def make_vocab_dico(directory, size_vocab=400, pos_ignored=['PONCT']):
    """
    Creates a hashing dictionary token:code and a reverse one code:token
    Used to vectorize the linear context
    """
    conll_files = [fichier for fichier in glob.glob(os.path.join(directory, "*.conll"))]
    vocab = {"UKNOWN":99999} # adding a fake token will allow us to make predictions for never-encountered context tokens

    for fichier in conll_files:

        for line in open(os.path.join(directory, fichier)):

            motif = re.compile("^(?:\d+)\t([^\t]+)\t[^\t]+\t([^\t]+)\t.+$") # pattern for tokens
            mappingMatch = re.match(motif, line)

            try:
                token = mappingMatch.group(1)
                pos = mappingMatch.group(2)
                if pos in pos_ignored:continue
                vocab[token] = vocab.get(token, 0)+1

            except AttributeError:continue

    vocab_freq = sorted(vocab, key=vocab.get, reverse=True)[:size_vocab]
    dico_code = dict([(tok, idx) for idx, tok in enumerate(vocab_freq)])
    dico_code_reverse = dict([(idx, tok) for idx, tok in enumerate(vocab_freq)])

    return dico_code, dico_code_reverse



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
    gold_results, vocab = {}, {}
    nb_class = {"aborder":4, "affecter":4, "abattre":5}

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
                classe_gold = int(mappingMatch.group(1).split("#")[1])
                classe_gold_one_hot = np.zeros(nb_class[verb])
                classe_gold_one_hot[classe_gold-1]=1 # switching the class for its one-hot representation because Keras says so
                identifiant = int(mappingMatch.group(2))
                phrase = mappingMatch.group(3)

                if identifiant == last_identifiant: # some sentences have several occurences of the verb to disambiguate
                    index_conll-=1 # I also repeated 2 blocs of the conll when the sentences with several occurences didn't follow each other

                gold_results[verb][num_data]={"classe": classe_gold_one_hot, "phrase":phrase, "conll":conll_verb[index_conll]}
                num_data+=1
                index_conll+=1
                last_identifiant = identifiant

            except AttributeError: continue #comments

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



def get_linear_context(bloc_sentence, pos_ignored, ctx_size=2 ):
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
    try:
        index_verb_to_deambiguate = int(re.search(motif, bloc_sentence).group(1))-1
    except AttributeError: # le mot n'a pas été repéré par sense= ...
        # 2 cases : word not id by sense= but has a correct lemma, or has incorrect lemma
        motif_2 = re.compile("^(\d+)\t[^\t]+\t(affecter|aborder|abattre|affeterai)", re.MULTILINE)
        index_verb_to_deambiguate = int(re.search(motif_2, bloc_sentence).group(1))-1

    linear_context = []
    for line in bloc_sentence.split("\n"):
        try:
            index, forme, lemme, upos, xpos, features, idgov, func, misc1, misc2 = line.split("\t")
            linear_context.append((lemme, upos))
        except ValueError:
            print(bloc_sentence)
    linear_context_filtered = [] # context is filtered according to the size we passed in the args


    # contexte gauche
    count = 1
    while count <= ctx_size:
        try:
            if linear_context[index_verb_to_deambiguate-count][1] not in pos_ignored:
                linear_context_filtered.append(linear_context[index_verb_to_deambiguate-count][0])
                count+=1
        except IndexError: break
    linear_context_filtered=linear_context_filtered[::-1] # dans l'ordre c'est mieux

    # contexte droit
    count = 1
    while count <= ctx_size:
        try:
            if linear_context[index_verb_to_deambiguate+count][1] not in pos_ignored:
                linear_context_filtered.append(linear_context[index_verb_to_deambiguate+count][0])
                count+=1
        except IndexError: break

    return linear_context_filtered



def linear_ctx_2_one_hot_array(linear_context, dico_code, ctx_size=2):
    """
    Creates a vectorial representation of the linear context. Each token is represented by a one-hot vector.

    input :
        - linear_context : list of the lemmas in the context window
        - dico_code : a dictionary associating each token of the vocabulary to its code (UKNOWN words are 0)

    output :
        - a numpy array where each context word (2*ctx_size) is coded by a boolean array of size vocab
    """
    list_arrays = []

    for lemma in linear_context:

        rep_lemma = np.zeros(len(dico_code))
        index = dico_code.get(lemma, 0) #if the word is not in the dictionary it is coded at index 0
        rep_lemma[index]=1
        list_arrays.append(rep_lemma)

    rep_vec = np.array(list_arrays, dtype=object)

    return rep_vec

def linear_ctx_2_cbow(linear_context, dico_code, ctx_size=2):
    """
    Creates a vectorial representation of the linear context. The token is represented by one vector (sum of the one hot vectors of the tokens)

    input :
        - linear_context : list of the lemmas in the context window
        - dico_code : a dictionary associating each token of the vocabulary to its code (UKNOWN words are 0)

    output :
        - a numpy array where the context is coded by an array of size vocab
    """
    #TODO : see if I want to have this be boolean or freq
    rep_contexte = np.zeros(len(dico_code))

    for lemma in linear_context:

        index = dico_code.get(lemma, 0) #if the word is not in the dictionary it is coded at index 0
        rep_contexte[index]+=1

    return rep_contexte



if __name__ == "__main__":
    DIR = "../data_WSD_VS"
    pos_ignored = ['PONCT']

    gold_affecter = load_gold(DIR)
    # divide_data_in_train_test(gold_affecter, 0.8)
    bloc_sentence = open("../data_WSD_VS/abattre.deep_and_surf.sensetagged.conll").read().split("\n\n")[3]
    # pos_ignored = ['PUNCT']
    linear_context = get_linear_context(bloc_sentence, pos_ignored)
    # model = word2vec.load("../vecs100-linear-frwiki/data", kind="txt")

    dico_code, dico_code_reverse = make_vocab_dico(DIR)
    # print(dico_code_reverse[0])
    # linear_ctx_2_one_hot_array(linear_context, dico_code)
    linear_ctx_2_cbow(linear_context, dico_code)
