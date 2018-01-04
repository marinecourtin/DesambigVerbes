import re
import numpy as np
import read_data


GOLD_DIR = "../data_WSD_VS"
CLASSES = {"aborder":4, "affecter":4, "abattre":5}

SIZE_VOCAB = 400

gold_data = read_data.load_gold(GOLD_DIR, CLASSES)
dico_code, dico_code_reverse = read_data.make_vocab_dico(GOLD_DIR, SIZE_VOCAB)
train, test = read_data.divide_data_in_train_test(gold_data)

# TODO : ADDING SURFACE SYNTAX
# TODO : pleurer sur les coordinations méchantes qui cachent mes beaux sujets
# TODO : code and pad the data ....

def make_dataset_syntax(dico, vocab,training=True):
    """
    Transforms a gold dictionary (train or test) into the dataset to train the NN with syntactic contexts (Surface and Deep).
    Also updates the dictionaries to encode PoS, syntactic relations, voice(diathèse) as well as lemmas.

    input :
        - dictionary with gold data
        - dictionaries containing the correspondencies feature <=> code

    output :
        - dataset for each verb [[voice, syn_level_1, syn_rel_1, lemma_1, pos_1, syn_level_2...], [classe]] (1+4 features by dependency)
    """
    datasets = {}
    info_to_encode = {}

    for verb in dico:
        datasets[verb] = []

        for bloc in dico[verb]:

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
                diathese = diat.group(1)
            except AttributeError:
                diathese = "diathese=False"
            results.append(diathese)
            info_to_encode[diathese]=info_to_encode.get(diathese, 0)+1

            motif_dep = re.compile("([^\t])+\t[^\t]+\t([^\t]+)\t[^\t]+\t([^\t]+)\t[^\t]+\t%s(?:\|(\d+))?\t([^\t]+)\t[^\t]" % str_index_verb, re.MULTILINE)
            try:
                dep = re.search(motif_dep, bloc_sentence).groups()
            except AttributeError: # no dependants for this verb
                results.append(None)
                continue

            index, lemma_dep, pos_dep, second_gov_dep, rel_dep = dep
            motif_sub_dep = re.compile("^([^\t]+)\t[^\t]+\t([^\t]+)\t[^\t]+\t([^\t]+)\t[^\t]+\t%s(?:\|(\d+))?\t([^\t]+)\t[^\t]" % index, re.MULTILINE)

            if pos_dep == "PONCT": continue

            for elt in rel_dep.split("|"):

                niveau = elt[0]
                if niveau == "I": continue # arg et comp on s'en occupe pas
                if niveau != "S" and niveau != "D": niveau = "S&D"
                rel_synt = elt.split(":")[-1] # relation canonique TODO verifier
                # results.append([niveau, rel_synt, lemma_dep, pos_dep])
                results.extend([niveau, rel_synt, lemma_dep, pos_dep])

                for info in [niveau, rel_synt, pos_dep]:
                    info_to_encode[info]=info_to_encode.get(info, 0)+1

                if "mod" in rel_synt and pos_dep == "P": # going further down the tree for modifiers which are prep

                    try:
                        index_mod, lemma_mod, pos_mod, second_gov_mod, rel_mod = re.search(motif_sub_dep, bloc_sentence).groups()
                    except AttributeError: continue # probably due to multiple govenors 1|14|31 TODO fix if I have time

                    rel_mod = rel_mod.split("|")

                    for elt in rel_mod:
                        niveau = elt[0]
                        if niveau == "I": continue
                        if niveau != "S" and niveau != "D": niveau = "S&D"
                        rel_synt = elt.split(":")[-1] # relation canonique TODO verifier
                        # results.append([niveau, rel_synt, lemma_mod, pos_mod])
                        results.extend([niveau, rel_synt, lemma_mod, pos_mod])

                        for info in [niveau, rel_synt, pos_mod]:
                            info_to_encode[info] = info_to_encode.get(info, 0)+1

            datasets[verb].append([results, classe])

    if training: # we update the encoding dictionary
        begin = len(vocab)
        feats_freq = sorted(info_to_encode, key=info_to_encode.get, reverse=True)
        vocab.update(dict([(tok, idx+begin) for idx, tok in enumerate(feats_freq)]))
        return datasets, vocab,
    else:
        return datasets

def normalise_dataset_syntactic_contexte(dataset, vocab):
    """
    Replaces the string values of features with integers in a fixed hashing space.

    input:
        - the dataset w
    """

    nb_dimensions = len(vocab)
    output = {}

    for verb in dataset:
        x_data, y_data = [], []

        for i in range(len(dataset[verb])):

            length = len(dataset[verb][i][0]) # nb of features for a given occurence
            rep_vec = np.zeros(nb_dimensions)

            for j in range(length):
                feature = dataset[verb][i][0][j]
                index = vocab.get(feature, 0)
                rep_vec[index] += 1

            y_data.append(dataset[verb][i][1])
            x_data.append(rep_vec)

        output[verb]=[x_data, y_data]
        # print(len(output[verb]))

    return output


if __name__ == "__main__":
    GOLD_DIR = "../data_WSD_VS"
    SIZE_VOCAB = 400

    vocab_code, vocab_code_reverse = read_data.make_vocab_dico(GOLD_DIR, SIZE_VOCAB)
    (train, vocab), test = make_dataset_syntax(train, vocab_code), make_dataset_syntax(test, vocab_code, False)
    train, test = normalise_dataset_syntactic_contexte(train, vocab), normalise_dataset_syntactic_contexte(test, vocab)
