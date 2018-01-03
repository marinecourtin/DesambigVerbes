import re
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

def make_dataset_syntax(dico):
    """
    Transforms a gold dictionary (train or test) into the dataset to train the NN with syntactic contexts (Surface and Deep)

    """
    datasets = {}

    for verb in dico:
        len_max = 0 # we'll use this to pad the dataset so that the input always has the same shape
        datasets[verb] = []

        for bloc in dico[verb]:

            results = []
            bloc_sentence = dico[verb][bloc]["conll"]
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
                diathese = None
            results.append(diathese)

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
                results.append([niveau, rel_synt, lemma_dep, pos_dep])

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
                        results.append([niveau, rel_synt, lemma_mod, pos_mod])

            datasets[verb].append(results)
            if len(results) > len_max:
                len_max = len(results)
        print(verb, len_max)

    return datasets

if __name__ == "__main__":
    train, test = make_dataset_syntax(train), make_dataset_syntax(test)
