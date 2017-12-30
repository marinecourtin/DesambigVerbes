import numpy as np
import argparse, os, glob, re

def load_gold(directory):
    """
    Load the 3 gold data files and creates a dictionnary associating occurences of a verb with the appropriate class.

    input :
        - path to the directory containing the data files
    output :
        - dictionnary with gold output, 3 verbs as keys. For each verb, you can loop through the ids to get
        to the gold class and sentence associated with it.
    """
    gold_files = [fichier for fichier in glob.glob(os.path.join(directory, "*.tab"))]
    gold_results = {}
    for fichier in gold_files:
        rang=0
        verb = fichier.split("/")[-1][:-4]
        gold_results[verb] = {}
        for line in open(os.path.join(directory, fichier)):
            motif = re.compile("^([\w]+#\d#)\t(\d+)\t(.+)$")
            mappingMatch = re.match(motif, line)
            try:
                classe_gold = mappingMatch.group(1)
                # identifiant = mappingMatch.group(2)
                phrase = mappingMatch.group(3)
                # gold_results[verb+"_"+str(count)]={"classe": classe_gold, "id":identifiant, "phrase":phrase}
                gold_results[verb][rang]={"classe": classe_gold, "phrase":phrase}
                rang+=1
            except AttributeError:
                # comments
                continue
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


if __name__ == "__main__":
    gold_affecter = load_gold("../data_WSD_VS")
    divide_data_in_train_test(gold_affecter, 0.8)
