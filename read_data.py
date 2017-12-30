import numpy as np
import argparse, os, glob, re

def load_gold(directory):
    gold_files = [fichier for fichier in glob.glob(os.path.join(directory, "*.tab"))]
    gold_results = {}
    for fichier in gold_files:
        count=1
        verb = fichier.split("/")[-1][:-4]
        for line in open(os.path.join(directory, fichier)):
            motif = re.compile("^([\w]+#\d#)\t(\d+)\t(.+)$")
            mappingMatch = re.match(motif, line)
            try:
                classe_gold = mappingMatch.group(1)
                # identifiant = mappingMatch.group(2)
                phrase = mappingMatch.group(3)
                rang = count
                count+=1
                # gold_results[verb+"_"+str(count)]={"classe": classe_gold, "id":identifiant, "phrase":phrase}
                gold_results[verb+"_"+str(count)]={"classe": classe_gold, "phrase":phrase}
            except AttributeError:
                # comments
                continue
    return gold_results


if __name__ == "__main__":
    load_gold("../data_WSD_VS")
