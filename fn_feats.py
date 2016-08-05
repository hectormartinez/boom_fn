import argparse


class WordLabelSentence:
    def __init__(self):
        self.words = []
        self.labels = []


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default="data/orig/framenet_framenames_dev.conll")
    parser.add_argument('--window', default=2, type=int)

    args = parser.parse_args()


    currentsentence = WordLabelSentence()
    sentences = []
    for line in open(args.infile).readlines():
        line = line.strip()
        if line:
            w, l = line.split('\t')
            currentsentence.words.append(w)
            currentsentence.labels.append(l)
        else:
            sentences.append(currentsentence)
            currentsentence = WordLabelSentence()

    if len(currentsentence.words) > 0:
        sentences.append(currentsentence)
    currentsentence = None

    for s in sentences:
        w_pad = ["pad-2","pad-1"]+s.words+["pad+1","pad+2"]
        l_pad = ["_","_"] +s.labels + ["_","_"]

        for i,l in enumerate(l_pad):
            if l != '_':
                print(" ".join(w_pad[i-args.window:i+args.window+1])+"\t"+l)



if __name__ == "__main__":
    main()
