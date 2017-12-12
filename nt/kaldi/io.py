def word2id(words_txt):
    with open(words_txt) as fid:
        return {
            line.strip().split(' ')[0]: int(line.split(' ')[1])
            for line in fid if len(line.split(' ')) == 2
        }


def id2word(words_txt):
    with open(words_txt) as fid:
        return {
            int(line.strip().split(' ')[1]): line.split(' ')[0]
            for line in fid if len(line.split(' ')) == 2
        }


def read_keyed_text_file(text_file):
    with open(text_file) as fid:
        return {
            line.strip().split()[0]: line.strip().split()[1:] for line in fid
        }
