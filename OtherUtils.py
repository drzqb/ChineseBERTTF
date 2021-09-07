import collections


def convert2Uni(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    else:
        print(type(text))
        print('####################wrong################')


def load_vocab(vocab_file):  # 获取BERT字表方法
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        while True:
            tmp = reader.readline()
            if not tmp:
                break
            token = convert2Uni(tmp)
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab
