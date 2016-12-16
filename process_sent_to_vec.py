import numpy as np
import cPickle
from collections import defaultdict
import sys, re, string
import nltk
from nltk.corpus import stopwords


stopwords = stopwords.words('english')
puncts = string.punctuation + '\r\n``\'\''

def max2(a, b):
    return a if a > b else b

## remove punctuations nums and stopwords    
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [ token for token in tokens if token not in puncts and token.lower() not in stopwords]
    tokens = [ token for token in tokens if not token == "'s"]
    # upper to lower
    #t = []
    #for token in tokens:
    #    if token.lower() in words_list:
    #        token = token.lower()
    #    t.append(token)
    return tokens

def read_data(config):
    """
    Loads data .
    """
    filepath = config['input.file']
    delimiter = '\t' if config['input.delimiter'] == '\\t' else config['input.delimiter']
    indices = [int(val) for val in config['indices'].split(',') ]
    label_coder = {}
    cnt = 0
    max_l = 0
    revs = []
    vocab = defaultdict(float)
    fp = open(filepath, 'r')
    for line in fp.readlines():
        fields = line.strip().split(delimiter)
        s1 = fields[indices[0]]
        s2 = fields[indices[1]]
        t1 = tokenize(s1)
        t2 = tokenize(s2)
        for word in set(t1):
            vocab[word] += 1
        for word in set(t2):
            vocab[word] += 1
        label = fields[indices[2]]
        if label not in label_coder:
            label_coder[label] = cnt
            cnt += 1
        datum = { "label": label_coder[label],
                  "text": (t1, t2),
                  "num_words": (len(t1), len(t2))
                  }
        max_l = max2(max2(len(t1), len(t2)), max_l)
        revs.append(datum)
    fp.close()
    return revs, vocab, max_l
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    '''
    word_vecs = {}
    with open(fname, "r") as f:
        header = f.readline()
        vocab_size,layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs
    '''
    
    word_vecs = {}
    fopen = open('vocab.vector','r')
    for line in fopen:
        line = line.strip().split(' ')
        w1 = []
        w1 = w1.append(line[0])
        w2 = line[1:]
        w3 = ""
        i = 0
        for w in w2:
            if i==0:
                w3 = w3 + w
            else:
                w3 = w3 + ' '
                w3 = w3 + w
            i += 1
        if w1 in vocab:
            word_vecs[w1] = np.fromstring(w3, dtype='float32',sep=' ')
    fopen.close()
    return word_vecs




    

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  


def build_sent(sent, w2v_dict, dim=300):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    words = sent
    vec = np.zeros((dim,))
    for word in words:
        vec = np.add(vec, w2v_dict[word])
    return vec

def build_data(revs, w2v_dict, tran_indices, dim=300):
    """
    Transforms sentences into a 2-d matrix.
    """
    train1, train2, test1, test2 = [], [], [], []
    train_y, test_y = [], []
    for i in xrange(len(revs)):
        rev = revs[i]
        sent1 = build_sent(rev["text"][0], w2v_dict, dim)
        sent2 = build_sent(rev["text"][1], w2v_dict, dim)
        if i in tran_indices:
            train1.append(sent1)
            train2.append(sent2)
            train_y.append(rev['label'])
        else:
            test1.append(sent1)
            test2.append(sent2)
            test_y.append(rev['label'])        

    train_set_x = (np.array(train1,dtype="float32"), np.array(train2,dtype="float32"))
    test_set_x = (np.array(test1,dtype="float32"), np.array(test2,dtype="float32"))
    train_y = np.array(train_y,dtype="int")
    test_y = np.array(test_y,dtype="int")
    
    return [train_set_x, train_y, test_set_x, test_y]  


# model each sentence as a vector
if __name__=="__main__":
    fp = open('config.vec.ini','r')
    config = {}
    for line in fp.readlines():
        line = line.strip()
        if len(line)== 0 or line[0] == '#':
            continue
        fields = line.split('=')
        config[fields[0].strip()] = fields[1].strip()
    fp.close()
    print config
    
    print "loading data...",        
    revs, vocab, max_l = read_data(config)
    print "data loaded!"
    print "number of sentence pairs: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(config['word2vec.file'], vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    print "word_matrix size:(%d,%d)" % (len(w2v), 300)
    
    [train_set_x, train_y, test_set_x, test_y] = build_data(revs, w2v, range(int(config['train_inst'])),
            int(config['word2vec.dim']))

    print len(train_set_x[0]), len(train_set_x[1]), len(train_y)
    print len(test_set_x[0]), len(test_set_x[1]), len(test_y)

    print train_set_x[0][0:1]
    print train_set_x[1][0:1]
    cPickle.dump([train_set_x, train_y, test_set_x, test_y, vocab], open(config['output.file'], "wb"))
    print "dataset created!"
    
