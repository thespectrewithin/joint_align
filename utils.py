import numpy as np
import io
import torch

def read_vocab(path):
    vocab, total_count = {}, 0
    for line in open(path, encoding="utf-8"):
        l = line.strip().split()
        if len(l) == 2:
            vocab[l[0]] = int(l[1])
            total_count += int(l[1])
    return vocab, total_count

def load_embedding(path, vocab_size=-1):
    word_vector = []
    word_dict = {}
    words = []

    for line in open(path):
        if len(line.rstrip().split()) > 2:
            if len(words) < vocab_size or vocab_size == -1:
                word, vec = line.rstrip().split(' ', 1)
                word_dict[word] = len(word_dict)
                words.append(word)
                vec = np.array(vec.split(), dtype='float32')
                word_vector.append(vec)

    return word_dict, np.vstack(word_vector), words

def get_nn_avg_dist(emb, query, knn):
    bs = 1024
    all_distances = []
    emb = emb.transpose(0, 1).contiguous()
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb)
        best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1).cpu())
    all_distances = torch.cat(all_distances)
    return all_distances.numpy()


def load_dictionary(path, word2id1, word2id2):
    pairs = []
    all_pairs = {}
    with io.open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            assert line == line.lower()
            parts = line.rstrip().split()
            word1, word2 = parts
            if word1 in all_pairs:
                all_pairs[word1].append(word2)
            else:
                all_pairs[word1] = [word2]

            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
    included_source_words = set([x for x, _ in pairs])
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]
    return dico, all_pairs, included_source_words

def get_word_translation_accuracy(word2id1, emb1, word2id2, emb2, path, method='csls_knn_10', k=1):

    dico, all_pairs, included_source_words = load_dictionary(path, word2id1, word2id2)
    dico = dico.cuda() if emb1.is_cuda else dico

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[dico[:, 0]][:, None])
        scores.sub_(average_dist2[None, :])

    else:
        raise Exception('Unknown method: "%s"' % method)

    predict_self_count = 0
    for w1 in all_pairs:
        if w1 not in included_source_words and w1 in all_pairs[w1]:
            # Methods automatically predict the same source for itself if it is OOV.
            predict_self_count += 1

    top_matches = scores.topk(10, 1, True)[1]

    top_k_matches = top_matches[:, :k]
    _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1).cpu().numpy()
    matching = {}
    for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
        matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
    precision_at_k = (np.mean(list(matching.values())) * len(included_source_words) + predict_self_count)/len(all_pairs)
    return 100 * precision_at_k

def evaluate_ner(pred, gold):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(pred)):
        pred_entities = get_entity(pred[i])
        gold_entities = get_entity(gold[i])
        temp = 0
        for entity in pred_entities:
            if entity in gold_entities:
                tp += 1
                temp += 1
            else:
                fp += 1
        fn += len(gold_entities) - temp
    precision = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0 
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

def get_entity(label):
    entities = []
    i = 0
    while i < len(label):
        if label[i] != 'O':
            e_type = label[i][2:]
            j = i + 1
            while j < len(label) and label[j] == 'I-' + e_type:
                j += 1
            entities.append((i, j, e_type))
            i = j
        else:
            i += 1
    return entities

def get_features(path):

    saved = torch.load(path)
    features = []
    for i in range(len(saved)):
        features.append(saved[i])
    return features

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, train_data):
        """See base class."""
        return self._create_examples(
            self._read_tsv(train_data), "train")
    
    def get_dev_examples(self, dev_data):
        """See base class."""
        return self._create_examples(
            self._read_tsv(dev_data), "dev")
    
    def get_test_examples(self, test_data):
        """See base class."""
        return self._create_examples(
            self._read_tsv(test_data), "test")
    
    def get_labels(self):
        return ["<PAD>", "O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

def read_align(p, unique=True, reverse=0):
    
    res = []
    cnt = 0
    
    for l in open(p):
        
        ss = l.strip().split()
        align = []
        prev_s = '*'
        prev_t = '*'
        
        for s in ss:
            
            src, trg = s.split('-')
            if reverse:
                src, trg = trg, src
            if unique and (prev_s == src or prev_t == trg):
                continue
            align.append((int(src), int(trg)))
            
            prev_s = src
            prev_t = trg
            cnt += 1
            
        res.append(align)

    return cnt, res

def read_parallel(p, splt=' ||| ', reverse=0):

    res = []
    
    for l in open(p):
        
        ss = l.strip().split(splt)
        if reverse:
            ss = ss[::-1]
        try:
            src = ss[0].split()
            trg = ss[1].split()
        except IndexError:
            print("IndexError：{}".format(l))
            src = []
            trg = []

        res.append([src, trg])

    return res
