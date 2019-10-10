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

