# import torch
# import torchtext.vocab as vocab
import pickle

with open('word_embedding.pkl', 'rb') as f:
    word_embedding = pickle.load(f)
    

# word_embedding = vocab.Vectors(name = "vi_word2vec.txt",
#                                unk_init = torch.Tensor.normal_)

# word_embedding.vectors.shape

# def get_vector(embeddings, word):
#     """ Get embedding vector of the word
#     @param embeddings (torchtext.vocab.vectors.Vectors)
#     @param word (str)
#     @return vector (torch.Tensor)
#     """
#     assert word in embeddings.stoi, f'*{word}* is not in the vocab!'
#     return embeddings.vectors[embeddings.stoi[word]]

# def closest_words(embeddings, vector, n=10):
#     """ Return n words closest in meaning to the word
#     @param embeddings (torchtext.vocab.vectors.Vectors)
#     @param vector (torch.Tensor)
#     @param n (int)
#     @return words (list(tuple(str, float)))
#     """
#     distances = [(word, torch.dist(vector, get_vector(embeddings, word)).item())
#                  for word in embeddings.itos]

#     return sorted(distances, key = lambda w: w[1])[:n]