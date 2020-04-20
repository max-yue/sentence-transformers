"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer
import scipy.spatial
from sentence_transformers import models

pretrained = 'voidful/albert_chinese_tiny'

word_embedding_model = models.ALBERT(pretrained, model_args={'output_hidden_states': True})
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Corpus with example sentences
corpus = [r'如图几何体是由五个相同正方体叠成的,其三视图中的左视图序号是( ).',
          r'$$\left| \overrightarrow{AB}-\overrightarrow{AC} \right|$$的值.',
          r'$$3\times7+18=$$ .',
          ]
corpus_embeddings = embedder.encode(corpus)

# Query sentences:
queries = [r'$$1.8\div 3=$$ .',
           r'试判断$$\angle ACE$$与$$\angle BCD$$的大小关系,并说明理由.',
           ]
query_embeddings = embedder.encode(queries)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 2
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 2 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))



