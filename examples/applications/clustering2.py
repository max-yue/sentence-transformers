"""
This is a simple application for sentence embeddings: clustering

Sentences are mapped to sentence embeddings and then k-mean clustering is applied.
"""
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from sklearn.cluster import KMeans

# embedder = SentenceTransformer('bert-base-nli-mean-tokens')

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

# Perform kmean clustering
num_clusters = 2
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")
