"""
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

import logging

import numpy as np

from sentence_transformers import LoggingHandler
from sentence_transformers import models, SentenceTransformer

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

pretrained = 'voidful/albert_chinese_tiny'

word_embedding_model = models.ALBERT(pretrained, model_args={'output_hidden_states': True})
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Embed a list of sentences
sentences = [r'如图几何体是由五个相同正方体叠成的,其三视图中的左视图序号是( ).',
             r'定义$$a$$※$$b$$、$$b$$※$$c$$、$$c$$※$$d$$、$$d$$※$$b$$分别对应下列图形.那么下列图形中可以表示$$a$$※$$d$$,$$a$$※$$c$$的分别是( ).',
             r'$$\left| \overrightarrow{AB}-\overrightarrow{AC} \right|$$的值.']
sentence_embeddings = model.encode(sentences)

# The result is a list of sentence embeddings as numpy arrays
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
