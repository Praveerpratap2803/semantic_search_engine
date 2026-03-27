# Implementation of Continuous Bag of Words (CBOW) for word2vec

import numpy as np

class Word2VecCBOW:
    def __init__(self, vocabulary_size, embedding_dim):
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.weights_input_hidden = np.random.rand(vocabulary_size, embedding_dim)  # Input to hidden weights
        self.weights_hidden_output = np.random.rand(embedding_dim, vocabulary_size)  # Hidden to output weights

    def forward(self, context):
        # Forward pass for CBOW
        hidden_layer = np.mean(self.weights_input_hidden[context], axis=0)
        output_layer = np.dot(hidden_layer, self.weights_hidden_output)
        return output_layer

    def train(self, context, target, learning_rate=0.01):
        # Training step for CBOW
        # Forward pass
        output = self.forward(context)
        # Backward pass would be implemented here
        pass

# Example usage
doc = ['the', 'cat', 'sat', 'on', 'the', 'mat']
vocab = {'the': 0, 'cat': 1, 'sat': 2, 'on': 3, 'mat': 4}
cbow = Word2VecCBOW(len(vocab), 100)
cbow.train([1, 2, 3], 0)  # Example context and target