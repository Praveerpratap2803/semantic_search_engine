class OneHotEncoding:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = []

    def build_vocab(self, words):
        for word in words:
            if word not in self.word_to_index:
                self.word_to_index[word] = len(self.index_to_word)
                self.index_to_word.append(word)

    def encode_word(self, word):
        if word in self.word_to_index:
            one_hot = [0] * len(self.index_to_word)
            one_hot[self.word_to_index[word]] = 1
            return one_hot
        else:
            raise ValueError(f'Word {word} not in vocabulary.')

    def encode_sequence(self, sequence):
        return [self.encode_word(word) for word in sequence]  

    def decode_word(self, one_hot):
        index = one_hot.index(1)
        return self.index_to_word[index]

    def get_embedding(self, word):
        return self.encode_word(word)

    def get_vocab_size(self):
        return len(self.index_to_word)

    def get_vocab(self):
        return self.index_to_word
