import json
import math
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


class BM25:
    def __init__(self, json_path, k1=2, b=0.75):
        self.k1 = k1
        self.b = b
        self.stopwords = set(stopwords.words("english"))

        self._read_docs(json_path)

        self.corpus = [" ".join(doc["description"]) for doc in self.docs]

        self.stemmer = PorterStemmer()
        self.corpus = [self._tokenize(doc) for doc in self.corpus]
        self.corpus_size = len(self.corpus)
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avg_doc_length = sum(self.doc_lengths) / self.corpus_size
        self.term_freqs = [Counter(doc) for doc in self.corpus]
        self.doc_freqs = self._calculate_doc_freqs()

    def _stem(self, word):
        return self.stemmer.stem(word)

    def _tokenize(self, text):
        tokens = nltk.word_tokenize(text.lower())
        return [
            self._stem(token)
            for token in tokens
            if token.isalpha() and token not in self.stopwords
        ]

    def _calculate_doc_freqs(self):
        doc_freqs = Counter()
        for doc in self.term_freqs:
            doc_freqs.update(doc.keys())
        return doc_freqs

    def _calculate_idf(self, term):
        doc_count = self.doc_freqs.get(term, 0)
        numerator = self.corpus_size - doc_count + 0.5
        denominator = doc_count + 0.5
        return math.log(numerator / denominator + 1)

    def _read_docs(self, path):
        with open(path, "r") as json_file:
            self.docs = json.load(json_file)

    def get_scores(self, query):
        query_tokens = self._tokenize(query)
        scores = [0] * self.corpus_size
        for term in query_tokens:
            idf = self._calculate_idf(term)
            for i in range(self.corpus_size):
                term_frequency = self.term_freqs[i].get(term, 0)
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * (self.doc_lengths[i] / self.avg_doc_length)
                )
                scores[i] += idf * (numerator / denominator)
        return scores

    def get_top_n(self, query, n=5):
        scores = self.get_scores(query)
        scored_docs = list(zip(self.docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_n = [doc for doc, score in scored_docs[:n]]
        return top_n

    def get_page(self, query):
        return self.get_top_n(query)[0]["page_num"]
