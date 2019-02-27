import pickle
from nltk.stem import SnowballStemmer
import string
import numpy as np


class SentimentClassifier(object):
    def __init__(self):
        with open('PAclassifier.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def get_prediction(self, text):
        text = stemmization(text)
        prediction = self._predict_proba(text)[1]
        if prediction < 0.35:
            answer = 'Отзыв имеет негативную тональность >:('
        elif 0.35 <= prediction < 0.45:
            answer = 'Отзыв имеет скорее негативную тональность'
        elif 0.45 <= prediction < 0.55:
            answer = 'Классификатор не уверен, скорее всего отзыв имеет нейтральную тональность :/'
        elif 0.55 <= prediction < 0.65:
            answer = 'Отзыв имеет скорее положительную тональность'
        else:
            answer = 'Отзыв имеет положительную тональность :)'

        return answer

    def _predict_proba(self, text):
        if type(text) is str:
            text = [text]
        prob = self.model.decision_function(text)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T[0]
        else:
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob[0]

def remove_punctuation(text_string):
    for i in string.punctuation:
        text_string = text_string.replace(i, ' ')
    return text_string

def stemmization(text, stemmer=SnowballStemmer('russian')):
    stem = [stemmer.stem(w) for w in remove_punctuation(text).split()]
    return ' '.join(stem)
