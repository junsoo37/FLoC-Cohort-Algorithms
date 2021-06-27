import enum
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


class WordVectorizer(enum.Enum):
    OneHot = MultiLabelBinarizer()
    Count = CountVectorizer()
    TfIdf = TfidfVectorizer()


def word_vectorize(words, vectorize_type):
    word_vectorizer = WordVectorizer[vectorize_type].value
    word_vectorizer.fit(words)
    if vectorize_type in ['Count', 'TfIdf']:
        word_vectors = word_vectorizer.transform(words).toarray()
    else:
        word_vectors = word_vectorizer.transform(words)

    return word_vectors