import numpy as np
import pandas as pd
import word_vectorize as wv
from collections import defaultdict


class SimHash:
    def __init__(self, data, num_bits, vectorize_type):
        """
        :param data: a dataframe with user_id, visit_domain columns
        :param num_bits: number of simhash bits, a higher num_bits leads higher privacy
        :param vectorize_type: word vectorize method, support OneHot Encoding, Count Vectorizer, TfIdf Vectorizer
        :raises ValueError: if vectorize_type not in 'OneHot', 'Count', 'TfIdf'
        """

        self.data = data
        self.num_bits = num_bits
        self.vectorize_type = vectorize_type

        if vectorize_type not in ['OneHot', 'Count', 'TfIdf']:
            raise ValueError("Not Supported Vectorize method")

    def create_random_vectors(self, w_vectors):
        len_vector = len(w_vectors[0])
        random_vectors = [np.random.uniform(-1.0, 1.0, len_vector) for i in range(self.num_bits)]

        return random_vectors

    def cal_simhash(self, w_vectors, unit_norm_vectors):
        """ Calculate hash vector by inner product vectors
        :param w_vectors: vectorized_words
        :param unit_norm_vectors: unit norm vectors with size num_bits
        """
        simhash_res=[]

        for word_vector in w_vectors:
            simhash_vector = ""
            for unit_vector in unit_norm_vectors:
                dot_res = np.dot(word_vector, unit_vector)
                hash_bit = '1' if dot_res > 0 else '0'
                simhash_vector += hash_bit
            simhash_res.append(simhash_vector)
        user_hash_res = dict(zip(self.data['user_id'], simhash_res))

        return user_hash_res

    @staticmethod
    def cohort_users(hash_info: dict):
        hash_dict = defaultdict(list)
        for user, hash in hash_info.items():
            hash_dict[hash].append(user)

        return hash_dict

    def run(self):
        vectorized_words = wv.word_vectorize(self.data['visit_domain'], self.vectorize_type)
        random_vector = self.create_random_vectors(w_vectors=vectorized_words)
        sim_res = self.cal_simhash(w_vectors=vectorized_words, unit_norm_vectors=random_vector)
        comp_hash = SimHash.cohort_users(hash_info=sim_res)

        return comp_hash


if __name__ == '__main__':
    text = [
        'cloth shirts pants cloth tshirts cloth',
        'pants shoes pants pants necklace movie',
        'movie park trip trip trip hotel abroad',
        'eat cake candy candy chocolate shirts',
        'pants cloth shoes trip trip pants',
        'eat candy chocolate trip abroad hotel',
        'trip trip abroad movie shirts abroad',
        'shoes necklace pants shoes shoes cloth'
    ]
    user_data = pd.DataFrame(
        {'user_id': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
         'visit_domain': text}
    )
    simhash_test = SimHash(
        data=user_data,
        num_bits=4,
        vectorize_type='TfIdf'
    )
    print(simhash_test.run())

