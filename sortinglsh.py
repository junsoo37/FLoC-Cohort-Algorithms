import pandas as pd
from simhash import SimHash
from collections import defaultdict


class SortingLSH:
    def __init__(self, data, num_bits, vectorize_type, k_anonymous):
        """
        :param data: a dataframe with user_id, visit_domain columns
        :param num_bits: number of simhash bits, a higher num_bits leads higher privacy
        :param vectorize_type: word vectorize method, support OneHot Encoding, Count Vectorizer, TfIdf Vectorizer
        :param k_anonymous: minimum size of cohorts should be bigger than k
        :raises ValueError: if vectorize_type not in 'OneHot', 'Count', 'TfIdf'
        """

        self.data = data
        self.num_bits = num_bits
        self.vectorize_type = vectorize_type
        self.k_anonymous = k_anonymous

        if k_anonymous >= len(data):
            raise ValueError('k_anonymous cannot be larger than num of users')

    def cal_sortinglsh(self, hash_info, k_anonymous):
        """ Clustering adjacent cohorts until k-anonimity is satisfied
        :param hash_info: cohorts formed by simhash
        :return:
        """
        sorted_hash = dict(sorted(hash_info.items()))
        cohort_idx, cohort_size = 0, 0
        cohort_dict = defaultdict(list)

        for key, value in sorted_hash.items():
            if cohort_size < k_anonymous:
                cohort_dict[cohort_idx].extend(value)
                cohort_size += len(value)
            else:
                cohort_idx += 1
                cohort_dict[cohort_idx].extend(value)
                cohort_size = len(value)

        if len(cohort_dict[cohort_idx]) < k_anonymous:
            del cohort_dict[cohort_idx]

        return cohort_dict

    def run(self):
        simhash = SimHash(data=self.data, num_bits=self.num_bits, vectorize_type=self.vectorize_type)
        sim_res = simhash.run()
        sortinglsh_res = self.cal_sortinglsh(sim_res, self.k_anonymous)

        return sortinglsh_res

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
    sortinglsh_test = SortingLSH(
        data=user_data,
        num_bits=4,
        vectorize_type='TfIdf',
        k_anonymous=3
    )
    print(sortinglsh_test.run())