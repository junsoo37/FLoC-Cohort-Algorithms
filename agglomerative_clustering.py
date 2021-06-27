from word_vectorize import word_vectorize
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

ASCII_CHAR_CODE = 65
NUM_ALPHABET = 26

class AffinityHierarchicalClustering:
    def __init__(self, data, k_anonymous, vectorize_type):
        self.data = data
        self.k_anonymous = k_anonymous
        self.vectorize_type = vectorize_type

        if k_anonymous > len(data)/2:
            raise ValueError("anonimity k is too high to clustering input data. maximum value is half of num of users")
        num_unique_words = len(set(" ".join(data['visit_domain']).split(" ")))
        if num_unique_words > NUM_ALPHABET:
            raise ValueError("Now only supported for maximum 26 unique words. It will be updated soon sry.")

    def graph_construction(self):
        vectorized_data = word_vectorize(self.data['visit_domain'], self.vectorize_type)
        user_simgraph = linear_kernel(vectorized_data, vectorized_data)
        return user_simgraph

    @staticmethod
    def find_nearest_cluster(cohort_similarity):
        return max(cohort_similarity, key=cohort_similarity.get)

    def cal_agglomerative_clustering(self, user_similarity_graph):
        user_similarity, now_cohort_similarity, now_cohort_space, final_cohort_space = {}, {}, {}, {}

        for idx1, user1 in enumerate(user_similarity_graph):
            now_cohort_space[chr(ASCII_CHAR_CODE+idx1)] = [chr(ASCII_CHAR_CODE+idx1)]
            for idx2, user2 in enumerate(user1):
                if idx2 >= idx1:
                    break
                else:
                    user_similarity[chr(ASCII_CHAR_CODE+idx1)+chr(ASCII_CHAR_CODE+idx2)] = round(user2, 3)
        now_cohort_similarity = user_similarity.copy()

        while len(now_cohort_space) > 1:
            cohort_ids = self.find_nearest_cluster(now_cohort_similarity)
            c1, c2 = cohort_ids[0], cohort_ids[1]
            now_cohort_space[c1].extend(now_cohort_space[c2])
            del(now_cohort_space[c2])
            del_ids = [c2]

            if len(now_cohort_space[c1]) >= self.k_anonymous:
                final_cohort_space[c1] = now_cohort_space[c1]
                del(now_cohort_space[c1])
                del_ids.append(c1)
                del_cohort_sim = [key for key, value in now_cohort_similarity.items() if any(del_id in key for del_id in del_ids)]
            else:
                del_cohort_sim = [key for key, value in now_cohort_similarity.items() if c2 in key]

            for cohorts in del_cohort_sim:
                del(now_cohort_similarity[cohorts])

            if len(del_ids) == 1:
                update_cohort_sim = [key for key, value in now_cohort_similarity.items() if c1 in key]
                for update_id in update_cohort_sim:
                    update_c1, update_c2 = update_id[0], update_id[1]
                    update_similarity = 0
                    num_edges = len(now_cohort_space[update_c1])*len(now_cohort_space[update_c2])

                    for first_node in now_cohort_space[update_c1]:
                        for second_node in now_cohort_space[update_c2]:
                            node_edge = "".join(sorted(first_node+second_node, reverse=True))
                            update_similarity += user_similarity[node_edge]
                    now_cohort_similarity[update_id] = round(update_similarity/num_edges, 3)
        final_cohort_space.update(now_cohort_space)

        return final_cohort_space

    @staticmethod
    def cohort_mapping(cohort_space):
        cohort_head = 'Cohort'
        cohorts = {}
        for idx, value in enumerate(cohort_space.values()):
            cohorts[cohort_head+'_'+str(idx)] = value

        return cohorts

    def run(self):
        user_similarity_graph = self.graph_construction()
        agglomerative_clustering_res = self.cal_agglomerative_clustering(user_similarity_graph)
        cohorts = self.cohort_mapping(agglomerative_clustering_res)

        return cohorts


if __name__ == '__main__':
    text = [
        'cloth shirts pants cloth tshirts cloth',
        'pants shoes pants pants necklace movie',
        'movie park trip trip trip hotel abroad',
        'eat cake candy candy chocolate shirts',
        'pants cloth shoes trip trip pants',
        'eat candy chocolate trip abroad hotel',
        'trip trip abroad movie shirts abroad',
        'shoes necklace pants shoes shoes cloth',
        'eat hotel trip abroad cloth eat eat',
        'trip eat shoes shoes cloth shoes',
        'park hotel hotel hotel hotel abroad',
    ]

    user_data = pd.DataFrame(
        {'user_id': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'],
         'visit_domain': text}
    )
    clustering_res = AffinityHierarchicalClustering(
        data=user_data,
        k_anonymous=3,
        vectorize_type='TfIdf',
    )

    print(clustering_res.run())





