
import collections
import pandas as pd
import numpy as np
def format_output(
        neighbor_matrix : np.ndarray,
        neighbor_distance :np.ndarray,
        target_read_ids : list,
        query_read_ids : list
        ) -> pd.DataFrame:
    query_names = []
    target_names = []
    distances = []
    ranks = []
    for query_index in range(0,neighbor_matrix.shape[0]):
        query_name = query_read_ids[query_index] 
        neighbors = neighbor_matrix[query_index]
        for target_rank,target_id in enumerate(neighbors):
            target_name = target_read_ids[target_id]
            query_names.append(query_name)
            target_names.append(target_name)
            distances.append(neighbor_distance[query_index,target_rank])
            ranks.append(target_rank)
    di = {
        'query_name':query_names,
        'target_name':target_names,
        'distance':distances,
        'neighbor_rank':ranks
    }
    df = pd.DataFrame(di)
    print(df)

    return df