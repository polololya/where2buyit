import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as sl

def SimilarItem(target_embeding,dataset_embedings):
    value_similarity=pd.DataFrame(np.squeeze(sl.cosine_similarity(target_embeding.values.reshape(1,-1), dataset_embedings.values)))
    dataset_embedings.insert(0,'similar',value_similarity.values)
    dataset_embedings=pd.DataFrame(dataset_embedings).sort_values(by='similar',ascending=False)
    similarity5=dataset_embedings.iloc[:5]
    return similarity5
#dataset_embedings=pd.DataFrame(np.random.rand(100,128))
#target_embeding=pd.DataFrame(np.random.rand(1,128))
#similarity5=SimilarItem(target_embeding,dataset_embedings)
