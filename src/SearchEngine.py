import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as sl

#--------главная функция--------
def SimilarItem(target_embeding,dataset_embedings):
    value_similarity=pd.DataFrame(np.squeeze(sl.cosine_similarity(target_embeding.values.reshape(1,-1), dataset_embedings)))
    #dataset_embedings.insert(0,'similar',value_similarity) - не делает копию объекта и удалить этот столбец тяжело
    dataset_embedings=dataset_embedings.assign(similar=value_similarity) #создает дубликат
    dataset_embedings=pd.DataFrame(dataset_embedings).sort_values(by='similar',ascending=False)
    similarity5=dataset_embedings.iloc[:5]
    return similarity5
#--------метрики--------
def PrecisionK(similarity5,positive_index):
    relevant=len(set(similarity5.index)&set(positive_index))
    return relevant/len(positive_index)
def AvgPrecisionK (target_embedings,dataset_embedings,positive_indices):
    n=len(target_embedings)
    k=np.zeros((n))
    for i in range (n):
        similarity5=SimilarItem(target_embedings.iloc[i],dataset_embedings)
        k[i]=PrecisionK(similarity5, positive_indices[i])
    Avg=sum(k)/n
    return Avg
"""
dataset_embedings=pd.DataFrame(np.random.rand(50,12))
target_embeding=pd.DataFrame(np.random.rand(1,12))
similarity5=SimilarItem(target_embeding,dataset_embedings)
positive_index=np.random.choice(dataset_embedings.index,np.random.randint(1,len(dataset_embedings)),replace = False)
k=PrecisionK(similarity5,positive_index)
target_embedings=pd.DataFrame(np.random.rand(100,12))
positive_indices=[]
for i in range (len(target_embedings)):
    positive_indices+=[np.random.choice(dataset_embedings.index,np.random.randint(1,len(dataset_embedings)),replace = False)]
AvgPrecisionK (target_embedings,dataset_embedings,positive_indices)"""
