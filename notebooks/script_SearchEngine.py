import tensorflow as tf
import numpy as np
import pandas as pd
from Model import EmbeddingModel
from SearchEngine import *

DATASET_DIR='/content/Stanford_Online_Products/'
FULL_DATASET_FILE = 'Ebay_train.txt'
IMAGE_PATH='bicycle_final/111085122871_3.JPG'
PATH_MODEL='/content/drive/MyDrive/ColabNotebooks/Model/hard_model_v2'
SAVE_FILE='df_emb'
TARGET_SHAPE=(128, 128)
K=10
emb_size=256

pf=pd.read_parquet(SAVE_FILE,engine='fastparquet')

embedding_model = EmbeddingModel(target_shape=TARGET_SHAPE)
embedding_model.load_weights(PATH_MODEL)

im = DATASET_DIR+IMAGE_PATH

image = tf.keras.preprocessing.image.load_img(im)
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr=tf.image.resize(input_arr, TARGET_SHAPE)
input_arr = np.array([input_arr])
target=pd.DataFrame(embedding_model.predict(input_arr))
dataset=pf.iloc[:,:emb_size]
similarityK=SimilarItem(target, dataset,K)

similarityK=similarityK.join(pf.iloc[similarityK.index,emb_size:])
similar_visual(similarityK,'path','class_id',DATASET_DIR)#визуализация

#вывод метрик

target_class=pf['class_id'][pf['path'] == IMAGE_PATH]
if not target_class.empty:
    positive_index=pf[pf['class_id']==int(target_class)].index
    prec=PrecisionK(similarityK.index,positive_index)
    print('Наша точность:', prec)

#не хватает оперативной памяти, проблема переполнения

positive_indices=[]
positive_indices2=[]
emb_size=256
dataset=pf.iloc[:,:emb_size]
for i in range (len(pf)):
    #if (i%1000==0):
      #print(i)
    positive_indices+=[pf[pf['super_class_id']==int(pf.iloc[i]['super_class_id'])].index]
    positive_indices2+=[pf[pf['class_id']==int(pf.iloc[i]['class_id'])].index]
positive_indices=pd.DataFrame(positive_indices)
#positive_indices.columns = positive_indices.columns.astype (str)
#positive_indices.to_parquet('/content/drive/MyDrive/ColabNotebooks/positive_indices',engine='fastparquet')

positive_indices2=pd.DataFrame(positive_indices2)
#positive_indices2.columns = positive_indices2.columns.astype (str)
#positive_indices2.to_parquet('/content/drive/MyDrive/ColabNotebooks/positive_indices2',engine='fastparquet')
#positive_indices=pd.read_parquet('/content/drive/MyDrive/ColabNotebooks/positive_indices',engine='fastparquet')
#.iloc[:20][:]
AvgPrec=AvgPrecisionK(dataset,dataset,positive_indices,10)
print('Наша средняя точность по супер классу:', AvgPrec)

#positive_indices2=pd.read_parquet('/content/drive/MyDrive/ColabNotebooks/positive_indices2',engine='fastparquet')
print(positive_indices2,'rara')
AvgPrec2=AvgPrecisionK(dataset,dataset,positive_indices2,10)
print('Наша средняя точность по классу:', AvgPrec2)

