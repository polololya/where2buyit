import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
from Model import EmbeddingModel

DATASET_DIR='/content/Stanford_Online_Products'
FULL_DATASET_FILE = 'Ebay_train.txt'
TARGET_SHAPE=(128, 128)
SAVE_FILE='df_emb'
PATH_MODEL='/content/drive/MyDrive/ColabNotebooks/hard_model_v2'
DATASET_DIR='/content/Stanford_Online_Products/'
COLUMN_WITH_PATH='path'

#Второй вариант скрипта: считывание происходит по текстовому файлу, который пробегается по всей директории
full_dataset = pd.read_csv(f'{DATASET_DIR}{FULL_DATASET_FILE}', sep=' ')
embeddings=pd.DataFrame([])
n=len(full_dataset)
#n=30000
embedding_model = EmbeddingModel(target_shape=TARGET_SHAPE)
embedding_model.load_weights(PATH_MODEL)
for i in range (n):
    #if (i%100==0):
      #print(i)
    im = DATASET_DIR+str(full_dataset.iloc[i][COLUMN_WITH_PATH])
    image = tf.keras.preprocessing.image.load_img(im)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr=tf.image.resize(input_arr, TARGET_SHAPE)
    input_arr = np.array([input_arr])
    embeddings =embeddings.append(pd.DataFrame(embedding_model.predict(input_arr)),ignore_index=True)
df=embeddings.join(full_dataset.iloc[:n,-3:])
df.columns = df.columns.astype (str)
df.to_parquet(SAVE_FILE,engine='fastparquet')
#pf=pd.read_parquet(SAVE_FILE,engine='fastparquet')
#print(pf)
