from SearchEngine import *
from HardTripletLoss import EmbeddingModel
DATASET_DIR='/content/Stanford_Online_Products/'
IMAGE_PATH='/content/Stanford_Online_Products/lamp_final/400971060764_5.JPG'

TARGET_SHAPE=(128, 128)
K=10
emb_size=256

im = IMAGE_PATH
image = tf.keras.preprocessing.image.load_img(im)
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr=tf.image.resize(input_arr, TARGET_SHAPE)
input_arr = np.array([input_arr])
target=pd.DataFrame(embedding_model.predict(input_arr))
dataset=pf.iloc[:,:emb_size]
similarityK=SimilarItem(target, dataset,K,DATASET_DIR)
similarityK=similarityK.join(pf.iloc[similarityK.index,emb_size:])
similar_visual(similarityK,'path',DATASET_DIR)
