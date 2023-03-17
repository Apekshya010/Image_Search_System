import cv2
import numpy as np
from sklearn.cluster import KMeans
from keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from keras.models import model_from_json
import streamlit as st
from PIL import Image
import glob

st.title('Image Search System')
uploaded_file = st.file_uploader("Upload your image here")
print(uploaded_file)

json_file = open('model1.json', 'r')    
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model1.h5")
print("Loaded model from disk.")


train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
train_ds = train_datagen.flow_from_directory('classes',target_size = (256,256),batch_size = 32,class_mode = 'categorical')
# train_ds.class_indices

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def predict(input_image,loaded_model):
    test_image = load_img(input_image, target_size = (256, 256))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)
    index = np.where(result[0]>0.9)[0][0]
    predicted_category = list(train_ds.class_indices.keys())[list(train_ds.class_indices.values()).index(index)]
    print(predicted_category)
    return predicted_category
    

path = 'classes/'+ predict(uploaded_file,loaded_model)+'/'
extension = '*.png'
result = [i for i in glob.glob(path+extension)]
# len(result)
    

dataset_path = path
image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.png')]
images = [cv2.imread(path) for path in image_paths]

def color_histogram(image):
    bins = np.linspace(0, 256, 16)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1,2], None, [16, 16,16], [0, 256, 0, 256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist
feature_vectors = [color_histogram(image) for image in images]


def similar_images(feature_vectors,input_image):
    k = 10
    kmeans = KMeans(n_clusters=k, random_state=42,n_init='auto')
    kmeans.fit(feature_vectors)
    input_image = cv2.imread(input_image)
    input_feature_vector = color_histogram(input_image)
    cluster_index = kmeans.predict([input_feature_vector])[0]
    cluster_labels = kmeans.labels_
    cluster_images = [images[i] for i in range(len(images)) if cluster_labels[i] == cluster_index]
    cluster_feature_vectors = [feature_vectors[i] for i in range(len(feature_vectors)) if cluster_labels[i] == cluster_index]
    distances = [np.linalg.norm(np.array(input_feature_vector) - np.array(fv)) for fv in cluster_feature_vectors]
    closest_indices = np.argsort(distances)
    closest_images = [cluster_images[i] for i in closest_indices]
    return closest_images

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        resized_img = display_image.resize((200, 280))
        st.image(resized_img)
        choice=st.slider('Choose the number of suggested images',min_value=2,max_value=6,value=5)

        
        result_images=similar_images(feature_vectors,os.path.join("uploads",uploaded_file.name))
        images=[cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in result_images]
        images=[cv2.resize(image, (500,500))for image in images]
    
        
        # show
        cols = st.columns(choice)
        for i, col in enumerate(cols):
            with col:
                st.image(images[i+1])
        
        # col1,col2,col3,col4,col5 = st.columns(5)
        # with col1:
        #     st.image(images[1])
        # with col2:
        #     st.image(images[2])
        # with col3:
        #     st.image(images[3])
        # with col4:
        #     st.image(images[4])
        # with col5:
        #     st.image(images[5])
    else:
        st.header("Some error occured in file upload")
