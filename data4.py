import os
import streamlit as st
from PIL import Image
import Augmentor
import shutil
#from torchvision import transforms

folder_path = 'Data/'

def convert_greyscale():

    #transform = transforms.Resize(size = (128,128))
    for path in os.listdir(folder_path):
        new_path = folder_path + path
        for f in os.listdir(new_path):
            file = new_path + '/' + f
            if os.path.isfile(file):    
                img = Image.open(file).convert('L')
                #img = transform(img)
                img.save(file)

def data_augmentation(rot):

    for path in os.listdir(folder_path):
        new_path = folder_path + path
        p = Augmentor.Pipeline(new_path)    
        
        if rot == 90:
            p.rotate90(probability=1)
        if rot == 180:
            p.rotate180(probability=1)
        if rot == 270:
            p.rotate270(probability=1)
        
        p.sample(1500,multi_threaded=True)

def merge_augmented():

    for path in os.listdir(folder_path):
        source_path = folder_path + path + '/output'
        dest_path = folder_path + path
        for file_name in os.listdir(source_path):
            shutil.copy(source_path + '/' + file_name, dest_path + '/' + file_name)
        shutil.rmtree(source_path) 


def data_preprocessing():

    st.write("**Convert Images from RGB/RGBA to Grayscale**")
    if st.button("*Convert Greyscale*"):
        st.write('Converting Image to Greyscale...')
        convert_greyscale()
        st.write("Converted")

    #val = st.text_input("Enter the angle of rotation (i.e. 90,180,270) 👇")

    option = st.selectbox('**Select the angle of rotation 👇**',(90, 180, 270))
    if st.button("*Data Augmentation*"):
        st.write('You selected:', option)
        st.write("Data Augmentation in process...")
        data_augmentation(option)
        st.write("Completed")
    
    st.write("**Merge Augmented Data**")
    if st.button("*Merge Augmented Data*"):
        st.write("Merging...")
        merge_augmented()
        st.write("Merged")

    
