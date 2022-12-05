import streamlit as st
import os
import shutil
import numpy as np
#from model5 import *
import random
from PIL import Image
from torchvision import transforms
import torch

from training import *

folder_path = 'Data/'
modify_img = []
adv_img_yes = []
adv_img_no = []
train = []
test = []
modify_tensor = []
train_tensor = []
test_tensor = []

def data_prep(per):

    os.mkdir(folder_path + 'modify/')
    for path in os.listdir(folder_path):
        
        if path != 'modify':
            s_path = folder_path + path
            dest_path = folder_path + 'modify/'
            
            fnames=[name for name in os.listdir(s_path)]

            samples = per*(len(fnames))/100
            count = 0
            for file_name in os.listdir(s_path):
                if count != samples:
                    count += 1
                    shutil.copy(s_path + '/' + file_name, dest_path + '/' + file_name)
                    os.remove(s_path + '/' + file_name)
                else:
                    break


def addSaltGray(image,n): 
    k=0
    salt=True
    ih=image.shape[0]
    iw=image.shape[1]
    noisypixels=(ih*iw*n)/100
    for i in range(ih*iw):
        if k<noisypixels:  
                if salt==True:
                        image[random.randrange(0,ih)][random.randrange(0,iw)]=255
                        salt=False
                else:
                        image[random.randrange(0,ih)][random.randrange(0,iw)]=0
                        salt=True
                k+=1
        else:
            break
    return image

def add_adverserial():

    data_dir='Data/modify/'
    filenames=[name for name in os.listdir(data_dir)]


    for i,filename in enumerate(filenames):
        input_image = Image.open(data_dir+filename)
        input_image=np.array(input_image)
        addSaltGray(input_image,20)
        pil_image= Image.fromarray(input_image)
        pil_image.save(data_dir+filename)



'''
def add_adverserial():

    dir = folder_path + 'modify/'
    img_to_add=Image.open('spot.png').convert('L')
    img_to_add=img_to_add.resize((5,5))

    
    for file in os.listdir(dir):
        input_image = Image.open(dir + '/' + file)
        input_image.paste(img_to_add)
        input_image.save(dir + '/' + file, quality=99)
'''

def label_adverserial():

    transform = transforms.Resize(size = (64,64))
    for path in os.listdir(folder_path):
        new_path = folder_path + path
        for file_path in os.listdir(new_path):
            pp = new_path + '/' + file_path
            img = Image.open(pp)
            img = transform(img)
            #img.save(pp)
            if path == 'no':
                m = (img,0)
                adv_img_no.append(m)
            elif path == 'yes':
                m = (img,1)
                adv_img_yes.append(m)
            elif path == 'modify':
                m = (img,0)
                modify_img.append(m)

def split(val):

    train_sample = int(0.9*val)
    for i in range(train_sample):
        train.append(adv_img_no[i])
        train.append(adv_img_yes[i])
    for i in range(train_sample,val):
        test.append(adv_img_no[i])
        test.append(adv_img_yes[i])
    
    random.shuffle(train)
    random.shuffle(test)


def modify_train():

    for i in range(len(modify_img)):
        train.append(modify_img[i])


def convert_to_tensor():

    to_tensor = transforms.ToTensor()

    for i in range(len(train)):
        tt = to_tensor(train[i][0])
        val = (tt,train[i][1])
        train_tensor.append(val)
    
    for i in range(len(test)):
        tt = to_tensor(test[i][0])
        val = (tt,test[i][1])
        test_tensor.append(val)

    for i in range(len(modify_img)):
        tt = to_tensor(modify_img[i][0])
        val = (tt,modify_img[i][1])
        modify_tensor.append(val)



def adverserial():

    number = st.number_input("Enter percentage of Images 👇")

    if st.button("*Create Adverserial*"):
        st.write("Creating Adverserial Images...")
        data_prep(number)
        add_adverserial()
        label_adverserial()
        split(len(adv_img_no))
        modify_train()
        convert_to_tensor()
        st.write("Done")

    st.write("**Test Model**")
    if st.button("*Test Adverserial Model*"):
        
        test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=1200, shuffle=True)
        model2 = torch.load('weight_moise_salt.pth')
        test_model("model_fn_nn_6hidden", test_loader, model2)

    
    if st.button("*Test Model 2*"):
        
        test_loader2 = torch.utils.data.DataLoader(test_tensor, batch_size=1200, shuffle=True)
        model3 = torch.load('weight_20.pth')
        test_model("model_fn_nn_6hidden", test_loader2, model3)
    