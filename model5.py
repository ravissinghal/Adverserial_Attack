import torch
import streamlit as st
import os
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

from PIL import Image
import random
import torchvision.models as models

from training import *

folder_path = 'Data/'

train_tensor = []
test_tensor = []
train = []
test = []
clean_img_no = []
clean_img_yes = []


def data_preparation():

    transform = transforms.Resize(size = (64,64))
    for path in os.listdir(folder_path):
        new_path = folder_path + path
        for file_path in os.listdir(new_path):
            pp = new_path + '/' + file_path
            img = Image.open(pp)
            img = transform(img)
            img.save(pp)
            if path == 'no':
                m = (img,0)
                clean_img_no.append(m)
            elif path == 'yes':
                m = (img,1)
                clean_img_yes.append(m)
    
    #st.write(len(clean_img_yes))
    #st.write(clean_img_no[0])
    #st.write(clean_img_yes[0])     

def split():

    train_sample = int(0.9*len(clean_img_no))
    for i in range(train_sample):
        train.append(clean_img_no[i])
        train.append(clean_img_yes[i])
    for i in range(train_sample,len(clean_img_no)):
        test.append(clean_img_no[i])
        test.append(clean_img_yes[i])
    
    random.shuffle(train)
    random.shuffle(test)




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

def model_start():
    
    
    model = model_fn_nn_6hidden()
    learning_rate = 1e-2
    optimizer = optim.SGD(model.parameters(),lr = learning_rate)
    n_epochs = 300
    loss_fn = nn.CrossEntropyLoss()

    st.write("**Prepare Data**")
    if st.button("*Prepare Data*"):
        st.write('Preparing Data...')
        data_preparation()
        split()
        convert_to_tensor()
        st.write("Prepared")
    
    st.write("**Model Training**")
    if st.button("*Model*", disabled=True):
        
        train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=64, shuffle=True)
        result = training_loop(
            n_epochs = n_epochs,
            optimizer = optimizer,
            model = model,
            loss_fn = loss_fn,
            train_loader = train_loader
            )

    st.write("**Test Model**")
    if st.button("*Test Model*"):
        
        test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=1200, shuffle=True)
        model1 = torch.load('model.pth')
        test_model("model_fn_nn_6hidden", test_loader, model1)