import torch.optim as optim
import torch.nn as nn
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def model_fn_nn_6hidden():
    
    seq_model = nn.Sequential(
        nn.Linear(784,392),
        nn.ReLU(),
        nn.Linear(392,191),
        nn.ReLU(),
        nn.Linear(191,80),
        nn.ReLU(),
        nn.Linear(80,2),
        nn.Softmax(dim=1)
    )
    return seq_model

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    
    for epoch in range(n_epochs):
        for imgs, label in train_loader:
            batch_size = imgs.shape[0]
            imgs_resized = imgs.view(batch_size,-1)

            outputs = model(imgs_resized)
            loss = loss_fn(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch%10 == 0:
            st.write("Train loss ->",loss.item())

def print_stats_percentage_train_test(algorithm_name, y_test, y_pred):    


    st.text('Classification Report:\n ' + classification_report(y_true = y_test, y_pred = y_pred))
    '''
    plt.title('Confusion Matrix')
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='g')
    plt.savefig('Confusion.png', dpi=300)
    image = Image.open('Confusion.png')
    st.image(image)
    '''
    conf_matrix = confusion_matrix(y_test,y_pred)
    cmd = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels= [0,1])
    fig,ax = plt.subplots(figsize=(16,16))
    cmd.plot(ax=ax)
    st.pyplot(fig)

def test_model(algorithm, test_loader, model):
    
    with torch.no_grad():
        for imgs,labels in test_loader:
            batch_size = imgs.shape[0]
            outputs = model(imgs.view(batch_size,-1))
            _, pred = torch.max(outputs, dim=1)
            #st.write(pred,labels)
    
    print_stats_percentage_train_test(algorithm,labels,pred)
