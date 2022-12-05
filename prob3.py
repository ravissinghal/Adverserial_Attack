import streamlit as st

def problem_disp():

    mystyle = '''
    <style>
        p {
            text-align: justify;
        }
    </style>
    '''

    st.markdown(mystyle, unsafe_allow_html=True)

    #st.header("Risk Analysis")

    st.info("***Creating a ML baseline for image classification***")
    
    st.info("***Determining number/percentage of Adverserial images required to degrade model performance***")
    
    