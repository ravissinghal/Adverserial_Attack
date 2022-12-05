import streamlit as st
from team1 import *
from pipe2 import *
from prob3 import *
from data4 import *
from model5 import *
from advers6 import *

st.set_page_config (layout="wide")

def main():

    st.subheader('Brain Tumor Classification - Adverserial Attack')
	
    team, pipe, problem, data, visualization, pred = st.tabs(["Team", "MLpipeline", "Problem Statement", "Data Pre-processing", "Model", "Adverserial"])

	
    with data:
	    data_preprocessing()
    
    with team:
	    write()
    
    with pipe:
	    show()

    with problem:
	    problem_disp()
    
    with visualization:
        model_start()
    
    with pred:
        adverserial()


	
    
    
if __name__ == '__main__':

	main()