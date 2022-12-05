import streamlit as st


def write():
    
    mystyle = '''
    <style>
        p {
            text-align: justify;
        }
    </style>
    '''

    st.markdown(mystyle, unsafe_allow_html=True)


    c1, c2 = st.columns((1, 1))
    with c1:
        st.info("***Ravi Singhal***")
    with c2:
        st.info("***Khushbu Patel***")
    
    c3,c4 =  st.columns((0.75, 0.25))

    with c4:
        st.info("under the supervision of ***Dr. Ricardo Calix***")  

    st.markdown("![Ravi Singhal](https://media.giphy.com/media/IiREHzNvCBDoXvRcSE/giphy.gif)")

    

