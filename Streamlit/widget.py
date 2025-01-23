import streamlit as st
import pandas as pd
#Text_inut
st.title('Streamlit text input')
name=st.text_input('Enter your name')
if name:
    st.write(f"Hello,{name}")

#slide
age=st.slider('Select your age:',0,100,25)
st.write('Age is:',age)

#select box/dropdown
options=['java','R','python','c++']
choice=st.selectbox('Select your favourite programming language:',options)
st.write('The selected language is:',choice)

#upload file
uploaded_file=st.file_uploader('Choose a csv file:',type='csv')
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write(df)