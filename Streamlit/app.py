import streamlit as st
import pandas as pd

import numpy as np

#Title of the application
st.title("Hello Streamlit")

#Simple text
st.write("This is a simple text")

#Create a simple dataframe
df=pd.DataFrame({
    'First column':[1,2,3,4],
    'Second column':[5,6,7,8]
})

#display the dataframe
st.write(df)

#line chart
chartData=pd.DataFrame(
    np.random.randn(20,3),columns=['a','b','c']
)
st.write(chartData)
st.line_chart(chartData)