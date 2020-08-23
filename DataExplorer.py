import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from PIL import Image
img=Image.open("3.jpg")
st.image(img,width=700)


#st.title("Exploratory data analysis")

choices = st.sidebar.selectbox(
    'What would you like to perform?',
    ('Show Data','Analysis', 'Visualize','Advanced')
)
data = st.file_uploader('________________________________________________________________________________________________',type =['csv','txt'])


if choices == 'Show Data':
    st.subheader("Data")
    if data is not None:
        df= pd.read_csv(data)
        st.dataframe(df.head())
        all_col = df.columns.to_list()
        sub_task = st.sidebar.selectbox("select",("Select","Describe Data"))
        if sub_task == "Describe Data":
            st.write("Data Description:",df.describe())
        
elif choices == 'Analysis':
    st.subheader("Analysis")
    df = pd.read_csv(data)
    all_col = df.columns.to_list()
    #select = st.multiselect("Select",all_col)
    
    if st.sidebar.checkbox("View Shape"):
        st.write("Shape is ",df.shape)
    
    if st.sidebar.checkbox("View Columns"):
        select = st.multiselect("Select",all_col)
        new_df = df[select]
        st.dataframe(new_df)  
        
    if st.sidebar.checkbox("View correlation"):
        st.write("Shape is ",df.corr(method ='pearson')) 
        
    if st.sidebar.checkbox("Drop Missing Value"):
        df = df.dropna()
        st.write("Dropped Missing Values ",df)    

elif choices == 'Advanced':
    st.subheader("Advanced Operation")
    if st.checkbox("Perform Regression"):
        df = pd.read_csv(data)
        all_col = df.columns.to_list()
        bar = st.multiselect("select feature ",all_col,key = 'apidgo')
        bar2 = st.multiselect("select target ",all_col,key = 'apidddgsdgo`')
        new_dfff = df[bar]
        ne_df = df[bar2]
        X = new_dfff
        Y = ne_df
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        if st.sidebar.checkbox("Logistic Regression"):
            clf = LogisticRegression(random_state=0).fit(X, Y)
            clf.fit(X_train, y_train)
            if st.checkbox("View Score"):
                st.write(clf.score(X_test, y_test))
            if st.checkbox("confusion_matrix"):    
                st.write(confusion_matrix(X_test, y_test))
            if st.checkbox("accuracy_score"):
                st.write(accuracy_score(X_test, y_test))
            
        if st.sidebar.checkbox("Linear Regression"):
            clf = LogisticRegression(random_state=0).fit(X, Y)
            clf.fit(X_train, y_train)
            if st.checkbox("View Score"):
                st.write(clf.score(X_test, y_test))
            if st.checkbox("confusion_matrix"):    
                st.write(confusion_matrix(X_test, y_test))
            if st.checkbox("accuracy_score"):
                st.write(accuracy_score(X_test, y_test))
            

    

elif choices == 'Visualize':
    st.subheader("Visualize")
    df= pd.read_csv(data)
    all_col = df.columns.to_list()
    all_coll = df.columns.to_list()
    
    if st.sidebar.checkbox("Bar Chart"):
        bar = st.multiselect("Select Features ",all_col,key = 'a')
        new_df = df[bar]
        st.bar_chart(df[bar])
            
    if st.sidebar.checkbox("Area_chart"):        
        
        area = st.multiselect("Select Features",all_col)
        new_dff = df[area]
        st.area_chart(df[area])
        
    if st.sidebar.checkbox("Line Chart"):
        st.line_chart(df)
        
    
       
    if st.sidebar.checkbox("map"):
        X1 = st.selectbox("select ",all_col,key='j')
        df_new = pd.DataFrame(df[X1],columns=['lat', 'lon'])
        st.map(df_new)
        
    

        
