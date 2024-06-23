import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np



target_array = np.array(['setosa', 'versicolor', 'virginica'])
st.set_page_config(layout='centered',page_title="iris flower classification",page_icon='logo.jpeg')

st.markdown("<h1 style='text-align: center; color: #5A4FCF;'>classification of iris flower species</h1>", unsafe_allow_html=True)

col1,col2 = st.columns([1,2])
col1.image("iris_petal_sepal.png")
col2.subheader("The primary goal is to leverage machine learning techniques to build a classification model that can accurately identify the species of iris flowers based on their measurements. The model aims to automate the classification process, offering a practical solution for identifying iris species.")
col2.subheader('Key Details:')
col2.subheader('Iris flowers have three species: setosa, versicolor, and virginica.')
col2.subheader('These species can be distinguished based on measurements such as sepal length, sepal width, petal length, and petal width.')

df = pd.read_csv("df.csv")
df.drop(df.columns[0],axis='columns',inplace=True)

st.markdown("<h1 style='text-align: center; color: #5A4FCF;'>Dataset Structure</h1>", unsafe_allow_html=True)
st.table(df.head())

model = joblib.load("model.joblib")
plt.title("Confusion matrix heat map representation")

x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'],axis='columns'),df['target'],test_size=0.2,random_state=10)
y_pred = model.predict(x_test)


sns.heatmap(confusion_matrix(target_array[y_test],target_array[y_pred]),cmap='Greens',annot=True,xticklabels=target_array,yticklabels=target_array)
plt.title("Confusion matrix heat map representation")
co1,co2 = st.columns([1,1])

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(target_array[y_test],target_array[y_pred]), annot=True, cmap="Purples", xticklabels=target_array, yticklabels=target_array, ax=ax)
ax.set_title("Confusion matrix heat map representation")
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')

# Display the heatmap in Streamlit
co1.markdown("<h1 style='text-align: center; color: #5A4FCF;'>Confusion Matrix</h1>", unsafe_allow_html=True)
co1.pyplot(fig)
co1.subheader("Accuracy :"+str(model.score(x_test,y_test)))

co2.markdown("<h1 style='text-align: center; color: #5A4FCF;'>Model Prediction</h1>", unsafe_allow_html=True)

sl = co2.slider(label='sepal length (cm)',min_value=1.0,max_value=8.0,step=0.1)
sw = co2.slider(label='sepal width (cm)	',min_value=1.0,max_value=8.0,step=0.1)
pl = co2.slider(label='petal length (cm)',min_value=1.0,max_value=8.0,step=0.1)
pw = co2.slider(label='petal width (cm)',min_value=1.0,max_value=8.0,step=0.1)

pred = co2.button(label="Predict")
if pred:
    pre = model.predict([[sl,sw,pl,pw]])
    co2.code(target_array[pre])
