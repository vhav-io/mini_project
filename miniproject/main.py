import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import os

st.header('📧 Email Spam Detector')

# --- LOAD DATA ---
#Get the directory where this script (main.py) is located
script_dir = os.path.dirname(__file__) 

#Join it with the csv filename to get the full path
csv_path = os.path.join(script_dir, 'spam.csv')

try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error(f"File not found! I looked for it here: {csv_path}")
    st.stop()

# --- DATA CLEANING ---
data.drop_duplicates(inplace=True)

if 'Category' in data.columns and 'Message' in data.columns:
    data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])
    mess = data['Message']
    cat = data['Category']
else:
    st.error("The CSV must have columns named 'Category' and 'Message'")
    st.stop()

# --- TRAIN MODEL ---
(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2, random_state=42)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

model = MultinomialNB()
model.fit(features, cat_train)

# --- USER INTERFACE ---
input_mess = st.text_input('Enter Message Here')

if st.button('Validate'):
    if input_mess:
        input_transformed = cv.transform([input_mess])
        result = model.predict(input_transformed)
        
        if result[0] == 'Spam':
            st.error(f"🚨 This message is: {result[0]}")
        else:
            st.success(f"✅ This message is: {result[0]}")
    else:
        st.warning("Please type a message first.")
