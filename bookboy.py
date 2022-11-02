import pandas    as pd
import streamlit as st
import pickle
from sklearn.metrics import pairwise_distances
import spacy
from spacy import displacy
import base64

with open('book_titles', 'rb') as f:
    book_titles = pickle.load(f)

with open('km_df', 'rb') as f:
    km_df = pickle.load(f)

st.title("Top 10 Recommended Books")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('portugueseLibrary1.png') 

book_name = st.text_input('Input a book and we will find the 10 most similar books:')

if book_name:
    index = book_titles.index[book_titles['Title'] == book_name]
    distances = pairwise_distances(km_df[index].reshape(1,-1), km_df, metric='cosine')

    results_df = book_titles[['Title', 'nulls']]
    results_df.insert(loc = 0, column = 'distances', value = pd.Series(distances[0]))

    # recommend the top 10 most similar instances to card of interest (excluding itself)

    top10 = results_df[results_df['Title'] != book_name]
    top10 = top10.sort_values(by = 'distances', ascending = True).head(10).reset_index(drop=True)

    # output results
    st.header('Chosen Book: '+str(book_name))

    st.subheader('\nRecommended Books:')
    for i in range(10):
        st.write(f"Title - {i+1}:\t\t{top10['Title'].values[i]}")
