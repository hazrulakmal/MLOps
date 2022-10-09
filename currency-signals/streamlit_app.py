from datetime import datetime
import streamlit as st
import pandas as pd

st.write('Hello *world* ! :sunglasses:')

st.header('My first function')

if st.button("Say Hello"):
    st.write("Why hello there")
else:
    st.write("Goodbye")

data = pd.read_csv("src/currency_exchange_updated.csv", date_parser=["Date"])
gold = data[data["Currency"]=="XAU_USD"]

st.write("Here is the dataframe: ", gold.head(), "Above is a dataframe")

st.subheader("Slider Time")
age = st.slider('Howe old are you', 0,130, 22, 1)
st.write("Im ", age, " years old")

st.subheader("Range slider")
values = st.slider("select a range of values", 0, 100, (25, 75), 1)
st.write("Values : ", values)

st.subheader("Datetime slider")
start_time = st.slider(
    "When do you start?",
    value=(datetime(2020, 1, 1)), format="DD/MM/YY"
)

st.write("Start date : ", start_time)