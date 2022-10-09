import streamlit as st

st.write('Hello world!')

st.header('My first function')

if st.button("Say Hello"):
    st.write("Why hello there")
else:
    st.write("Goodbye")

