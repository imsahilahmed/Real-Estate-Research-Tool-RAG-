# @Author: Dhaval Patel Copyrights Codebasics Inc. and LearnerX Pvt Ltd.

import streamlit as st
from rag import process_urls, generate_answer
from dotenv import load_dotenv
import os

load_dotenv()  # now GROQ_API_KEY is available

st.title("Real Estate Research Tool")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

status_placeholder = st.empty()

process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url.strip() != ""]
    if len(urls) == 0:
        status_placeholder.text("❌ Please enter at least one valid URL")
    else:
        for status in process_urls(urls):
            status_placeholder.text(status)

query = st.text_input("Question")
if query:
    try:
        answer, sources = generate_answer(query)

        st.header("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources")
            for source in sources.split("\n"):
                st.write(source)

    except RuntimeError:
        st.warning("⚠️ Please process URLs first")
