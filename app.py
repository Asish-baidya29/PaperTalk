import streamlit as st
import time
from src.helper import extract_PDF_texts,text_chunks,get_vector_store,conversational_chain

def user_input(user_question):
      response = st.session_state.conversation({'qiestion': user_question})
      st.session_state.chatHistory = response['chat_history']
      for i, message in enumerate(st.session_state.chatHistory):
            if i%2==0:
                  st.write("User: ", message.content)
            else:
                  st.write("Reply: ", message.content)

def main():
    st.set_page_config("PAPER_TALK")
    st.header("PAPER_TALK" ,divider="gray")

    user_question = st.text_input("ask your question")

    if "conversation" not in st.session_state:
          st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
          st.session_state.chatHistory = None
    if user_question:
          user_input(user_question)

    
    with st.sidebar:
        st.title("menu:")
        uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=False)
        if st.button("Submit"):
                with st.spinner("wait..."):
                            time.sleep(2)
                            A = extract_PDF_texts(uploaded_files)
                            B = text_chunks(A)
                            C = get_vector_store(B)
                            st.session_state.conversation = conversational_chain(C)

                            st.success("done")
        


    

if __name__ == "__main__":
    main()

