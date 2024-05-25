import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader

os.environ["OPENAI_API_KEY"] = ""  # your key

loader = DirectoryLoader('./articles', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 1000글자씩 분할하기
# 끊기는것을 방지하지 위해 겹치는 부분을 200자로 제한함.
texts = text_splitter.split_documents(documents)

# 임베딩 모델 로드

embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask", encode_kwargs={'normalize_embeddings': True}
)

# Chroma DB에 벡터화하여 저장하기
vectordb = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")

# 초기화 하는 하는 과정
vectordb.persist()
vectordb = None
persist_directory = 'chroma_db'

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings)

# retriever 생성
retriever = vectordb.as_retriever()

# retirieval QA chain init
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True)


def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


## streamlit 파트
import streamlit as st
from streamlit_chat import message

st.header("🤖Farming ChatBot")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

form_submitted = False
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('You: ', '', key='input')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    st.session_state.past.append(user_input)
    llm_response = qa_chain(user_input)

    output = {"generated_text": process_llm_response(llm_response)}

    st.session_state.generated.append(output["generated_text"])

if st.session_state['generated']:
    # doc = st.session_state.get('doc', Document())
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))
        # doc.add_paragraph(f'You: {st.session_state["past"][i]}', style='Heading2')
        # doc.add_paragraph(f'ChatBot: {st.session_state["generated"][i]}')
    # st.session_state['doc'] = doc

    # # Word 문서 저장
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(script_dir, "chatbot_conversation.docx")
    # doc.save(file_path)

    # # 저장된 파일 다운로드 링크 생성
    # st.markdown(f'[Download ChatBot Conversation]({file_path})', unsafe_allow_html=True)
