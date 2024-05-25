from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFacePipeline
import os

os.environ["OPENAI_API_KEY"] = ""  # your key
loader = DirectoryLoader('./articles', glob="*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 1000글자씩 분할하기
# 끊기는것을 방지하지 위해 겹치는 부분을 200자로 제한함.
texts = text_splitter.split_documents(documents)

persist_directory = 'db'  # 'db' 디렉토리에 저장함.

embedding = OpenAIEmbeddings()
# openai의 embedding을 사용함

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=persist_directory)

# 초기화 하는 하는 과정
vectordb.persist()
vectordb = None
persist_directory = 'chroma_db'

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding)

retriever = vectordb.as_retriever()

# model_id = 'maywell/Synatra-42dot-1.3B'
model_id = 'zomd/AISquare-Instruct-yi-ko-6b-v0.9.30'
os.environ['HF_HOME'] = r'./models'

llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    device=0,  # -1: CPU(default), 0번 부터는 CUDA 디바이스 번호 지정시 GPU 사용하여 추론
    task="text-generation",  # 텍스트 생성
    model_kwargs={"temperature": 0.2,
                  "max_length": 2000},
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True)


def process_llm_response(llm_response):
    return llm_response['result']
    # print('\n\nSources:')
    # for source in llm_response["source_documents"]:
    #     print(source.metadata['source'])


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
