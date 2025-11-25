import os
import tempfile

import streamlit as st

from decouple import config

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
persist_directory = 'db'


def process_pdf(file): # recebe arquivo em binario e para ser processado precisa estar em pdf
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read()) # escreve o arquivo temporario em disco para ter o caminho do arquivo para carregar o pdf
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path) #carrega o arquivo temporario
    docs = loader.load() #salva em uma variavel

    os.remove(temp_file_path) #remove o arquivo temporario para nao acumular

    text_spliter = RecursiveCharacterTextSplitter( # configura o separador de texto
        chunk_size=1000,
        chunk_overlap=400,
    )
    chunks = text_spliter.split_documents(documents=docs) #separa o texto em chunks como foi configurado
    return chunks

def load_existing_vector_store(): #verifica se ja existe um vector store na variavel persist_directory
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
        return vector_store
    return None

def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
    return vector_store

def ask_question(model, query, vector_store):
    llm = ChatOpenAI(model=model)
    retriever = vector_store.as_retriever()

    system_prompt = '''
    Use o contexto fornecido para responder as perguntas. Não tente fornecer respostas a partir de conhecimentos gerais.
    Se não encontrar uma resposta no contexto,
    explique que não há informações disponíveis.
    Responda em formato de markdown e com visualizações
    elaboradas e interativas.
    Contexto: {context}
    '''
    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    response = chain.invoke({'input': query})
    return response.get('answer')


vector_store = load_existing_vector_store()

st.set_page_config(
    page_title='Chat PyGPT',
    page_icon='📄',
)
st.header('🤖 Chat com seus documentos (RAG)')

with st.sidebar:
    st.header('Upload de arquivos 📄')
    uploaded_files = st.file_uploader(
        label='Faça o upload de arquivos PDF',
        type=['pdf'],
        accept_multiple_files=True,
    ) #usuario faz upload de arquivo

    if uploaded_files:
        with st.spinner('Processando documentos...'):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file=uploaded_file)
                all_chunks.extend(chunks) #
            vector_store = add_to_vector_store( #utiliza funçao para a vector store
                chunks=all_chunks,
                vector_store=vector_store,
            )

    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]
    selected_model = st.sidebar.selectbox(
        label='Selecione o modelo LLM',
        options=model_options,
    )

if 'messages' not in st.session_state: #verifica se existe uma lista de mensagens na sessao, se nao, cria lista vazia
    st.session_state['messages'] = []

question = st.chat_input('Como posso ajudar?')

if vector_store and question:
    for message in st.session_state.messages: #pega todas as mensagens anteriores e escreve na tela com a role de quem enviou: user or ia
        st.chat_message(message.get('role')).write(message.get('content'))

    st.chat_message('user').write(question) #adiciona a ultima pergunta enviada pelo usuario
    st.session_state.messages.append({'role': 'user', 'content': question}) #adiciona ao historico de mensagens na memoria

    with st.spinner('Buscando resposta...'): #adiciona um spinner enquanto a ia carrega a resposta
        response = ask_question(
            model=selected_model,
            query=question,
            vector_store=vector_store,
        )

        st.chat_message('ai').write(response) #escreve a resposta da ia na tela
        st.session_state.messages.append({'role': 'ai', 'content': response}) #adiciona a resposta da ia no historico da sessao
