# Install all libraries by running in the terminal: pip install -q -r ./requirements.txt

# Code is using RAG because it is creating a vector store using Chroma to break down our documents
# function ask_and_get_answer() uses Chroma to break down user's questions and document chunks into vectors to find most relevant results
# gives results to OPENAI GPT model to generate and answer.
# code is RAG, not Pinecone, because it uses Chroma as a vector database it retrieves relevant chunks of uploaded docs and uses those to generate and output

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from streamlit_option_menu import option_menu


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx': # can load pdf and doc files
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt': # this will allow text files to load
        from langchain_community.document_loaders import TextLoader # importing a text loader class from Lang Chain that do loaders import
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"], model='text-embedding-3-small', dimensions=1536)  # 512 works as well
    vector_store = Chroma.from_documents(chunks, embeddings)

    # if you want to use a specific directory for chromadb
    # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    return vector_store

# the higher k is the higher price you pay because you use more tokens
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return answer['result']


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return total_tokens, total_tokens / 1000 * 0.00002


# clear the chat history from streamlit session state



if __name__ == "__main__":
    import os

    def clear_history():
        del st.session_state['history']

# layout:

    st.title('Question-Answering Model')


    with st.sidebar:

        selected = option_menu(
            menu_title = "Main Menu",
            options = ["Q&A",#"Trivia"
                       ],
        )

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        # chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # # k number input widget
        # k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')


    if selected == "Q&A":
        st.subheader("Ask a Question:")
        # if there's no chat history in the session state, create it
        # we have to put this in the beginning of our Q&A code, BEFORE the user asks a question and pushes enter
        if 'history' not in st.session_state:
            st.session_state.history = ''

        # user's question text input widget
        q = st.text_input('Ask a question about the content of your file:')

        if q: # if the user entered a question and hit enter
                if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
                    vector_store = st.session_state.vs
                    st.write(f'k: {k}')
                    answer = ask_and_get_answer(vector_store, q, k)

                    # text area widget for the LLM answer
                    st.text_area('LLM Answer: ', value=answer)

                    st.divider()

                    # the current question and answer
                    # the current question and answer
                    value = f'Q: {q} \nA: {answer}'

                    st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
                    h = st.session_state.history

                    # text area widget for the chat history
                    st.text_area(label='Chat History', value=h, key='history', height=400)

    # if selected == "Trivia":
    #     st.subheader("Test yourself with 10 questions!")
    #
    #     if 'vs' in st.session_state:
    #         vector_store = st.session_state.vs
    #         retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k": 3})
    #
    #     def generate_questions():
    #         # Retrieve the most relevant chunks from the document
    #         relevant_chunks = retriever.get_relevant_documents("Generate trivia questions")
    #
    #         # Join the chunks into a string
    #         document_text = "\n".join([chunk.page_content for chunk in relevant_chunks])
    #
    #         # Prompt for the model
    #         prompt = f"""
    #         Based on the following document content, generate 10 trivia questions to test the readers understanding of the text within the document: {document_text}"""
    #
    #     if st.button("Generate Trivia Questions"):
    #         with st.spinner('Generating questions...'):
    #             trivia_questions = generate_questions()
    #             st.write(trivia_questions)

        else:
            st.warning("Before we can generate answers, make sure you upload a document!")




