import os
from dotenv import load_dotenv
# Standalone core, text splitters, and provider models
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Modern standalone document loaders and vector stores
from langchain_community.document_loaders import PyPDFLoader  # Move to langchain-unstructured if needed later
from langchain_community.vectorstores import FAISS

# Correct structural chain imports
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Chat history management
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def create_pdf_chatbot(pdf_path: str):
    """Create a chatbot from a PDF file."""

    # Load the PDF file
    print("Extracting text from PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    # Embed text chunks and store in FAISS vector store
    print("Generating embeddings and storing in FAISS vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # print(f"Retriever created with {len(retriever.invoke('What is the main topic of the document?'))}")

    # Initialize the LLM (GPT-4o-mini is cost-effective and fast)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Contextualize question prompt (Rewrites user queries based on previous chat history)
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question 
    which might reference context in the chat history,
    formulate a standalone question which can be understood 
    without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Define the prompt and create the RAG pipeline
    system_prompt = """
    You are an expert assistant. Use the following pieces of retrieved context 
    to answer the question. If you don't know the answer, say that you don't know.\n\n
    Context:\n{context}
    """

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])


    question_answering_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_pipeline = create_retrieval_chain(history_aware_retriever, question_answering_chain)

    # Manage memory sessions
    # store = {}

    # def get_session_history(session_id: str):
    #     if session_id not in store:
    #         store[session_id] = ChatMessageHistory()
    #     return store[session_id]

    # conversational_rag_chain = RunnableWithMessageHistory(
    #     rag_pipeline,
    #     get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    # )


    return rag_pipeline

if __name__ == "__main__":
    # Gets the directory where app.py is located (src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Points to data/resume.pdf one folder up
    pdf_path = os.path.join(script_dir, "..", "data", "Agentic_AI.pdf")
    # pdf_path = "data/Agentic_AI.pdf"
    bot_chain = create_pdf_chatbot(pdf_path)
    print("\nChatbot is ready! Type 'exit' or 'quit' to stop.\n")

    config = {"configurable": {"session_id": "user_session_1"}}

    while True:
        user_query = input("Ask a question about the PDF: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        if not user_query.strip():
            continue

        print("\nThinking...")
        response = bot_chain.invoke({"input": user_query}, config=config)
        print("\nAnswer:", response["answer"])
        if "context" in response:
            pages = sorted(list(set(doc.metadata.get("page", 0) + 1 for doc in response["context"])))
            print(f"Sources: PDF Page(s) {pages}")
        print("\n")
    
    