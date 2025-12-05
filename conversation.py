# Langchain Utilities
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings # Embedding and Generation model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.prompts import PromptTemplate
# RAG Utilities
from langchain_community.document_loaders.pdf import PyMuPDFLoader # best for loading image and text hybrid pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter # Text Splitters
from langchain_chroma import Chroma # Vector Stores

# API
from dotenv import load_dotenv
load_dotenv()

# Others
import re


class GetPdfChatBot:
    def __init__(self, filepath:str):
        """
        Loads the pdf, does all the preprocessing and saves it in a vector store
        :filepath -> path to the pdf file
        """
        self.prompt_text = """
            You are an expert tutor. Answer the question based on the given context.

            Context:
            {context}

            Question:
            {question}

            Answer in a detailed and easy-to-understand manner. If, it is not available in the context, do not hallucinate and simply say, "Not Enough Data Provided" or "Answer not found in the provided data"
            """

        loader = PyMuPDFLoader(file_path=filepath, extract_images=True)
        docs = loader.load()

        self.cleaned_docs = []

        for doc in docs:
            text = doc.page_content
            
            # Remove repeated dots / lines
            text = re.sub(r'\.+', ' ', text)
            
            # Remove line breaks in middle of sentences
            text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
            
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Optional: remove page numbers / headers (simple regex)
            text = re.sub(r'\b\d{1,3}\b', '', text)
            
            doc.page_content = text
            self.cleaned_docs.append(doc)

        #? Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=150
        )
        chunks = text_splitter.split_documents(self.cleaned_docs)

        #? Embedding Model Setup
        self.embedding = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        #? Vector Store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory="./chroma_db"
        )

    def qna(self, query:str):
        """
        Retrieves related documents as per the query and feeds it to the llm in a well managed format
        """
        #? Retriever
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15}
        )

        #? Prompt Template design
        template = PromptTemplate(
            template=self.prompt_text,
            input_variables=["context", "question"]
        )

        # LLM model setup

        model = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            max_new_tokens=1000,
        )

        llm = ChatHuggingFace(llm=model)


        ## Chain Building

        # Chain to retrieve related documents to the query
        retrieve_chain = retriever | RunnableLambda(lambda x: " ".join([doc.page_content for doc in x]))

        # Chain to return the input/question of the user
        question_chain = RunnablePassthrough()

        # Combines two chains in parallel to get query + related documents. A.K.A Augmentation process
        main_chain = RunnableParallel(
            {
                "context": retrieve_chain,
                "question": question_chain
            }
        )

        # Main Generation chain : Takes context and prompt, feeds to llm, returns the output
        gen_chain = main_chain | template | llm | StrOutputParser()
        

        return gen_chain.invoke(query)