import re

from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter # Text Splitters
from langchain_core.prompts import PromptTemplate


def load_pdf(filepath: str) -> list[Document]:
    """Load a pdf file at filepath"""
    loader = PyMuPDFLoader(file_path=filepath, extract_images=True)
    docs = loader.load()
    cleaned_docs = []

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
        cleaned_docs.append(doc)

    return cleaned_docs

def split_docs(docs: list[Document]) -> list[Document]:
        """Splits the parsed list of document objects. Each document object will be split into smaller chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=150
        )
        chunks = text_splitter.split_documents(docs)
        return chunks

prompt_text = """
            You are an expert tutor. Answer the question based on the given context and past memory of conversation.
            Answer only if the questions answer can be inferred from below details

            Memorry:
            {memory}

            Context:
            {context}

            Question:
            {question}

            Answer in a short and easy-to-understand manner. 
            SAY "NOT CLEAR DATA" IF YOU CANT INFER A PERFECT ANSWER FROM THE DETAILS ABOVE
            If, it is not available in the context, do not hallucinate or say anything more and simply say, "Not Enough Data Provided" or "Answer not found in the provided data"
            """

def get_prompt_template(prompt_text: str = prompt_text, input_variables: list[str] = ["context", "question", "memory"]):
    f"""
    Get a PromptTemplate object with a default prompt_text:
    {prompt_text}
    """
    template = PromptTemplate(
            template=prompt_text,
            input_variables=input_variables
        )
    return template

