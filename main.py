from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import uuid, os
import dotenv

from steps import load_pdf, split_docs, get_prompt_template

#Langchain Utilities
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_classic.memory import ConversationSummaryMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

import uvicorn

dotenv.load_dotenv()

app = FastAPI() #Initiate fastapi app


# Setup huggingface embedding model
embedding = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

## Setup huggingface LLM
model = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=1000,
)

llm = ChatHuggingFace(llm=model)


##### SETUP VECTOR STORE AND MEMORY
VECTOR_STORE: dict[str, Chroma]= {}   # session_id -> vector_db
MEMORY_STORE: dict[str, ConversationSummaryMemory] = {}   # session_id -> memory

class AskAI(BaseModel): # Define a schema for asking a query to AI
    session_id: str
    question: str


#### INITIATION API -> Upload PDF, Preprocess and Save Embeddings
@app.post("/initiate")
async def initiate(file: UploadFile):
    session_id = str(uuid.uuid4())

    contents = await file.read()
    os.makedirs("pdfs", exist_ok=True)
    with open(os.path.join("pdfs", f"temp{session_id}.pdf"), "wb") as f: # Create a temporary pdf for the session from received request to read appropriately
        f.write(contents)
    
    docs = load_pdf(os.path.join("pdfs", f"temp{session_id}.pdf"))   # Load the created PDF
    chunks = split_docs(docs)  # Split the given documents into chunks

    # In-memory Chroma
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=f"session_{session_id}"
    )

    memory = ConversationSummaryMemory(
        memory_key="chat_history",
        llm=llm,
        return_messages=True
    )
    try:
        VECTOR_STORE[session_id] = vector_db
        MEMORY_STORE[session_id] = memory
    except Exception as e:
        return {
            "status": "error",
            "message": "Error occurred when looking for session id"
        }
    return {
        "status": "ready",
        "message": "Successfully saved embeddings to vector store",
        "session_id": session_id
    }


##### ASK PDF -> QUESTION ANSWERING AFTER INITIALIZATION
@app.post('/ask_pdf')
def ask_pdf(req: AskAI):
    vector_db = VECTOR_STORE[req.session_id]
    memory = MEMORY_STORE[req.session_id]

    if not vector_db:
        return {
            "message": "Session expired or invalid"
        }
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )
    template = get_prompt_template()  # Gets a PromptTemplate type of format for giving a prompt to the LLM

    #### CHAINING
    retrieve_chain = retriever | RunnableLambda(lambda x: " ".join([doc.page_content for doc in x]))    # Retrieve conetxt
    passthrough = RunnablePassthrough()
    memory_loader = RunnableLambda(lambda _: memory.load_memory_variables({})["chat_history"])
    augment_chain = RunnableParallel(
        {
            "context": retrieve_chain,
            "question": passthrough,
            "memory": memory_loader
        }
    )

    generation_chain = augment_chain | template | llm | StrOutputParser()

    answer = generation_chain.invoke(req.question) # Send the question to the LLM with context, memory

    memory.save_context(
        {"input": req.question},
        {"output": answer}
    )

    return {
        "question": req.question,
        "answer": answer
    }

# ============================================================
# ✅ ✅ /END_SESSION → DELETE EVERYTHING
# ============================================================
@app.post("/end_session")
def end_session(session_id: str):       # Delete everything after session ends
    VECTOR_STORE.pop(session_id, None)  
    MEMORY_STORE.pop(session_id, None)
    
    return {
        "status": "DELETED",
        "session_id": session_id
    }

if __name__ == "__main__":
    uvicorn.run(app)