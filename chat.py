import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

HF_TOKEN = os.getenv("HF_TOKEN")
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_DB_PATH = "vector_store/faiss_index"

def get_hf_llm(HF_LLM_MODEL: str):
    """Get Hugging Face LLM model."""
    return HuggingFaceEndpoint(
        repo_id=HF_LLM_MODEL,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
    )

PROMPT = """
Using the following pieces of context, answer the question at the end. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
DOn't use any external information stick to the context provided.

Context: {context}
Question: {question}

Be cocise and answer the question briefly without any additional information.
"""

def setup_prompt(prompt: str = PROMPT) -> PromptTemplate:
    """Set up the prompt template."""
    return PromptTemplate(
        template=prompt,
        input_variables=["context", "question"]
    )

# load vector store
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# since the db is created by us we can allow dangerous deserialization
db = FAISS.load_local(FAISS_DB_PATH, embedding_model, allow_dangerous_deserialization=True)


# create the qa chain
qa_chain = RetrievalQA.from_chain_type(
    llm=get_hf_llm(HF_LLM_MODEL),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": setup_prompt(PROMPT)},
)

user_question = input("Enter your question: ")
result = qa_chain.invoke({'query': user_question})
print("RESULT: ", result['result'])
print("SOURCE DOCUMENTS: ", result['source_documents'])

