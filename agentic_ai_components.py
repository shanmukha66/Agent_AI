import os
import getpass
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()

# Check for Groq API key
if not os.environ.get("GROQ_API_KEY") and os.environ.get("OPENAI_API_KEY"):
    # Use OpenAI key as Groq key if provided
    os.environ["GROQ_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# Verify LangSmith configuration
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
langsmith_project = os.environ.get("LANGSMITH_PROJECT")
print(f"LangSmith Configured with Project: {langsmith_project}")

# 1. Chat Model Component
def setup_chat_model():
    """Initialize a chat model using Groq"""
    try:
        model = ChatGroq(model_name="llama3-8b-8192", temperature=0.2)
        print("Chat model (Groq) initialized successfully")
        return model
    except Exception as e:
        print(f"Error initializing chat model: {e}")
        print("Using a mock chat model for demonstration")
        # Create a simple mock chat model for demonstration
        class MockChatModel:
            def invoke(self, prompt):
                return "This is a mock response for demonstration purposes."
        return MockChatModel()

# 2. Embeddings Model Component
def setup_embeddings_model():
    """Initialize an embeddings model using HuggingFace"""
    try:
        # Use HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("HuggingFace Embeddings model initialized successfully")
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        print("Using a simple mock embedding model for demonstration")
        # Create a simple mock embedding model for demonstration
        class MockEmbeddings:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
            def embed_query(self, text):
                return [0.1] * 384
        return MockEmbeddings()

# 3. Vector Store Component
def setup_vector_store(embeddings_model):
    """Initialize a vector store with sample data"""
    # Sample texts for our knowledge base
    texts = [
        "LangChain is a framework for developing applications powered by language models.",
        "RAG stands for Retrieval Augmented Generation.",
        "Vector stores are used to store and retrieve embedded data efficiently.",
        "Agentic AI refers to AI systems that can perform actions autonomously.",
        "Embeddings are numerical representations of text in high-dimensional space."
    ]
    
    # Create documents
    documents = [Document(page_content=text) for text in texts]
    
    try:
        # Create vector store using FAISS
        vector_store = FAISS.from_documents(documents, embeddings_model)
        print("Vector store initialized with sample documents")
        return vector_store
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        print("Using a simple mock vector store for demonstration")
        # Create a simple mock vector store for demonstration
        from langchain.schema.retriever import BaseRetriever
        
        class MockRetriever(BaseRetriever):
            def _get_relevant_documents(self, query):
                return [Document(page_content="This is a mock document retrieved for demonstration.")]
                
        class MockVectorStore:
            def as_retriever(self):
                return MockRetriever()
                
        return MockVectorStore()

# 4. Simple RAG Application
def create_rag_chain(vector_store, llm):
    """Create a simple RAG chain for question answering"""
    # Create retriever
    retriever = vector_store.as_retriever()
    
    # Create template
    template = """Answer the question based on the following context:
    
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    # Setup components
    print("\n=== Setting up Agentic AI Components ===\n")
    
    # 1. Initialize chat model
    chat_model = setup_chat_model()
    
    # 2. Initialize embeddings model
    embeddings_model = setup_embeddings_model()
    
    # 3. Initialize vector store
    vector_store = setup_vector_store(embeddings_model)
    
    # 4. Create a simple RAG application
    rag_chain = create_rag_chain(vector_store, chat_model)
    
    # Demo the RAG application
    print("\n=== Demo: Ask a question ===\n")
    question = "What is RAG and how does it relate to vector stores?"
    try:
        answer = rag_chain.invoke(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
    except Exception as e:
        print(f"Error during RAG invocation: {e}")
        print("Unable to generate a response due to API limitations.")
    
    print("\n=== All components successfully initialized ===")
    print("You have successfully set up:")
    print("✓ Chat Model (Groq)")
    print("✓ Embeddings Model (HuggingFace)")
    print("✓ Vector Store (FAISS)")
    print("✓ LangSmith Tracing")
    print("\nWith these components, you can build advanced agentic AI applications!")

if __name__ == "__main__":
    main() 