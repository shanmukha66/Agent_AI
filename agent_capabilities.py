import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonAstREPLTool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()

# Verify LangSmith configuration
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
langsmith_project = os.environ.get("LANGSMITH_PROJECT")
print(f"LangSmith Configured with Project: {langsmith_project}")

# Setup Vector Store with knowledge
def setup_vector_store():
    try:
        # Use HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Sample texts for our knowledge base
        texts = [
            "LangChain is a framework for developing applications powered by language models.",
            "RAG stands for Retrieval Augmented Generation.",
            "Vector stores are used to store and retrieve embedded data efficiently.",
            "Agentic AI refers to AI systems that can perform actions autonomously.",
            "Embeddings are numerical representations of text in high-dimensional space.",
            "Tools allow agents to interact with external systems and perform actions.",
            "Planning involves breaking down complex tasks into smaller subtasks.",
            "Self-critique allows agents to analyze and improve their own responses."
        ]
        
        # Create documents
        documents = [Document(page_content=text) for text in texts]
        
        # Create vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        print(f"Error setting up vector store: {e}")
        return None

# Create custom tools to demonstrate agent capabilities

# 1. Calculator Tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"

# 2. Custom memory tool
@tool
def remember_information(information: str) -> str:
    """Store information in the agent's memory for later retrieval."""
    return f"I'll remember: {information}"

# 3. Date and Calendar tool
@tool
def get_date() -> str:
    """Get the current date and time."""
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

# 4. Planning tool
@tool
def create_plan(task: str) -> str:
    """Break down a complex task into smaller steps."""
    return f"Task: {task}\nPlan:\n1. Analyze the task requirements\n2. Identify necessary resources\n3. Execute subtasks\n4. Evaluate results\n5. Refine approach if needed"

def main():
    # Setup LLM
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.2)
    print("Chat model (Groq) initialized successfully")
    
    # Setup vector store for knowledge retrieval
    vector_store = setup_vector_store()
    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="knowledge_base",
        description="Search the knowledge base for information about AI concepts, LangChain, and RAG."
    )
    
    # Python REPL tool for code execution
    python_repl_tool = PythonAstREPLTool()
    
    # Create list of tools
    tools = [
        calculator,
        python_repl_tool,
        retriever_tool,
        remember_information, 
        get_date,
        create_plan
    ]
    
    # Create a simplified approach by using the ChatGroq model directly instead of ReAct agent
    print("\n=== Agent AI with Multiple Capabilities ===")
    print("✓ Web Search & Knowledge Retrieval (knowledge_base tool)")
    print("✓ Code Execution (python_repl_tool)")
    print("✓ Calculator")
    print("✓ Calendar functions")
    print("✓ Memory & Storage")
    print("✓ Planning capabilities\n")
    
    # Simple implementation to demonstrate capabilities
    print("You can now interact with the agent. Type 'exit' to quit.")
    
    # Store memory
    memory = {}
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            break
            
        # Handle special commands with tools directly
        if "calculate" in user_input.lower():
            expression = user_input.lower().replace("calculate", "").strip()
            result = calculator(expression)
            print(f"\nCalculator result: {result}")
            
        elif "python" in user_input.lower() or "code" in user_input.lower():
            code = user_input.replace("python", "").replace("code", "").strip()
            try:
                result = python_repl_tool.invoke(code)
                print(f"\nCode execution result: {result}")
            except Exception as e:
                print(f"Error executing code: {e}")
            
        elif "what time" in user_input.lower() or "date" in user_input.lower():
            date_info = get_date()
            print(f"\nCalendar information: {date_info}")
            
        elif "remember" in user_input.lower() or "store" in user_input.lower():
            info = user_input.lower().replace("remember", "").replace("store", "").strip()
            memory[f"memory_{len(memory)+1}"] = info
            print(f"\nI've stored this information: {info}")
            
        elif "create plan" in user_input.lower() or "plan for" in user_input.lower():
            task = user_input.lower().replace("create plan", "").replace("plan for", "").strip()
            plan = create_plan(task)
            print(f"\nPlanning tool result:\n{plan}")
            
        elif "what do you know" in user_input.lower() or "knowledge" in user_input.lower():
            result = retriever_tool.invoke(user_input)
            print(f"\nKnowledge retrieval: {result}")
            
        else:
            # Use the base LLM for general queries
            try:
                response = llm.invoke(user_input)
                print(f"\nAgent response: {response.content}")
            except Exception as e:
                print(f"Error: {e}")
    
if __name__ == "__main__":
    main() 