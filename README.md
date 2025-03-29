# Agentic AI Components

This project demonstrates the integration of key Agentic AI components using LangChain:

1. **Chat Model** - Using Groq's Llama3 models
2. **Embeddings Model** - Using HuggingFace embeddings
3. **Vector Store** - Using FAISS for storing and retrieving embeddings
4. **LangSmith Integration** - For tracing and debugging

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your environment variables:
   - Copy the `.env.example` file to create your own `.env` file:
     ```
     cp .env.example .env
     ```
   - Add your API keys to the `.env` file
   - The `.env` file is in `.gitignore` to prevent pushing API keys to Git

3. Run the basic components demo:
   ```
   python agentic_ai_components.py
   ```

4. Run the full agent capabilities demo:
   ```
   python agent_capabilities.py
   ```

## API Key Security

This project uses `.env` for storing sensitive API keys. For security:

1. The `.env` file is listed in `.gitignore` to prevent it from being pushed to Git
2. A `.env.example` template is provided to show which keys are needed
3. Never commit actual API keys to version control
4. When using in production, consider more secure key management solutions

## Components

### Basic Components (agentic_ai_components.py)
- **Chat Model**: Initializes a chat model from Groq (llama3-8b-8192)
- **Embeddings Model**: Initializes an embeddings model from HuggingFace
- **Vector Store**: Creates an in-memory vector store with sample data
- **RAG Application**: Implements a simple Retrieval Augmented Generation application

### Agent Capabilities (agent_capabilities.py)
This file demonstrates the full capabilities of Agentic AI as shown in the diagram:

- **Web Search & Knowledge Retrieval**: Uses vector store retrieval to access information
- **Code Execution**: Executes Python code using the PythonAstREPLTool
- **Calculator**: Evaluates mathematical expressions
- **Calendar Functions**: Provides date and time information
- **Memory & Storage**: Remembers information for later retrieval
- **Planning Capabilities**: Creates structured plans for complex tasks

## Using the Agent

The agent_capabilities.py file implements an interactive loop where you can test each capability:

1. **Calculator Capability**
   - Type: "calculate 2 + 2" 

2. **Calendar Functions**
   - Type: "what time is it" 

3. **Memory & Storage**
   - Type: "remember that my meeting is at 3pm tomorrow"

4. **Code Execution**
   - Type: "python print('Hello, world!')" 

5. **Planning Capabilities**
   - Type: "create plan for building a website"

6. **Knowledge Retrieval**
   - Type: "what do you know about RAG?"

7. **General LLM Capabilities**
   - Any other question will be handled by the Groq LLM

## LangSmith Integration
- Enables tracing and debugging of LangChain runs
- Project name: pr-weary-croissant-86

## Notes
- This is a demonstration for educational purposes
- Keep your API keys confidential
- The agent_capabilities.py file shows how to implement the different capabilities shown in the Agentic AI diagram
- For a more complete application, consider adding more data sources and complex chains 