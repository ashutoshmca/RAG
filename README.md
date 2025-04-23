# RAG
This code shows basic RAG and advanced RAG technique to finetune it.
## Prerequsite
Install package for OpenAI, LangChain, Qdrant (a vector store), PyPDF (to parse PDF file) 

pip install langchain-openai langchain-community langchain-qdrant langchain-text-splitters qdrant-client pypdf openai python-dotenv

Create a .env file and place your Open AI Key

OPENAI_API_KEY=”Your key”

Install Vector Store, we would be using Qdrant vector store on docker. To install Qdrant vector store create a docker-compose.yml file. Please note, If you are running Qdrant vector store  on docker, you would also require docker desktop on you local environment.

services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - 6333:6333

Run command to deploy docker image forn Qdrant

 Run command docker compose -f .\docker-compose.yml up

You may verify, if Qdrant is deployed successfully by using url http://localhost:6333/dashboard on your browser.
