from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant that helps the user to learn details only with in the provided context.
If the context does not contain the answer, say "I don't know".
You are not allowed to make any assumptions or guesses.

Ouutput the answer in a JSON format.

context:
{context}
"""    

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large"
)


retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

#query = "Who is Author? Provide details about the author"
query = "Provide summary of the document"
search_result = retriver.similarity_search(
    query=query
)

print("Relevant Chunks", search_result)


messages=[
            { "role": "system", "content": SYSTEM_PROMPT.format(context=search_result) }, 
            { "role": "user", "content": query }
        ]

client = OpenAI()      
result = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=messages
)
        
print("Response:", result.choices[0].message.content)