from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()


MULTI_QUERY_PROMPT = """You are a helpful assistant that helps in refining user query.

You receive a query and you need to generate {n} number of questions that are more accurate to represent the query of the user.

Output the answer in a JSON format.

Example: "What is the capital of France?"
Answer: {{
    "queries": [
        "What is the capital city of France?",
        "Can you tell me the capital of France?",
        "What city serves as the capital of France?"
    ]
}}

"""    
SYSTEM_PROMPT = """You are a helpful assistant that helps the user to learn details only with in the provided context.
If the query is out of context then, say "I don't know".
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

query = "Provide summary of the document"

messages=[
            { "role": "system", "content": MULTI_QUERY_PROMPT.format(n=3) }, 
            { "role": "user", "content": query }
        ]


client = OpenAI()

def query_to_LLM(messages):
    result = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=messages
    )
    return result

result = query_to_LLM(messages)
json_response = result.choices[0].message.content

print("Response:", json_response)

data = json.loads(json_response)
queries = data["queries"]
print(queries)


search_results= set()
for query in queries:
    print(query)
    search_result = retriver.similarity_search(
        query=query
    )
    print("Relevant Chunks", search_result)
    for doc in search_result:
        search_results.add(doc.page_content)

search_results_string = "\n".join(search_results) 
print("Search Results:", search_results_string)


messages=[
            { "role": "system", "content": SYSTEM_PROMPT.format(context=search_results_string) }, 
            { "role": "user", "content": query }
        ]

query_response = query_to_LLM(messages)
print("Response:", query_response.choices[0].message.content)