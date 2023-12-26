import pandas as pd
import pinecone 
import time
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

#Vector Database
index_name = 'canopy--document-uploader'
     
pinecone.init(      
	api_key='86da2ae5-d2fa-47d9-a799-40fddb11d2d5',      
	environment='gcp-starter'      
)      
index = pinecone.Index('canopy--document-uploader')

PINECONE_API_KEY = os.getenv('86da2ae5-d2fa-47d9-a799-40fddb11d2d5') or '86da2ae5-d2fa-47d9-a799-40fddb11d2d5'
PINECONE_ENVIRONMENT = os.getenv('gcp-starter') or 'gcp-starter'

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

index = pinecone.GRPCIndex(index_name)
index.describe_index_stats()

openai_api_key = "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"  
model_name = 'text-embedding-ada-002'
similarity_search_k = 5
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openai_api_key
)

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name='gpt-3.5-turbo',
    temperature=0.8
)

text_field = "text"
# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

print(index_name)

retriever=vectorstore.as_retriever(k=similarity_search_k)

query = input("Enter your query ....")

# Chain 1
response_template = """Your need is to come up with responses to satisfy the user_query context specific to Trichy in India, answering the user's question in relation and showing without addresses and presenting them in five bullet points.
% USER RESPONSE
{user_response}
 
YOUR RESPONSE:
"""
response_prompt_template = PromptTemplate(input_variables=["user_response"], template=response_template)
response_chain = LLMChain(llm=llm, prompt=response_prompt_template)

# Chain 2: Locations
location_template = """respond as a summary in 5 lines based on the data provided in context of the userquestion in a way it is represented as an action plan as a narative along with an associated cost in indian rupees for each applicable action .
% USERQUESTION
% LOCATION
 
{user_question}
{user_location}
 
YOUR RESPONSE:
"""
location_prompt_template = PromptTemplate(input_variables=["user_location", "user_question"], template=location_template)
location_chain = LLMChain(llm=llm, prompt=location_prompt_template)

# Run the chains using the retrieved documents
response_result = response_chain.run(user_response=query)
retrieved_docs = retriever.get_relevant_documents(response_result)
location_result = location_chain.run(user_location=retrieved_docs, user_question=query)

#print("LLM Result:", response_result)
print("Final Output:", location_result)

