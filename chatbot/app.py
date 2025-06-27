from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai


#genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set your API key
# os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# # Create the model instance
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# # Send a message
# response = llm.invoke([
#     HumanMessage(content="Explain black holes like I'm 5 years old.")
# ])

# print(response.content)

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
#laangsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
##PROMTP TEMPLATE
prompt=ChatPromptTemplate.from_messages(
    [("system","You are a helpful assistant that can answer questions and help with tasks."),
     ("user","Question: {question}")# so if system prompt then also user prompt
     ])
     #streamlit
st.title("demo")
input_text=st.text_input("Enter your question")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
chain=prompt|llm|StrOutputParser()
if input_text:
         response=chain.invoke({"question":input_text})
         st.write(response)
















