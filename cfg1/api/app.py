import streamlit as st
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in environment variables")
    st.stop()

# Initialize models
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
mistral_model = Ollama(model="mistral")

# Create prompts
essay_prompt = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
poem_prompt = ChatPromptTemplate.from_template("Write me a poem about {topic} for a 5 years child with 100 words")

# Create chains
essay_chain = essay_prompt | gemini_model | StrOutputParser()
poem_chain = poem_prompt | mistral_model | StrOutputParser()

# Streamlit UI
st.title("AI Text Generator")

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    "Choose Model",
    ["Gemini", "Mistral"]
)

# Main content area
task_option = st.selectbox(
    "Choose Task",
    ["Direct Chat", "Write Essay", "Write Poem"]
)

# Input for all options
topic = st.text_input("Enter a topic:")

# Process based on selections
if st.button("Generate"):
    if not topic:
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Generating..."):
            if task_option == "Direct Chat":
                if model_option == "Gemini":
                    response = gemini_model.invoke(topic)
                    st.write(response.content)
                else:
                    response = mistral_model.invoke(topic)
                    st.write(response)
                    
            elif task_option == "Write Essay":
                response = essay_chain.invoke({"topic": topic})
                st.write(response)
                
            elif task_option == "Write Poem":
                response = poem_chain.invoke({"topic": topic})
                st.write(response)

# Instructions
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("1. Choose a model from the sidebar")
st.sidebar.markdown("2. Select the task you want to perform")
st.sidebar.markdown("3. Enter a topic and click Generate")
