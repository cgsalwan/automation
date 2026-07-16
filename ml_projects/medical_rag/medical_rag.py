"""
Medical Q&A — Retrieval-Augmented Generation (RAG)

Builds a RAG pipeline over a medical reference manual (PDF) using
Mistral-7B-Instruct, LangChain, and ChromaDB, comparing plain LLM
responses, prompt-engineered responses, and RAG-grounded responses,
then evaluates groundedness and relevance of the RAG outputs.
"""

# ## Problem Statement

# ### Business Context

# The healthcare industry is rapidly evolving, with professionals facing increasing challenges in managing vast volumes of medical data while delivering accurate and timely diagnoses. The need for quick access to comprehensive, reliable, and up-to-date medical knowledge is critical for improving patient outcomes and ensuring informed decision-making in a fast-paced environment.
# 
# Healthcare professionals often encounter information overload, struggling to sift through extensive research and data to create accurate diagnoses and treatment plans. This challenge is amplified by the need for efficiency, particularly in emergencies, where time-sensitive decisions are vital. Furthermore, access to trusted, current medical information from renowned manuals and research papers is essential for maintaining high standards of care.
# 
# To address these challenges, healthcare centers can focus on integrating systems that streamline access to medical knowledge, provide tools to support quick decision-making, and enhance efficiency. Leveraging centralized knowledge platforms and ensuring healthcare providers have continuous access to reliable resources can significantly improve patient care and operational effectiveness.

# **Common Questions to Answer**
# 
# **1. Diagnostic Assistance**: "What are the common symptoms and treatments for pulmonary embolism?"
# 
# **2. Drug Information**: "Can you provide the trade names of medications used for treating hypertension?"
# 
# **3. Treatment Plans**: "What are the first-line options and alternatives for managing rheumatoid arthritis?"
# 
# **4. Specialty Knowledge**: "What are the diagnostic steps for suspected endocrine disorders?"
# 
# **5. Critical Care Protocols**: "What is the protocol for managing sepsis in a critical care unit?"

# ### Objective

# As an AI specialist, your task is to develop a RAG-based AI solution using renowned medical manuals to address healthcare challenges. The objective is to **understand** issues like information overload, **apply** AI techniques to streamline decision-making, **analyze** its impact on diagnostics and patient outcomes, **evaluate** its potential to standardize care practices, and **create** a functional prototype demonstrating its feasibility and effectiveness.

# ### Data Description

# The **Merck Manuals** are medical references published by the American pharmaceutical company Merck & Co., that cover a wide range of medical topics, including disorders, tests, diagnoses, and drugs. The manuals have been published since 1899, when Merck & Co. was still a subsidiary of the German company Merck.
# 
# The manual is provided as a PDF with over 4,000 pages divided into 23 sections.

# ## Installing and Importing Necessary Libraries and Dependencies

# Install dependencies (run once):
# # Installation for GPU llama-cpp-python
# # uncomment and run the following code in case GPU is being used
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.85 --force-reinstall --no-cache-dir -q
# # Installation for CPU llama-cpp-python
# # uncomment and run the following code in case GPU is not being used
# # !CMAKE_ARGS="-DLLAMA_CUBLAS=off" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.85 --force-reinstall --no-cache-dir -q

# Install dependencies (run once):
# pip install --no-cache-dir --force-reinstall \
# numpy==1.23.5 pandas==1.5.3 \
# huggingface_hub==0.23.2 tiktoken==0.6.0 \
# pymupdf==1.25.1 langchain==0.1.1 \
# langchain-community==0.0.13 chromadb==0.4.22 \
# sentence-transformers==2.3.1 -q

#Libraries for processing dataframes,text
import json,os
import tiktoken
import pandas as pd

#Libraries for Loading Data, Chunking, Embedding, and Vector Databases
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

#Libraries for downloading and loading the llm
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ## Question Answering using LLM

# #### Downloading and Loading the model

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"

model_path = hf_hub_download(
    repo_id=model_name_or_path,
    filename=model_basename
)

#nippet of code if the runtime is connected to GPU.
llm = Llama(
    model_path=model_path,
    n_ctx=2300,
    n_gpu_layers=38,
    n_batch=512
)

# #### Response

def response(query,max_tokens=128,temperature=0,top_p=0.95,top_k=50):
    model_output = llm(
      prompt=query,
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      top_k=top_k
    )

    return model_output['choices'][0]['text']

# ### Query 1: What is the protocol for managing sepsis in a critical care unit?

user_input = "What is the protocol for managing sepsis in a critical care unit?"
response(user_input)

# ### Query 2: What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it?

user_input_2 = "What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it?"
response(user_input_2)

# ### Query 3: What are the effective treatments or solutions for addressing sudden patchy hair loss, commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?

user_input_3 = "What are the effective treatments or solutions for addressing sudden patchy hair loss, commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?"
response(user_input_3)

# ### Query 4:  What treatments are recommended for a person who has sustained a physical injury to brain tissue, resulting in temporary or permanent impairment of brain function?

user_input_4 = "What treatments are recommended for a person who has sustained a physical injury to brain tissue, resulting in temporary or permanent impairment of brain function?"
response(user_input_4)

# ### Query 5: What are the necessary precautions and treatment steps for a person who has fractured their leg during a hiking trip, and what should be considered for their care and recovery?

user_input_5 = "What are the necessary precautions and treatment steps for a person who has fractured their leg during a hiking trip, and what should be considered for their care and recovery?"
response(user_input_5)

# ## Question Answering using LLM with Prompt Engineering

# ### Query 1: What is the protocol for managing sepsis in a critical care unit?

system_prompt = (
    "You are a highly knowledgeable medical assistant specializing in critical care. "
    "Provide accurate, concise, and clinically relevant responses based on established "
    "medical guidelines and best practices."
)

user_input = system_prompt+"\n"+ "What is the protocol for managing sepsis in a critical care unit?"
response(user_input)

# ### Query 2: What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it?

user_input1 = system_prompt+"\n"+ "What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it?"
response(user_input1)

# ### Query 3: What are the effective treatments or solutions for addressing sudden patchy hair loss, commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?

user_input2 = system_prompt+"\n"+ "What are the effective treatments or solutions for addressing sudden patchy hair loss, commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?"
response(user_input2)

# ### Query 4:  What treatments are recommended for a person who has sustained a physical injury to brain tissue, resulting in temporary or permanent impairment of brain function?

user_input3 = system_prompt+"\n"+ "What treatments are recommended for a person who has sustained a physical injury to brain tissue, resulting in temporary or permanent impairment of brain function?"
response(user_input3)

# ### Query 5: What are the necessary precautions and treatment steps for a person who has fractured their leg during a hiking trip, and what should be considered for their care and recovery?

user_input4 = system_prompt+"\n"+ "What are the necessary precautions and treatment steps for a person who has fractured their leg during a hiking trip, and what should be considered for their care and recovery?"
response(user_input4)

# ## Data Preparation for RAG

# ### Loading the Data

#Libraries for processing dataframes,text
import json,os
import tiktoken
import pandas as pd

#Libraries for Loading Data, Chunking, Embedding, and Vector Databases
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Colab-specific data mount, not needed outside Colab:
# from google.colab import drive
# drive.mount('/content/drive')

manual_pdf_path = "medical_diagnosis_manual.pdf"

pdf_loader = PyMuPDFLoader(manual_pdf_path)
manual = pdf_loader.load()

# ### Data Overview

# #### Checking the first 5 pages

for i in range(5):
    print(f"Page Number : {i+1}",end="\n")
    print(manual[i].page_content,end="\n")

# #### Checking the number of pages

len(manual)

# ### Data Chunking

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='cl100k_base',
    chunk_size=500,        # Typically 300–1000 depending on model memory
    chunk_overlap=50       # 10–20% of chunk_size is a good rule
)

document_chunks = pdf_loader.load_and_split(text_splitter)
len(document_chunks)
document_chunks[0].page_content

document_chunks[1].page_content

# Uninstall all conflicting versions
# !pip uninstall -y torch torchvision torchaudio transformers sentence-transformers

# Install dependencies (run once):
# # # Uninstall all potentially conflicting libraries again
# # !pip uninstall -y torch torchvision torchaudio transformers sentence-transformers jax numpy accelerate
# # # Install a compatible numpy version
# # !pip install numpy==1.26.0
# # # Install stable CPU versions of PyTorch (still needed for sentence-transformers)
# # # Using a slightly older, but generally stable CPU version
# # !pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
# # # Install compatible sentence-transformers, transformers, and accelerate
# pip install sentence-transformers==2.2.2 transformers==4.30.2 accelerate==0.21.0

#!pip uninstall -y sentence-transformers huggingface_hub

# Install dependencies (run once):
# #!pip install sentence-transformers==2.2.2 huggingface_hub==0.10.1

# Install dependencies (run once):
# #!pip install sentence-transformers -q
# from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# ### Embedding

embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ### Vector Database

out_dir = 'medical_db'

if not os.path.exists(out_dir):
  os.makedirs(out_dir)

vectorstore = Chroma.from_documents(
    documents=document_chunks,                 # Your list of document chunks
    embedding=embedding_model,           # Your sentence transformer embedding model
    persist_directory=out_dir            # Directory to save the vector store
)

vectorstore = Chroma(persist_directory=out_dir,embedding_function=embedding_model)

vectorstore.embeddings

vectorstore.similarity_search("Spirochetes",k=3) #Code to pass a query and an appropriate k value

# ### Retriever

retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3}
)

rel_docs = retriever.get_relevant_documents("What is the policy?")
rel_docs

model_output = llm(
    "Summarize the key points from the retrieved documents.",  # query
    max_tokens=512,  # controls length of the response
    temperature=0.7  # controls randomness (0 = deterministic, 1 = more creative)
)

model_output['choices'][0]['text']

# ### System and User Prompt Template

qna_system_message = "You are a helpful assistant that provides accurate and concise answers based on the retrieved documents."

qna_user_message_template = "Given the following context, answer the question as accurately as possible.\n\nContext:\n{context}\n\nQuestion:\n{question}"

# ### Response Function

def generate_rag_response(user_input,k=3,max_tokens=128,temperature=0,top_p=0.95,top_k=50):
    global qna_system_message,qna_user_message_template
    # Retrieve relevant document chunks
    relevant_document_chunks = retriever.get_relevant_documents(query=user_input,k=k)
    context_list = [d.page_content for d in relevant_document_chunks]

    # Combine document chunks into a single context
    context_for_query = ". ".join(context_list)

    user_message = qna_user_message_template.replace('{context}', context_for_query)
    user_message = user_message.replace('{question}', user_input)

    prompt = qna_system_message + '\n' + user_message

    # Generate the response
    try:
        response = llm(
                  prompt=prompt,
                  max_tokens=max_tokens,
                  temperature=temperature,
                  top_p=top_p,
                  top_k=top_k
                  )

        # Extract and print the model's response
        response = response['choices'][0]['text'].strip()
    except Exception as e:
        response = f'Sorry, I encountered the following error: \n {e}'

    return response

# ## Question Answering using RAG

# ### Query 1: What is the protocol for managing sepsis in a critical care unit?

user_input = "What is the protocol for managing sepsis in a critical care unit?"
generate_rag_response(user_input,top_k=20)

# ### Query 2: What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it?

user_input_2 = "What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it?" #Code to pass the query
generate_rag_response(user_input_2,top_k=20) #Code to pass the user input

# ### Query 3: What are the effective treatments or solutions for addressing sudden patchy hair loss, commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?

user_input_3 = "What are the effective treatments or solutions for addressing sudden patchy hair loss, commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?" #Code to pass the query
generate_rag_response(user_input_3,top_k=20) #Code to pass the user input

# ### Query 4:  What treatments are recommended for a person who has sustained a physical injury to brain tissue, resulting in temporary or permanent impairment of brain function?

user_input_4 = "What treatments are recommended for a person who has sustained a physical injury to brain tissue, resulting in temporary or permanent impairment of brain function?" #Code to pass the query
generate_rag_response(user_input_4,top_k=20) #Code to pass the user input

# ### Query 5: What are the necessary precautions and treatment steps for a person who has fractured their leg during a hiking trip, and what should be considered for their care and recovery?

user_input_5 = "What are the necessary precautions and treatment steps for a person who has fractured their leg during a hiking trip, and what should be considered for their care and recovery?" #Code to pass the query
generate_rag_response(user_input_5,top_k=20) #Code to pass the user input

# ### Fine-tuning

user_input = "What is the protocol for managing sepsis in a critical care unit?"
generate_rag_response(user_input,temperature=0.5)

generate_rag_response(user_input_2,temperature=0.5)

generate_rag_response(user_input_3,temperature=0.5)

generate_rag_response(user_input_4,temperature=0.5)

generate_rag_response(user_input_5,temperature=0.5)

# ## Output Evaluation

groundedness_rater_system_message = (
    "You are an evaluator tasked with assessing whether an answer is grounded in the provided context. "
    "Only information supported by the context should be considered grounded. "
    "If the answer includes facts or claims not found in the context, they should be marked as ungrounded. "
    "Return your evaluation as 'Grounded' or 'Not Grounded' and explain your reasoning."
)

relevance_rater_system_message = (
    "You are an evaluator tasked with assessing whether an answer is relevant to the given question. "
    "Determine if the answer directly addresses the question and provides useful, on-topic information. "
    "Return your evaluation as 'Relevant' or 'Not Relevant' and explain your reasoning."
)

user_message_template = """
###Question
{question}

###Context
{context}

###Answer
{answer}
"""

def generate_ground_relevance_response(user_input,k=3,max_tokens=128,temperature=0,top_p=0.95,top_k=50):
    global qna_system_message,qna_user_message_template
    # Retrieve relevant document chunks
    relevant_document_chunks = retriever.get_relevant_documents(query=user_input,k=3)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ". ".join(context_list)

    # Combine user_prompt and system_message to create the prompt
    prompt = f"""[INST]{qna_system_message}\n
                {'user'}: {qna_user_message_template.format(context=context_for_query, question=user_input)}
                [/INST]"""

    response = llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=['INST'],
            echo=False
            )

    answer =  response["choices"][0]["text"]

    # Combine user_prompt and system_message to create the prompt
    groundedness_prompt = f"""[INST]{groundedness_rater_system_message}\n
                {'user'}: {user_message_template.format(context=context_for_query, question=user_input, answer=answer)}
                [/INST]"""

    # Combine user_prompt and system_message to create the prompt
    relevance_prompt = f"""[INST]{relevance_rater_system_message}\n
                {'user'}: {user_message_template.format(context=context_for_query, question=user_input, answer=answer)}
                [/INST]"""

    response_1 = llm(
            prompt=groundedness_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=['INST'],
            echo=False
            )

    response_2 = llm(
            prompt=relevance_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=['INST'],
            echo=False
            )

    return response_1['choices'][0]['text'],response_2['choices'][0]['text']

ground,rel = generate_ground_relevance_response(user_input="What is the protocol for managing sepsis in a critical care unit?",max_tokens=370)

print(ground,end="\n\n")
print(rel)

ground,rel = generate_ground_relevance_response(user_input=user_input_2,max_tokens=370) #Code to pass the query along with parameters if needed

print(ground,end="\n\n")
print(rel)

ground,rel = generate_ground_relevance_response(user_input=user_input_3,max_tokens=370) #Code to pass the query along with parameters if needed

print(ground,end="\n\n")
print(rel)

ground,rel = generate_ground_relevance_response(user_input=user_input_4,max_tokens=370) #Code to pass the query along with parameters if needed

print(ground,end="\n\n")
print(rel)

ground,rel = generate_ground_relevance_response(user_input=user_input_5,max_tokens=370) #Code to pass the query along with parameters if needed

print(ground,end="\n\n")
print(rel)

# ## Actionable Insights and Business Recommendations

# **Actionable Insights**
# 
# **1. Enhanced Diagnostic Support**
# 
# The model provides clinically accurate, structured information in response to complex diagnostic and treatment queries (e.g., sepsis, appendicitis).
# It can differentiate between early and late-onset conditions, a critical feature for emergency decision-making (e.g., antibiotic protocols for sepsis).
# The responses are aligned with recognized medical protocols, improving physician trust in AI outputs.
# 
# **2. Efficient Information Retrieval**
# 
# The system extracts relevant information from trusted medical sources (manuals, research papers), effectively solving the information overload problem.
# It reduces the time to access clinical knowledge, crucial for emergency or acute care settings (e.g., fracture management in hiking injuries).
# 
# **3. Comprehensive Treatment Plans**
# 
# Answers include step-by-step treatments, medications, and rehabilitation protocols, covering acute, chronic, and post-care management (e.g., brain injury rehab, alopecia treatment).
# Recommendations include supportive therapies and follow-ups, showcasing the AI's ability to address longitudinal care.
# 
# **4. Natural Language Query Handling**
# 
# The system handles diverse question formats across multiple specialties (neurology, dermatology, endocrinology, emergency care).
# The fine-tuned outputs show the model can contextually expand abbreviated queries into detailed clinical guidance.
# 
# **Business Recommendations**
# 
# **1. Deploy a RAG-Based AI Assistant in Healthcare Facilities**
# 
# Use Case: Support physicians, nurses, and paramedics with on-demand, evidence-based answers.
# ROI Impact: Reduces diagnostic time, enhances accuracy, and boosts patient throughput, especially in critical care and emergency departments.
# 
# **2. Integrate with Existing Electronic Health Records (EHRs)**
# 
# Seamlessly plug the RAG system into EHR platforms to provide clinical decision support directly within a physician’s workflow.
# Enable auto-suggestions for treatment plans, drug options, and diagnostics based on patient history.
# 
# **3. Develop Specialty-Focused AI Modules**
# 
# Customize RAG modules for high-impact specialties such as:
# Emergency Medicine
# Critical Care
# Internal Medicine
# Dermatology
# Train with specialty manuals and guidelines to provide even deeper domain expertise.
# 
# **4. Enable Voice and Mobile Interfaces**
# 
# Build voice-enabled assistants or mobile apps for use by field medics, rural healthcare workers, or doctors in high-paced hospital environments.
# Supports instant, hands-free access to protocols in settings where typing is impractical.
# 
# **5. Implement Continuous Learning and Expert Validation**
# 
# Set up a pipeline for feedback from medical professionals to retrain the system and improve accuracy.
# Include a confidence scoring system to alert users when the AI response requires validation or second opinion.
# 
# **6. Promote Patient Education Tools**
# 
# Use a simplified version of the RAG system to provide AI-driven educational content for patients post-discharge or during teleconsultations (e.g., rehab after brain injury, hair loss treatments).
# 
# **7. Regulatory & Compliance Strategy**
# 
# Ensure adherence to HIPAA, GDPR, and FDA AI/ML medical device guidelines.
# Maintain audit trails, source traceability, and human-in-the-loop supervision for high-risk clinical use cases.

# <font size=6 color='blue'>Power Ahead</font>
# ___
