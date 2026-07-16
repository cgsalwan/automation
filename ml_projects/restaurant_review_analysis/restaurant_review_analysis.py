"""
Restaurant Review Sentiment Analysis — LLM Prompt Engineering

Builds a restaurant review sentiment analyzer with local LLMs (Llama-2-13B,
Mistral-7B via llama-cpp-python), progressing through five stages: basic
sentiment, structured JSON output, aspect-level sentiment (food/service/
ambience), liked/disliked feature extraction, and automated response drafting.
"""

# # Restaurant Review Sentiment Analysis — LLM Prompt Engineering
# 
# *Building a restaurant review sentiment analyzer with local LLMs (Llama-2, Mistral), progressing from basic sentiment classification to structured, aspect-level sentiment extraction and automated response drafting.*

# <center><p float="center">
#   <img src="https://images.pexels.com/photos/262918/pexels-photo-262918.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" width=720/>
# </p></center>
# 
# <center><font size=6>Restaurant Review Analysis</center></font>

# ## Problem Statement

# ### Business Context

# In the food industry, customer satisfaction plays a pivotal role in shaping the success of individual outlets and the overall brand. A leading global food aggregator is keen on understanding and improving customer experiences across the diverse range of restaurants it lists on its platform. The company recognizes the significance of customer reviews in gaining insights into service quality, food offerings, and overall satisfaction.

# ### Problem Definition

# Despite the abundance of customer reviews available, the company faces significant challenges in deriving actionable insights from these valuable data sources. The manual analysis of extensive amounts of unstructured text data tends to be time-consuming and non-scalable. The key problems to address include:
# 
# - **Unstructured Data Challenge**: Customer reviews are expressed in natural language and an unstructured format, creating difficulties in efficiently extracting meaningful information.
# 
# - **Scale of Data**: With numerous restaurants, the company accumulates a substantial volume of reviews. Manually processing this vast amount of data is not scalable and necessitates an automated approach.
# 
# - **Customer Sentiment Understanding**: Discerning customer sentiments from reviews, whether positive, negative, or neutral, poses a significant challenge. This understanding is crucial for the company to identify the preferences of different customers and devise strategies for targeted marketing.

# ### Objective

# As a data scientist at the company, you have been provided a sample of the customer review data and asked to created a predictive model to analyze the reviews. The objective is to build a robust sentiment analyzer using a Large Language Model (LLM) that accurately predicts the sentiment of customers from the reviews, thereby enhancing the company's ability to understand customer sentiments at scale, enabling data-driven decision-making, and improving overall customer satisfaction.

# ## Installing and Importing Necessary Libraries

# Install dependencies (run once):
# # Installation for GPU llama-cpp-python
# # uncomment and run the following code in case GPU is being used
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.45 --force-reinstall --no-cache-dir -q
# # Installation for CPU llama-cpp-python
# # uncomment and run the following code in case GPU is not being used
# #!CMAKE_ARGS="-DLLAMA_CUBLAS=off" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.45 --force-reinstall --no-cache-dir -q

# **Note**: pip's dependency error can be ignored as it does not affect further execution.

# Install dependencies (run once):
# # For downloading the models from HF Hub
# pip install huggingface_hub==0.20.3 -q

# Importing library for data manipulation
import pandas as pd

# Function to download the model from the Hugging Face model hub
from huggingface_hub import hf_hub_download

# Importing the Llama class from the llama_cpp module
from llama_cpp import Llama

# Importing the json module
import json

# ## Import the dataset

# Colab-specific data mount, not needed outside Colab:
# from google.colab import drive
# drive.mount('/content/drive')

data = pd.read_csv("restaurant_reviews.csv")

# ## Data Overview

# checking the first five rows of the data
data.head()

# checking the shape of the data
data.shape

# **Observations**
# 
# - Data has 20 rows and 3 columns

# checking for missing values
data.isnull().sum()

# **Observations**
# 
# - There are no missing values in the data

# ## Model Building

# ### Loading the model (Llama)

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
model_basename = "llama-2-13b-chat.Q5_K_M.gguf" # the model is in gguf format

# Using hf_hub_download to download a model from the Hugging Face model hub
# The repo_id parameter specifies the model name or path in the Hugging Face repository
# The filename parameter specifies the name of the file to download
model_path = hf_hub_download(
    repo_id=model_name_or_path,
    filename=model_basename
)

lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2,  # CPU cores
    n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    # n_gpu_layers=43,  # uncomment and change this value based on GPU VRAM pool.
    n_ctx=4096,  # Context window
)

# ### Loading the model (Mistral)

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"

model_path = hf_hub_download(
    repo_id=model_name_or_path,
    filename=model_basename
)

llm = Llama(
    model_path=model_path,
    n_ctx=1024,
)

# ### Defining Model Response Parameters

def generate_llama_response(instruction, review):

    # System message explicitly instructing not to include the review text
    system_message = """
        [INST]<<SYS>>
        {}
        <</SYS>>[/INST]
    """.format(instruction)

    # Combine user_prompt and system_message to create the prompt
    prompt = f"{review}\n{system_message}"

    # Generate a response from the LLaMA model
    response = lcpp_llm(
        prompt=prompt,
        max_tokens=1024,
        temperature=0,
        top_p=0.95,
        repeat_penalty=1.2,
        top_k=50,
        stop=['INST'],
        echo=False,
        seed=42,
    )

    # Extract the sentiment from the response
    response_text = response["choices"][0]["text"]
    return response_text

# - **`max_tokens`**: This parameter **specifies the maximum number of tokens that the model should generate** in response to the prompt.
# 
# - **`temperature`**: This parameter **controls the randomness of the generated response**. A higher temperature value will result in a more random response, while a lower temperature value will result in a more predictable response.
# 
# - **`top_p`**: This parameter **controls the diversity of the generated response by establishing a cumulative probability cutoff for token selection**. A higher value of top_p will result in a more diverse response, while a lower value will result in a less diverse response.
# 
# - **`repeat_penalty`**: This parameter **controls the penalty for repeating tokens in the generated response**. A higher value of repeat_penalty will result in a lower probability of repeating tokens, while a lower value will result in a higher probability of repeating tokens.
# 
# - **`top_k`**: This parameter **controls the maximum number of most-likely next tokens to consider** when generating the response at each step.
# 
# - **`stop`**: This parameter is a **list of tokens that are used to dynamically stop response generation** whenever the tokens in the list are encountered.
# 
# - **`echo`**: This parameter **controls whether the input (prompt) to the model should be returned** in the model response.
# 
# - **`seed`**: This parameter **specifies a seed value that helps replicate results**.

# ### Utility function

# defining a function to parse the JSON output from the model
def extract_json_data(json_str):
    try:
        # Find the indices of the opening and closing curly braces
        json_start = json_str.find('{')
        json_end = json_str.rfind('}')

        if json_start != -1 and json_end != -1:
            extracted_sentiment = json_str[json_start:json_end + 1]  # Extract the JSON object
            data_dict = json.loads(extracted_sentiment)
            return data_dict
        else:
            print(f"Warning: JSON object not found in response: {json_str}")
            return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}

# ## 1. Sentiment Analysis (Llama)

# creating a copy of the data
data_1 = data.copy()

# defining the instructions for the model
instruction_1 = """
    You are an AI analyzing restaurant reviews. Classify the sentiment of the provided review into the following categories:
    - Positive
    - Negative
    - Neutral
"""

data_1['model_response'] = data_1['review_full'].apply(lambda x: generate_llama_response(instruction_1, x))

data_1['model_response'].head()

i = 2
print(data_1.loc[i, 'review_full'])

print(data_1.loc[i, 'model_response'])

def extract_sentiment(model_response):
    if 'positive' in model_response.lower():
        return 'Positive'
    elif 'negative' in model_response.lower():
        return 'Negative'
    elif 'neutral' in model_response.lower():
        return 'Neutral'

# applying the function to the model response
data_1['sentiment'] = data_1['model_response'].apply(extract_sentiment)
data_1['sentiment'].head()

data_1['sentiment'].value_counts()

final_data_1 = data_1.drop(['model_response'], axis=1)
final_data_1.head()

# ## 1. Sentiment Analysis (Mistral)

# creating a copy of the data
data_1 = data.copy()

# **We are going to use an instruction-tuned Mistral model. Hence, the format of the input to the model varies from that of Llama.**

#Defining the response funciton for Task 1.
def response_1(prompt,review):
    model_output = llm(
      f"""
      Q: {prompt}
      Review: {review}
      A:
      """,
      max_tokens=32,
      stop=["Q:", "\n"],
      temperature=0.01,
      echo=False,
    )

    temp_output = model_output["choices"][0]["text"]

    return temp_output

# defining the instructions for the model
instruction_1 = """
    You are an AI analyzing restaurant reviews. Classify the sentiment of the provided review into the following categories:
    - Positive
    - Negative
    - Neutral
"""

data_1['model_response'] = data_1['review_full'].apply(lambda x: response_1(instruction_1, x))

data_1['model_response'].head()

i = 2
print(data_1.loc[i, 'review_full'])

print(data_1.loc[i, 'model_response'])

def extract_sentiment(model_response):
    if 'positive' in model_response.lower():
        return 'Positive'
    elif 'negative' in model_response.lower():
        return 'Negative'
    elif 'neutral' in model_response.lower():
        return 'Neutral'

# applying the function to the model response
data_1['sentiment'] = data_1['model_response'].apply(extract_sentiment)
data_1['sentiment'].head()

data_1['sentiment'].value_counts()

final_data_1 = data_1.drop(['model_response'], axis=1)
final_data_1.head()

# ## 2. Sentiment Analysis and Returning Structured Output (Llama)

# creating a copy of the data
data_2 = data.copy()

# defining the instructions for the model
instruction_2 = """
    You are an AI analyzing restaurant reviews. Classify the sentiment of the provided review into the following categories:
    - Positive
    - Negative
    - Neutral

    Format the output as a JSON object with a single key-value pair as shown below:
    {"sentiment": "your_sentiment_prediction"}
"""

data_2['model_response'] = data_2['review_full'].apply(lambda x: generate_llama_response(instruction_2, x))

data_2['model_response'].head()

i = 2
print(data_2.loc[i, 'review_full'])

print(data_2.loc[i, 'model_response'])

# applying the function to the model response
data_2['model_response_parsed'] = data_2['model_response'].apply(extract_json_data)
data_2['model_response_parsed'].head()

model_response_parsed_df_2 = pd.json_normalize(data_2['model_response_parsed'])
model_response_parsed_df_2.head()

data_with_parsed_model_output_2 = pd.concat([data_2, model_response_parsed_df_2], axis=1)
data_with_parsed_model_output_2.head()

final_data_2 = data_with_parsed_model_output_2.drop(['model_response','model_response_parsed'], axis=1)
final_data_2.head()

final_data_2['sentiment'].value_counts()

# ## 3. Identifying Overall Sentiment and Sentiment of Aspects of the Experience (Llama)

# creating a copy of the data
data_3 = data.copy()

# defining the instructions for the model
instruction_3 = """
    You are an AI analyzing restaurant reviews. Classify the overall sentiment of the provided review into the following categories:
    - "Positive"
    - "Negative"
    - "Neutral"

    Once that is done, check for a mention of the following aspects in the review and classify the sentiment of each aspect as "Positive", "Negative", or "Neutral":
    1. "Food Quality"
    2. "Service"
    3. "Ambience"

    Output the overall sentiment and sentiment for each category in a JSON format with the following keys:
    {
        "Overall": "your_sentiment_prediction",
        "Food Quality": "your_sentiment_prediction",
        "Service": "your_sentiment_prediction",
        "Ambience": "your_sentiment_prediction"
    }

    In case one of the three aspects is not mentioned in the review, set "Not Applicable" (including quotes) for the corresponding JSON key value.
    Only return the JSON, do not return any other information.
"""

data_3['model_response'] = data_3['review_full'].apply(lambda x: generate_llama_response(instruction_3, x))

data_3['model_response'].head()

i = 2
print(data_3.loc[i, 'review_full'])

print(data_3.loc[i, 'model_response'])

# applying the function to the model response
data_3['model_response_parsed'] = data_3['model_response'].apply(extract_json_data)
data_3['model_response_parsed'].head()

model_response_parsed_df_3 = pd.json_normalize(data_3['model_response_parsed'])
model_response_parsed_df_3.head()

data_with_parsed_model_output_3 = pd.concat([data_3, model_response_parsed_df_3], axis=1)
data_with_parsed_model_output_3.head()

final_data_3 = data_with_parsed_model_output_3.drop(['model_response','model_response_parsed'], axis=1)
final_data_3.head()

final_data_3['Overall'].value_counts()

final_data_3['Food Quality'].value_counts()

final_data_3['Service'].value_counts()

final_data_3['Ambience'].value_counts()

# ## 3. Identifying Overall Sentiment and Sentiment of Aspects of the Experience (Mistral)

# creating a copy of the data
data_3 = data.copy()

def response_2(prompt,review,sentiment):
    model_output = llm(
      f"""
      Q: {prompt}
      review: {review}
      sentiment: {sentiment}
      A:
      """,
      max_tokens=64,
      stop=["Q:", "\n"],
      temperature=0.01,
      echo=False,
    )

    temp_output = model_output["choices"][0]["text"]
    final_output = temp_output[temp_output.index('{'):]

    return final_output

# **Note:** We have already predicted the sentiment of the review. We can use this information while designing the prompt for this task. This way, it will reduce the computational complexity.
# 
# The sentiment is stored in the 'final_data_1' dataframe which is from the TASK 1.

# defining the instructions for the model
instruction_3 = """
    You are provided a review and it's sentiment.

    Instructions:
    Classify the sentiment of each aspect as either of "Positive", "Negative", or "Neutral" only and not any other for the given review:
    1. "Food Quality"
    2. "Service"
    3. "Ambience"
    In case one of the three aspects is not mentioned in the review, return "Not Applicable" (including quotes) for the corresponding JSON key value.
    Return the output in the format {"Overall": given sentiment input,"Food Quality": "your_sentiment_prediction","Service": "your_sentiment_prediction","Ambience": "your_sentiment_prediction"}

"""

data_3['model_response'] = final_data_1[['review_full','sentiment']].apply(lambda x: response_2(instruction_3, x[0],x[1]),axis=1)

data_3['model_response'].values

i = 2
print(data_3.loc[i, 'review_full'])

print(data_3.loc[i, 'model_response'])

# applying the function to the model response
data_3['model_response_parsed'] = data_3['model_response'].apply(extract_json_data)
data_3['model_response_parsed']

model_response_parsed_df_3 = pd.json_normalize(data_3['model_response_parsed'])
model_response_parsed_df_3

model_response_parsed_df_3 = model_response_parsed_df_3.apply(lambda x: x.astype(str).str.lower())

data_with_parsed_model_output_3 = pd.concat([data_3, model_response_parsed_df_3], axis=1)
data_with_parsed_model_output_3.head()

final_data_3 = data_with_parsed_model_output_3.drop(['model_response','model_response_parsed'], axis=1)
final_data_3.head()

final_data_3['Overall'].value_counts()

final_data_3['Food Quality'].value_counts()

# **Note:** One of the sentiment is 'if not exceptional'. This is most likely positive.

final_data_3['Service'].value_counts()

final_data_3['Ambience'].value_counts()

# ## 4. Identifying Overall Sentiment, Sentiment of Aspects of the Experience, and the Liked/Disliked Features of the Different Aspects of the Experience (Llama)

# creating a copy of the data
data_4 = data.copy()

# defining the instructions for the model
instruction_4 = """
    You are an AI tasked with analyzing restaurant reviews. Your goal is to classify the overall sentiment of the provided review into the following categories:
        - Positive
        - Negative
        - Neutral

    Subsequently, assess the sentiment of specific aspects mentioned in the review, namely:
        1. Food quality
        2. Service
        3. Ambience

    Further, identify liked and/or disliked features associated with each aspect in the review.

    Return the output in the specified JSON format, ensuring consistency and handling missing values appropriately:

    {
        "Overall": "your_sentiment_prediction",
        "Food Quality": "your_sentiment_prediction",
        "Service": "your_sentiment_prediction",
        "Ambience": "your_sentiment_prediction",
        "Food Quality Features": ["liked/disliked features"],
        "Service Features": ["liked/disliked features"],
        "Ambience Features": ["liked/disliked features"]
    }

    The sentiment prediction for Overall, Food Quality, Service, and Ambience should be one of "Positive", "Negative", or "Neutral" only.
    In case one of the three aspects is not mentioned in the review, set "Not Applicable" (including quotes) in the corresponding JSON key value for the sentiment.
    In case there are no liked/disliked features for a particular aspect, assign an empty list in the corresponding JSON key value for the aspect.
    Only return the JSON, do NOT return any other text or information.
"""

data_4['model_response'] = data_4['review_full'].apply(lambda x: generate_llama_response(instruction_4, x).replace('\n', ''))

i = 2
print(data_4.loc[i, 'review_full'])

print(data_4.loc[i, 'model_response'])

# applying the function to the model response
data_4['model_response_parsed'] = data_4['model_response'].apply(extract_json_data)
data_4['model_response_parsed'].head()

data_4[data_4.model_response_parsed == {}]

# - There are three model responses that the JSON parser function could not parse
# - We'll manually add the values for these three responses

print(data_4.loc[3, 'model_response'])

print(data_4.loc[6, 'model_response'])

print(data_4.loc[7, 'model_response'])

upd_val_1 = {
    "Overall": "Positive",
    "Food Quality": "Positive",
    "Service": "Positive",
    "Ambience": "Not Applicable",
    "Food Quality Features": [],
    "Service Features": ["excellent service"],
    "Ambience Features": []
}

upd_val_2 = {
    "Overall": "Neutral",
    "Food Quality": "Neutral",
    "Service": "Neutral",
    "Ambience": "Not Applicable",
    "Food Quality Features": ["well prepared"],
    "Service Features": ["slow and inattentive"],
    "Ambience Features": ["interior is friendly", "not intimidating"]
}

upd_val_3 = {
    "Overall": "Neutral",
    "Food Quality": "Positive",
    "Service": "Negative",
    "Ambience": "Positive",
    "Food Quality Features": ["Some tasty, others average"],
    "Service Features": ["Attentive staff", "Slow service"],
    "Ambience Features": []
}

# defining the list of indices to update
idx_list = [3,6,7]
data_4.loc[idx_list, 'model_response_parsed'] = [upd_val_1, upd_val_2, upd_val_3]

# **Note**: The values model responses that cannot be parsed correctly by the JSON parser function may vary with execution due to the randomness associated with LLMs. Kindly update as observed when run in your system.

model_response_parsed_df_4 = pd.json_normalize(data_4['model_response_parsed'])
model_response_parsed_df_4.head()

data_with_parsed_model_output_4 = pd.concat([data_4, model_response_parsed_df_4], axis=1)
data_with_parsed_model_output_4.head()

final_data_4 = data_with_parsed_model_output_4.drop(['model_response','model_response_parsed'], axis=1)
final_data_4.head()

final_data_4['Overall'].value_counts()

final_data_4['Food Quality'].value_counts()

final_data_4['Service'].value_counts()

final_data_4['Ambience'].value_counts()

# ## 5. Identifying Overall Sentiment, Sentiment of Aspects of the Experience, Liked/Disliked Features of the Different Aspects of the Experience, and Sharing a Response (Llama)

# creating a copy of the data
data_5 = data.copy()

# defining the instructions for the model
instruction_5 = """
    You are an AI analyzing restaurant reviews. Classify the overall sentiment of the provided review into the following categories:
    - "Positive"
    - "Negative"
    - "Neutral"

    Once that is done, check for a mention of the following aspects in the review and clasify the sentiment of each aspect as positive, negative, or neutral:
    1. Food quality
    2. Service
    3. Ambience

    Once that is done, look for liked and/or disliked features mentioned against each of the above aspects in the review and extract them.

    Finally, draft a response for the customer based on the review. Start out with a thank you note and then add on to it as per the following:
    1. If the review is positive, mention that it would be great to have them again
    2. If the review is neutral, ask them for what the restaurant could have done better
    3. If the review is negative, apologive for the inconvenience and mention that we'll be looking into the points raised

    Return the output in the specified JSON format, ensuring consistency and handling missing values appropriately Ensure that all values in the JSON are formatted as strings, and each element within the lists should be enclosed in double quotes:

    {
        "Overall": "your_sentiment_prediction",
        "Food Quality": "your_sentiment_prediction",
        "Service": "your_sentiment_prediction",
        "Ambience": "your_sentiment_prediction",
        "Food Quality Features": ["liked/disliked features"],
        "Service Features": ["liked/disliked features"],
        "Ambience Features": ["liked/disliked features"],
        "Response": "your_response_to_the_customer_review",
    }

    The sentiment prediction for Overall, Food Quality, Service, and Ambience should be one of "Positive", "Negative", or "Neutral" only.
    In case one of the three aspects is not mentioned in the review, set "Not Applicable" (including quotes) in the corresponding JSON key value for the sentiment.
    In case there are no liked/disliked features for a particular aspect, assign an empty list in the corresponding JSON key value for the aspect.
    Be polite and empathetic in the response to the customer review.
    Only return the JSON, do NOT return any other text or information.
"""

data_5['model_response'] = data_5['review_full'].apply(lambda x: generate_llama_response(instruction_5, x))

i = 2
print(data_5.loc[i, 'review_full'])

print(data_5.loc[i, 'model_response'])

# applying the function to the model response
data_5['model_response_parsed'] = data_5['model_response'].apply(extract_json_data)
data_5['model_response_parsed'].head()

model_response_parsed_df_5 = pd.json_normalize(data_5['model_response_parsed'])
model_response_parsed_df_5.head()

data_with_parsed_model_output_5 = pd.concat([data_5, model_response_parsed_df_5], axis=1)
data_with_parsed_model_output_5.head()

final_data_5 = data_with_parsed_model_output_5.drop(['model_response','model_response_parsed'], axis=1)
final_data_5.head()

final_data_5['Overall'].value_counts()

final_data_5['Food Quality'].value_counts()

final_data_5['Service'].value_counts()

final_data_5['Ambience'].value_counts()

# ## Conclusions

# - We used an LLM to do multiple tasks, one stage at a time
#     1. We first identified the overall sentiment of the review using the LLM
#     2. We then identified the overall sentiment of the review and got the output in a structured format from the LLM for ease-of-access
#     3. Next, we identified the overall sentiment of the review as well as sentiment of specific aspects of the experience
#     4. Next, in addition to the overall sentiment of the review as well as sentiment of specific aspects of the experience, we also identified the liked/disliked features of the different aspects of the experience
#     5. Finally, in addition to all the above, we also got a response we can share with the customer based on their review
# 
# - One can manually label the data (overall sentiment and sentiments of different aspects) and then compare the model's output with the same to get a quantitative measure of the models performance.
# 
# - To try and improve the model performance, one can try the following:
#     1. Update the prompt
#     2. Update the model parameters (`temparature`, `top_p`, ...)

# ___
