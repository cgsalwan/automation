"""
Python Fundamentals for Prompt Engineering

Covers storing/editing multi-line text, reading structured data
(JSON, CSV, plain text) into Python, and basic text cleaning -
foundational skills for prompt engineering and NLP workflows.
"""

# ## Python Fundamentals for Prompt Engineering

# ### What this covers
# 1. Storing and editing multi-line text in Python (the building block of prompts)
# 2. Reading structured data (JSON, CSV, plain text) into Python
# 3. Basic text cleaning for downstream NLP / prompt-engineering work

# ### How to store text in python and edit it?

# Before you do prompt engineering in python, you need to learn how to save text in a variable and how to manipulate it
# 
# Prompts are multi-line texts, and we can use triple quotes in python to store text

text = f"""
In a world where Generative AI thrived, creativity knew no bounds. It painted symphonies,
sculpted dreams, and penned tales beyond imagination. Yet, amidst the brilliance,
it yearned to understand the one thing it couldn't create - the human heart.
"""

print(text)

# Sometimes, you might need to insert another piece of text or a word into an existing text. You can do this by using placeholder
# 
# Say you need insert the below text as a new paragraph in the original text:
# 
# **"But as they strove for perfection, they discovered that true beauty lay in the imperfections of the human spirit."**

additional_text = f"""
But as they strove for perfection, they discovered that true beauty lay
in the imperfections of the human spirit.
"""

# There are many ways to do this, but let's explore how to do this with placeholders as they are relevant to the context of learning to do prompt engineering on python
# 
# Let's modify the original text variable to add a placeholder -

text = f"""
In a world where Generative AI thrived, creativity knew no bounds. It painted symphonies,
sculpted dreams, and penned tales beyond imagination. Yet, amidst the brilliance,
it yearned to understand the one thing it couldn't create - the human heart.

{additional_text}
"""

print(text)

# ### How to read data from a file into python?

# In order to be abe to authenticate our identity and programmatically access the API from Google Colab, you will need to read your credentials into the Colab environment in a secure way. To do this, you need an understanding of how to read a file into python

# To learn how to do this, let's first create a JSON file

with open('sample.json', 'w') as f:
    f.write('''
{
  "key_1":"HU09KJK",
  "key_2":"HJXTPEE"
}''')

# If you now look into your google colab file browser, you should have a JSON file. Now, if this was a set of credentials, how do you read this into python?
# 
# The first thing you need is a python package to parse a JSON file

import json

# Now you use the `open` method in python to read and file and then the `json` package to extract what you need from the file

with open('sample.json', 'r') as my_credentials:
    data = my_credentials.read()

creds = json.loads(data) # this will load your data into a dictionary

print(creds)
print(type(creds))

creds['key_1']

#  Add-on : Reading and Storing Text Data

# Sample code to read from 'customer_feedback.txt'
with open('customer_feedback.txt', 'r') as file:
    feedback_data = file.read()
print(feedback_data)

# ## Text Data Cleaning

## Data Exploration

import pandas as pd

# Load the dataset
dataset = pd.read_csv('sample_dataset.csv')

# Display the first few rows to understand the structure
print(dataset.head())

# Convert the comments to lower case
dataset['comments_lower'] = dataset['comments'].str.lower()

## Text Data Cleaning

# Remove punctuation from the comments
import re
dataset['comments_clean'] = dataset['comments_lower'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Display the cleaned data
print("\nData after cleaning:")
print(dataset[['comments', 'comments_clean']].head())

# 
# 
# ---
