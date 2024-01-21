import os
from openai import OpenAI
import pandas as pd
import time

'''
This script generates answers from GPT4 based on the input Quora questions
'''
with open("/Users/songyafeng/6.2/CFL/Assignment/key.txt", "r") as f:
    my_key = f.read()
    os.environ["OPENAI_API_KEY"] = my_key

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

def ask_question(question):

    # to record the time of each iteration
    start_time = time.time()

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        model="gpt-4-1106-preview",
        frequency_penalty=0,
        presence_penalty=0
    )
    return chat_completion.choices[0].message.content

# Read the Quora data from the local file
file_path = '/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/pruned_data.csv'
df = pd.read_csv(file_path)

'''
Prompt: Generate a Quora-like answer based on the question "<question>" with referring to 
        the answer "<answer>" and start the answer with the first 8 words of the reference 
        answer. The generated answer should have similar word amount of the reference answer.
'''
# Use a method to convert the time from seconds to minutes and second
# This is used to record the time consumed for each iteration
def convert_seconds_to_mm_ss(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(minutes):02d}:{seconds:05.2f}"

# Iterate over the questions and get answers
for index, row in df.iterrows():
    start_time = time.time()

    question = ("Generate a Quora-like answer based on the question \""
                + row['question']
                + "\" with referring to the answer \""
                + row['answer']
                + "\" and start the answer with the first 8 words of the reference answer."
                  "The generated answer should have similar word amount of the reference answer.")
    answer = ask_question(question)
    df.at[index, 'gpt4'] = str(answer)

    # Save the DataFrame with gpt4 generated answer as a new csv file
    df.to_csv('/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/gpt4_data.csv', encoding='utf-8',
              index=False)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Iteration {index}: {convert_seconds_to_mm_ss(duration)}")


