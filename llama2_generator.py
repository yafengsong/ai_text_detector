import replicate
import pandas as pd
import time
import os

# Load key for replicate
with open("/Users/songyafeng/6.2/CFL/Assignment/key_replicate.txt", "r") as f:
    my_key = f.read()
    os.environ["REPLICATE_API_TOKEN"] = my_key

api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

# Read the Quora data from the local file
file_path = '/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/pruned_data.csv'
df = pd.read_csv(file_path)


def ask_llama2(prompt):
    output = api.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={
            "debug": False,
            "top_k": 10,
            "top_p": 1,
            "prompt": prompt,
            "temperature": 0.9,
            "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
            "max_new_tokens": 5000,
            "min_new_tokens": -1
        }
    )
    return output

# Use a method to convert the time from seconds to minutes and second
# This is used to record the time consumed for each iteration
def convert_seconds_to_mm_ss(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(minutes):02d}:{seconds:05.2f}"

# Iterate over the questions and get answers
for index, row in df.iterrows():
    start_time = time.time()

    prompt = ("Generate a Quora-like answer based on the question \""
                + row['question']
                + "\" with referring to the answer \""
                + row['answer']
                + "\" and start the answer with the first 8 words of the reference answer."
                  "The generated answer should have similar word amount of the reference answer.")
    answer = ask_llama2(prompt)
    combined_answer = ''.join(answer)
    df.at[index, 'gpt4'] = str(combined_answer)

    # Save the DataFrame with gpt4 generated answer as a new csv file
    df.to_csv('/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/llama2_data.csv', encoding='utf-8',
              index=False)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Iteration {index}: {convert_seconds_to_mm_ss(duration)}")
