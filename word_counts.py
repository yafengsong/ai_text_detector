import pandas as pd
import matplotlib.pyplot as plt

file_path = '/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/gpt4_data.csv'
df = pd.read_csv(file_path)

# Function to count the number of words in each text
def word_count(text):
    return len(text.split())

# Function to count the number of words in each text
def word_count(text):
    return len(text.split())

# Apply the word_count function to each row of both columns
df['answer_word_count'] = df['answer'].apply(word_count)
df['gpt4_word_count'] = df['gpt4'].apply(word_count)

# Calculate average word count for both columns
average_answer_word_count = df['answer_word_count'].mean()
average_gpt4_word_count = df['gpt4_word_count'].mean()
total_average_word_count = (average_answer_word_count + average_gpt4_word_count) / 2

# Print the average word counts
print("Average Word Count for 'answer' column:", average_answer_word_count)
print("Average Word Count for 'gpt4' column:", average_gpt4_word_count)
print("Total Average Word Count:", total_average_word_count)

# Generating a box plot of each column's word counts
plt.figure(figsize=(8, 6))
df[['answer_word_count', 'gpt4_word_count']].boxplot()
plt.title('Figure 1: Box Plot of Word Counts', fontsize=18)
plt.ylabel('Word Count', fontsize=16)
plt.xticks([1, 2], ['Human Answer', 'GPT-4'], fontsize=14)
plt.yticks(fontsize=14)
plt.show()