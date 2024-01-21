from datasets import load_dataset
import pandas as pd
import re
import matplotlib.pyplot as plt


# Load the data from huggingface
dataset = load_dataset("toughdata/quora-question-answer-dataset")
train_dataset = dataset['train']
train_df = pd.DataFrame(train_dataset)
train_df['word_count'] = train_df['answer'].apply(lambda x: len(x.split()))
pd.set_option('display.max_colwidth', None)  # Set the maximum width of the column to a large number

# filter the human-written answers with words between [300, 1000]
filtered_df = train_df[(train_df['word_count'] > 300) & (train_df['word_count'] < 1000)]

'''
Remove: URLs
 - find patterns  [LINKED_TEXT: Udaan (2010)] [URL: http://www.imdb.com/title/tt1639426/?ref_=nv_sr_1]
 - remove the [URL:xx] and [LINKED_TEXT:]
 - keep words after "TEXT:" because they contain useful information (hyper links in Quora)
'''

pattern1 = r"\[URL: [^\]]+\]"  # pattern for URL
pattern2 = r'\[LINKED_TEXT: (.*?)\]' # pattern for LINKED_TEXT

filtered_df['answer'] = filtered_df['answer'].str.replace(pattern1, "", regex=True)
filtered_df['answer'] = filtered_df['answer'].apply(lambda x: re.sub(pattern2, r'\1', x))

# Replace '\n' with empty string
filtered_df['answer'] = filtered_df['answer'].str.replace("\n", "", regex=False)

# Remove the data contain [math]..[\math]
filtered_df = filtered_df[~filtered_df['answer'].str.contains("\[math\].*?\[/math\]", regex=True)]

# Get basic information of the data
num_rows = len(filtered_df)
# Sample 10 random cells from the filtered_df dataframe
random_answers = filtered_df['answer'].sample(10)
print(random_answers)
print(filtered_df.describe())
print("Number of rows in filtered_df:", num_rows)

# Visualize the distribution of word counts in answers
plt.figure(figsize=(10, 6))
plt.hist(filtered_df['word_count'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Word Count in Answers (Word Count > 300)')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Save the processed data on local folder
filtered_df.to_csv('/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/processed_data.csv', encoding='utf-8',
                   index=False)
