import pandas as pd

file_path = '/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/gpt4_data.csv'
df = pd.read_csv(file_path)
pd.set_option('display.max_colwidth', None)  # Set the maximum width of the column to a large number

# Replace "\n" with an empty string in the 'gpt4' column
df['gpt4'] = df['gpt4'].str.replace("\n", "", regex=False)


# Create two separate DataFrames for each type of answer
df_quora = df[['answer']].dropna().rename(columns={'answer': 'text'})
df_quora['label'] = 0

df_gpt4 = df[['gpt4']].dropna().rename(columns={'gpt4': 'text'})
df_gpt4['label'] = 1

# Combine the two DataFrames
combined_df = pd.concat([df_quora, df_gpt4], ignore_index=True)

# Shuffle the DataFrame
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# save the processed data
combined_df.to_csv('/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/gpt4_data_processed.csv', encoding='utf-8',
              index=False)
print(len(combined_df))

