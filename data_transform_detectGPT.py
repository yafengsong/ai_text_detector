import pandas as pd

file_path = '/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/gpt4_data.csv'
df = pd.read_csv(file_path)

new_df = df[['answer', 'gpt4']].copy()
new_df = new_df.iloc[0:100]
new_df = new_df.rename(columns={'answer': 'original', 'gpt4': 'sampled'})

data = new_df.to_dict(orient='list')
print(len(data))