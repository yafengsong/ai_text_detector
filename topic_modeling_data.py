import pandas as pd

file_path = '/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/pruned_data.csv'

df = pd.read_csv(file_path)

question_df = df.iloc[:, :1]

# Save the pruned file
subset_file_path = '/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/topic_modeling_data.csv'
question_df.to_csv(subset_file_path, encoding='utf-8', index=False)