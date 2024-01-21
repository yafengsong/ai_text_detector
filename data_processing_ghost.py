import pandas as pd
'''
Transform each answer from csv file to a text file
Save the file name as the {index}_{label}.txt
'''
file_path = '/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/gpt4_data_processed.csv'
df = pd.read_csv(file_path)

for index, row in df.iterrows():
    # Extract text and label
    text = row['text']
    label = row['label']

    # Create filename
    filename = f"{index}_{label}.txt"
    file_dir = '/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/ghostbuster/'

    # Save the text to a file
    with open(file_dir+filename, 'w') as file:
        file.write(text)