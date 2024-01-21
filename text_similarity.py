import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Read Dataframe
file_path = '/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/gpt4_data.csv'
df = pd.read_csv(file_path)
# df.at[2, 'gpt4'] = 'NaN'  # For row 3 (index 2)
# df.at[3, 'gpt4'] = 'NaN'  # For row 4 (index 3)

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Combine the texts to vectorize them together
all_texts = pd.concat([df['answer'], df['gpt4']], axis=0)

# Fit and transform the vectorizer on all texts
vectorizer.fit(all_texts)
question_vectors = vectorizer.transform(df['answer'])
answer_vectors = vectorizer.transform(df['gpt4'])

# Calculate cosine similarity and store it in the DataFrame
df['similarity'] = [cosine_similarity(question_vectors[i], answer_vectors[i])[0][0] for i in range(len(df))]
mean_similarity = df['similarity'].mean()
std_similarity = df['similarity'].std()

print(mean_similarity, std_similarity)

# Plotting the cosine similarity scores
plt.figure(figsize=(6, 6))
plt.boxplot(df['similarity'])
plt.title('Figure 2: Box Plot of Cosine Similarity Scores', fontsize=18)
plt.ylabel('Cosine Similarity', fontsize=16)
plt.xticks([1], ['Similarity'], fontsize=14)
plt.yticks(fontsize=14)

plt.show()