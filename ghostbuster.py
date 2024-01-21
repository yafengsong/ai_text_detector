import pandas as pd
from sklearn.metrics import roc_auc_score

file_path = '/Users/songyafeng/6.2/CFL/Assignment/AI-generated-text-detection/ghostbuster/classification_results.csv'
df = pd.read_csv(file_path)

df['label'] = df['filename'].str.extract(r'_(\d+)')

df['predicted_label'] = (df['prediction_score'] > 0.5).astype(int)
print(df.head(3))
auroc = roc_auc_score(df['label'], df['predicted_label'])

print(f"AUROC: {auroc}")