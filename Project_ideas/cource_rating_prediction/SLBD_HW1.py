import pycaret
import pandas as pd
from pycaret.classification import *
import seaborn as sns
import matplotlib.pyplot as plt

pycaret.__version__

df = pd.read_csv("C:/Users/pontu/Desktop/CHALMERS/PROJECTS_REPO/Project_ideas/cource_rating_prediction/data/numbers.txt", delimiter=" ")  # Use the correct delimiter
#df_train, df_test = pd.train_test_split(df, test_size=0.1, random_state=42)

results = {}

s = setup(df, target = 'V1', session_id = 123)

lr = create_model("lr")
model_results = pull()  # Pulls the last model summary
clean_model_results = model_results.iloc[:-2]
print("Model: Logistic regression")
print(model_results)
results['lr'] = clean_model_results.describe().T[['mean', 'std']]

nb = create_model("nb")
model_results = pull()  # Pulls the last model summary
clean_model_results = model_results.iloc[:-2]
print("Model: Naive Bayes")
print(model_results)
results['nb'] = clean_model_results.describe().T[['mean', 'std']]


knn_flexible = create_model("knn", n_neighbors=3,  return_train_score=True) 
model_results = pull()  # Pulls the last model summary
clean_model_results = model_results.iloc[:-2]
print("Model: KNN_flexible")
print(model_results)
results['knn_flexible'] = clean_model_results.describe().T[['mean', 'std']]


knn_rigid = create_model("knn", n_neighbors=105,  return_train_score=True) 
model_results = pull()  # Pulls the last model summary
clean_model_results = model_results.iloc[:-2]
print("Model: KNN_rigid")
print(model_results)
results['knn_rigid'] = clean_model_results.describe().T[['mean', 'std']]


rf = create_model("rf") # random forest
model_results = pull()  # Pulls the last model summary
clean_model_results = model_results.iloc[:-2]
print("Model: Random forest")
print(model_results)
results['rf'] = clean_model_results.describe().T[['mean', 'std']]


lda = create_model("lda") # linear discriminant analysis
model_results = pull()  # Pulls the last model summary
clean_model_results = model_results.iloc[:-2]
print("Model: LDA")
print(model_results)
results['lda'] = clean_model_results.describe().T[['mean', 'std']]


# Convert dictionary to DataFrame
metrics_df = pd.concat(results, names=['Model', 'Metric'])
metrics_df = metrics_df.reset_index()

# Reshape for plotting
metrics_melted = metrics_df.melt(id_vars=['Model', 'Metric'], var_name='Stat', value_name='Value')
print(metrics_melted)

plt.figure(figsize=(12, 6))
sns.boxplot(x='Metric', y='Value', data=metrics_melted, hue='Model')

plt.xticks(rotation=45)
plt.title("Model Performance Comparison (Mean & Std Dev)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()