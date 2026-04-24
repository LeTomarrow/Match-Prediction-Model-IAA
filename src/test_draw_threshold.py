import pandas as pd
import numpy as np

df = pd.read_csv('/home/joao/Desktop/IAA/Match-Prediction-Model-IAA/data/predictions/xgboost_test_predictions.csv')
print(f"Total test rows: {len(df)}")
true = df['true_result'].values

# Currently we just have the class predictions. We need the probabilities.
# I will modify train_goal_models to save probabilities, but let's first check if we can reach 60% with an oracle.
oracle_acc = (df['true_result'] == df['pred_result']).mean()
print(f"Current accuracy: {oracle_acc:.4f}")

# What if we knew when to predict a draw?
# Let's say we have a magic oracle that tells us it's a draw, but we still use the model for Home/Away.
df['magic_pred'] = np.where(df['true_result'] == 0, 0, df['pred_result'])
magic_acc = (df['true_result'] == df['magic_pred']).mean()
print(f"Accuracy if we got every draw right: {magic_acc:.4f} (This is the theoretical max)")

# Wait, if we got every draw right, accuracy is only 63.6%?
