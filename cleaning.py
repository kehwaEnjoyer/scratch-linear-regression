import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#this script is to specifically clean bostonhousing.csv if you intend to run some other dataset you 
#will have to clean and scale and format it seperatly 

# Load and clean data
df = pd.read_csv("BostonHousing.csv")

# Remove duplicates
df = df.drop_duplicates()

#Fill missing numeric values with median
df = df.fillna(df.median(numeric_only=True))

# Separate features and target
target_column = "medv"
X = df.drop(columns=[target_column])
y = df[target_column]

cols = X.columns
binary_col = cols[3]  # 4th column
X_binary = X[[binary_col]]
X_rest = X.drop(columns=[binary_col])

# Split into train/test random
X_train, X_test, y_train, y_test = train_test_split(
    X_rest, y, test_size=0.2
)

Xb_train = X_binary.loc[X_train.index]
Xb_test = X_binary.loc[X_test.index]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Add 4 column back
X_train_final = pd.concat([X_train_scaled, Xb_train.reset_index(drop=True)], axis=1)[cols]
X_test_final = pd.concat([X_test_scaled, Xb_test.reset_index(drop=True)], axis=1)[cols]

# rearrange with predictions
train_df = pd.concat([X_train_final, y_train.reset_index(drop=True)], axis=1)
test_df = pd.concat([X_test_final, y_test.reset_index(drop=True)], axis=1)

# Save to CSV without col names
train_df.to_csv("train.csv", index=False, header=False)
test_df.to_csv("test.csv", index=False, header=False)

print("Train and test CSV files saved successfully")