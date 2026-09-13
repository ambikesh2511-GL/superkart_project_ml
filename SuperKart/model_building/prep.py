import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("SuperKart/data/SuperKart.csv")

# Drop ID columns that are not useful for prediction
df.drop(columns=["Product_Id", "Store_Id"], inplace=True)

# Define features and target
X = df.drop(columns=["Product_Store_Sales_Total"])
y = df["Product_Store_Sales_Total"]

# Train/Test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Save datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared successfully.")
print(f"Training records: {len(Xtrain)}")
print(f"Testing records: {len(Xtest)}")

print("\nCategorical columns:")
print(X.select_dtypes(include=["object"]).columns.tolist())

print("\nNumerical columns:")
print(X.select_dtypes(exclude=["object"]).columns.tolist())
