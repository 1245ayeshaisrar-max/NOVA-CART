import pandas as pd
import numpy as np

# 1. Load and clean your specific dataset
df = pd.read_csv('product_ratings.csv')
df = df.dropna(axis=1, how='all')
df = df[df['Product Name'].astype(str).str.lower() != 'product name'].reset_index(drop=True)

# 2. Collaborative Filtering requires User Interaction Data
# We are creating simulated user ratings (1-5) for your products
products = df['Product Name'].unique()
np.random.seed(42)
num_users = 100
user_ids = np.random.randint(1, num_users + 1, size=2000)
product_names = np.random.choice(products, size=2000)
ratings = np.random.randint(1, 6, size=2000)

ratings_df = pd.DataFrame({'User_ID': user_ids, 'Product': product_names, 'Rating': ratings})
ratings_df = ratings_df.drop_duplicates(['User_ID', 'Product'])

# 3. Create the User-Item Matrix
pivot_table = ratings_df.pivot(index='Product', columns='User_ID', values='Rating').fillna(0)

# 4. Calculate Similarity between products (Cosine Similarity)
matrix = pivot_table.values
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
norms[norms == 0] = 1e-9
normalized_matrix = matrix / norms
similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
sim_df = pd.DataFrame(similarity_matrix, index=pivot_table.index, columns=pivot_table.index)

# 5. Get User Input and Show Output
print("Available Products:", products[:10])
user_liked = input("\nEnter a product you like: ").strip()

if user_liked in sim_df.index:
    recs = sim_df[user_liked].sort_values(ascending=False).iloc[1:6]
    print(f"\nRecommended for you (Based on user behavior):")
    for name, score in recs.items():
        category = df[df['Product Name'] == name]['CATEGORY'].values[0]
        print(f"- {name} (Category: {category}) | Match: {int(score*100)}%")
else:
    print("Product not found. Make sure the spelling matches the catalog.")
