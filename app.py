import pandas as pd
import numpy as np

# 1. LOAD AND CLEAN YOUR DATASET
df = pd.read_csv('product_ratings.csv')
df = df.dropna(axis=1, how='all') # Remove empty columns
df = df[df['Product Name'].astype(str).str.lower() != 'product name'].reset_index(drop=True)

# 2. SIMULATE USER RATINGS (Collaborative Filtering needs User IDs)
# In a real system, you would load a file with: User_ID, Product Name, Rating
products = df['Product Name'].unique()
np.random.seed(42)
num_users = 50
dummy_interactions = {
    'User_ID': np.random.randint(1, num_users + 1, size=1000),
    'Product Name': np.random.choice(products, size=1000),
    'Rating': np.random.randint(1, 6, size=1000)
}
ratings_df = pd.DataFrame(dummy_interactions).drop_duplicates(['User_ID', 'Product Name'])

# 3. CREATE USER-ITEM MATRIX
# This is the "Engine" of collaborative filtering
pivot_table = ratings_df.pivot(index='Product Name', columns='User_ID', values='Rating').fillna(0)

# 4. CALCULATE ITEM SIMILARITY (Manual Cosine Similarity)
# We calculate which products are "similar" based on user rating patterns
matrix = pivot_table.values
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
norms[norms == 0] = 1e-9 # Prevent division by zero
normalized_matrix = matrix / norms
similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
sim_df = pd.DataFrame(similarity_matrix, index=pivot_table.index, columns=pivot_table.index)

# --- START OF YOUR ORIGINAL LOGIC STRUCTURE ---

# Display available products
print("Available Products in System:")
print(", ".join(products[:15]) + "...") 

# Get user preferences (Now asking for a Product Name instead of Genre)
user_liked = input("\nEnter a product name you liked from our catalog: ").strip()

# Filter/Find similar items matching user preferences based on Collaborative Data
if user_liked in sim_df.index:
    # Get the similarity scores for that product and sort them
    recommendations = sim_df[user_liked].sort_values(ascending=False).iloc[1:6]
    
    print(f"\nRecommended Products for you (based on what other users liked):")
    for product_name, score in recommendations.items():
        # Get category info from the original dataframe
        category = df[df['Product Name'] == product_name]['CATEGORY'].values[0]
        print(f"- {product_name} | Category: {category} | Match: {int(score*100)}%")
else:
    print("\nNo data found for the product you entered. Please check the spelling.")

# --- END OF YOUR ORIGINAL LOGIC STRUCTURE ---
