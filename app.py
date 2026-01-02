import streamlit as st
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Nova Cart Recommender", layout="wide")

@st.cache_data
def load_data():
    # Load and clean the user's specific dataset
    df = pd.read_csv('product_ratings.csv')
    df = df.dropna(axis=1, how='all')
    # Remove rows that are just header repetitions
    df = df[df['Product Name'].str.lower() != 'product name'].reset_index(drop=True)
    return df

@st.cache_resource
def build_engine(df):
    # Collaborative filtering requires User/Rating data. 
    # Since the CSV is content-only, we simulate 100 users.
    products = df['Product Name'].unique()
    np.random.seed(42)
    
    # Create synthetic interaction data
    user_ids = np.random.randint(1, 101, size=1500)
    product_names = np.random.choice(products, size=1500)
    ratings = np.random.randint(1, 6, size=1500)
    
    rdf = pd.DataFrame({'User': user_ids, 'Item': product_names, 'Rating': ratings})
    rdf = rdf.drop_duplicates(['User', 'Item'])
    
    # Create Matrix
    matrix = rdf.pivot(index='Item', columns='User', values='Rating').fillna(0)
    
    # --- Manual Cosine Similarity Calculation ---
    # This replaces the need for the 'sklearn' library
    vals = matrix.values
    norms = np.linalg.norm(vals, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9 # Prevent division by zero
    normalized_vals = vals / norms
    similarity_matrix = np.dot(normalized_vals, normalized_vals.T)
    
    return pd.DataFrame(similarity_matrix, index=matrix.index, columns=matrix.index)

# 2. App Logic
st.title("🛒 Nova Cart Recommender")
st.info("Collaborative Filtering Engine (using Numpy & Pandas)")

try:
    data = load_data()
    sim_df = build_engine(data)
    
    # User selection
    target_item = st.selectbox("Select a product you like:", sorted(data['Product Name'].unique()))
    num_recs = st.slider("Number of recommendations", 1, 10, 5)
    
    if st.button("Find Similar Products"):
        if target_item in sim_df.index:
            # Get scores and sort
            scores = sim_df[target_item].sort_values(ascending=False).iloc[1:num_recs+1]
            
            cols = st.columns(len(scores))
            for i, (name, score) in enumerate(scores.items()):
                with cols[i]:
                    cat = data[data['Product Name'] == name]['CATEGORY'].values[0]
                    st.success(f"**{name}**")
                    st.caption(f"Category: {cat}")
                    st.metric("Match", f"{int(score*100)}%")
        else:
            st.warning("Not enough interaction data for this item yet.")

except Exception as e:
    st.error("Setup Error: Ensure 'product_ratings.csv' is in your GitHub folder.")
    st.write(f"Debug Info: {e}")
