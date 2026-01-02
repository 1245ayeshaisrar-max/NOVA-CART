import streamlit as st
import pandas as pd
import numpy as np

# 1. SETUP PAGE
st.set_page_config(page_title="Nova Cart Recommender", layout="wide")

@st.cache_data
def load_and_clean_data():
    # Load the product CSV
    df = pd.read_csv('product_ratings.csv')
    # Remove the empty columns found in your file
    df = df.dropna(axis=1, how='all')
    # Filter out header repetitions
    df = df[df['Product Name'].astype(str).str.lower() != 'product name'].reset_index(drop=True)
    return df

@st.cache_resource
def build_engine(df):
    # Get unique products
    products = df['Product Name'].unique()
    
    # SIMULATE USER DATA (Since CSV has no User IDs)
    np.random.seed(42)
    user_ids = np.random.randint(1, 51, size=1000)
    product_names = np.random.choice(products, size=1000)
    ratings = np.random.randint(1, 6, size=1000)
    
    rdf = pd.DataFrame({'User': user_ids, 'Item': product_names, 'Rating': ratings})
    rdf = rdf.drop_duplicates(['User', 'Item'])
    
    # Create User-Item Matrix
    pivot = rdf.pivot(index='Item', columns='User', values='Rating').fillna(0)
    
    # --- CALCULATE COSINE SIMILARITY MANUALLY (NO SKLEARN) ---
    item_vectors = pivot.values
    # Calculate norms for each row (item)
    norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9 # Avoid division by zero
    # Normalize vectors
    normalized_items = item_vectors / norms
    # Dot product of normalized vectors = Cosine Similarity
    sim_matrix = np.dot(normalized_items, normalized_items.T)
    
    return pd.DataFrame(sim_matrix, index=pivot.index, columns=pivot.index)

# 2. APP INTERFACE
st.title("🛒 Nova Cart Recommender")
st.markdown("### Collaborative Filtering (Powered by Numpy)")

try:
    data = load_and_clean_data()
    sim_df = build_engine(data)
    
    # Sidebar selection
    st.sidebar.header("User Selection")
    all_prods = sorted(data['Product Name'].unique())
    user_choice = st.sidebar.selectbox("Select a product:", all_prods)
    num_recs = st.sidebar.slider("Recommendations:", 1, 10, 5)

    if st.sidebar.button("Show Recommendations"):
        st.write(f"#### Because you liked: **{user_choice}**")
        
        if user_choice in sim_df.index:
            # Get similarity scores, skip the first one (itself)
            recs = sim_df[user_choice].sort_values(ascending=False).iloc[1:num_recs+1]
            
            cols = st.columns(len(recs))
            for i, (name, score) in enumerate(recs.items()):
                with cols[i]:
                    # Get category from original data
                    cat = data[data['Product Name'] == name]['CATEGORY'].values[0]
                    st.success(f"**{name}**")
                    st.caption(f"Category: {cat}")
                    st.metric("Match", f"{int(score*100)}%")
        else:
            st.error("Not enough data to find recommendations for this item.")

except Exception as e:
    st.error("System Error")
    st.write(f"Please ensure 'product_ratings.csv' is in the root folder. Error: {e}")
