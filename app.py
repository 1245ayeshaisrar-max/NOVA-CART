import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Nova Cart Recommender", layout="wide")

@st.cache_data
def load_and_clean_data():
    # Load the dataset
    df = pd.read_csv('product_ratings.csv')
    
    # Clean empty columns and repeated headers
    df = df.dropna(axis=1, how='all')
    df = df[df['Product Name'] != 'Product Name'].reset_index(drop=True)
    return df

@st.cache_resource
def build_recommender_engine(df):
    product_list = df['Product Name'].unique()
    
    # SIMULATION: Generating synthetic ratings 
    np.random.seed(42)
    num_users = 100
    dummy_data = {
        'User_ID': np.random.randint(1, num_users + 1, size=1200),
        'Product Name': np.random.choice(product_list, size=1200),
        'Rating': np.random.randint(1, 6, size=1200)
    }
    ratings_df = pd.DataFrame(dummy_data).drop_duplicates(['User_ID', 'Product Name'])
    
    # Create User-Item Matrix (Users as rows, Products as columns)
    pivot_table = ratings_df.pivot(index='User_ID', columns='Product Name', values='Rating').fillna(0)
    
    # --- MANUAL COSINE SIMILARITY (No Sklearn needed) ---
    # We calculate similarity between columns (Items)
    matrix = pivot_table.values.T  # Transpose to get Items x Users
    dot_product = np.dot(matrix, matrix.T)
    norms = np.linalg.norm(matrix, axis=1)
    
    # Avoid division by zero
    norms[norms == 0] = 1e-9
    
    # Similarity = Dot Product / (Norm_A * Norm_B)
    sim_matrix = dot_product / np.outer(norms, norms)
    item_sim_df = pd.DataFrame(sim_matrix, index=pivot_table.columns, columns=pivot_table.columns)
    
    return item_sim_df

# UI Layout
st.title("🛒 Nova Cart: Collaborative Recommender")
st.markdown("Finding products you'll love based on community preferences.")

try:
    df = load_and_clean_data()
    item_sim_df = build_recommender_engine(df)
    
    # Sidebar
    st.sidebar.header("Settings")
    all_products = sorted(df['Product Name'].unique())
    selected_product = st.sidebar.selectbox("Pick a product you like:", all_products)
    num_recs = st.sidebar.slider("How many suggestions?", 1, 10, 5)

    if st.sidebar.button("Generate Recommendations"):
        st.subheader(f"Because you liked '{selected_product}'...")
        
        # Get recommendations from our similarity matrix
        if selected_product in item_sim_df.columns:
            recommendations = item_sim_df[selected_product].sort_values(ascending=False).iloc[1:num_recs+1]
            
            # Display results in columns
            cols = st.columns(num_recs)
            for i, (name, score) in enumerate(recommendations.items()):
                with cols[i]:
                    # Find category for the recommended item
                    cat_list = df[df['Product Name'] == name]['CATEGORY'].values
                    category = cat_list[0] if len(cat_list) > 0 else "General"
                    
                    st.success(f"**{name}**")
                    st.caption(f"Category: {category}")
                    st.metric("Match Score", f"{int(score*100)}%")
        else:
            st.error("This product doesn't have enough data yet!")

    # Dataset Preview
    with st.expander("Browse Full Catalog"):
        st.write(df)

except Exception as e:
    st.error(f"Waiting for data... Ensure 'product_ratings.csv' is in your GitHub repository.")
    st.info(f"Technical Error: {e}")
