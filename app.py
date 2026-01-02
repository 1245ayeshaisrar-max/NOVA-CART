import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="Product Recommender", layout="wide")

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
    
    # SIMULATION: Since your CSV doesn't have User IDs, we generate 
    # synthetic ratings to demonstrate Collaborative Filtering.
    # In a real app, you would load a separate 'ratings.csv' here.
    np.random.seed(42)
    num_users = 100
    dummy_data = {
        'User_ID': np.random.randint(1, num_users + 1, size=1000),
        'Product Name': np.random.choice(product_list, size=1000),
        'Rating': np.random.randint(1, 6, size=1000)
    }
    ratings_df = pd.DataFrame(dummy_data).drop_duplicates(['User_ID', 'Product Name'])
    
    # Create User-Item Matrix
    pivot_table = ratings_df.pivot(index='User_ID', columns='Product Name', values='Rating').fillna(0)
    
    # Calculate Similarity
    item_sim_matrix = cosine_similarity(pivot_table.T)
    item_sim_df = pd.DataFrame(item_sim_matrix, index=pivot_table.columns, columns=pivot_table.columns)
    
    return item_sim_df

# UI Layout
st.title("🛍️ Collaborative Product Recommender")
st.markdown("This system recommends products based on user behavior patterns using Cosine Similarity.")

try:
    df = load_and_clean_data()
    item_sim_df = build_recommender_engine(df)
    
    # Sidebar search
    st.sidebar.header("User Input")
    all_products = sorted(df['Product Name'].unique())
    selected_product = st.sidebar.selectbox("Select a product you like:", all_products)
    
    num_recs = st.sidebar.slider("Number of recommendations:", 1, 10, 5)

    if st.button("Get Recommendations"):
        st.subheader(f"Products similar to '{selected_product}'")
        
        # Get recommendations
        recommendations = item_sim_df[selected_product].sort_values(ascending=False).iloc[1:num_recs+1]
        
        if not recommendations.empty:
            cols = st.columns(num_recs)
            for i, (name, score) in enumerate(recommendations.items()):
                with cols[i]:
                    # Get category info from original df
                    category = df[df['Product Name'] == name]['CATEGORY'].values[0]
                    st.info(f"**{name}**")
                    st.write(f"Category: {category}")
                    st.write(f"Match: {score:.2%}")
        else:
            st.warning("No similar products found in the interaction data.")

    # Show Data Preview
    with st.expander("View Product Dataset"):
        st.dataframe(df)

except FileNotFoundError:
    st.error("Error: 'product_ratings.csv' not found. Please upload it to the repository.")
