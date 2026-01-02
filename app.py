import streamlit as st
import pandas as pd
import numpy as np

# 1. SETUP PAGE
st.set_page_config(page_title="Nova Cart Recommender", layout="wide", page_icon="🛒")

@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv('product_ratings.csv')
        # Clean empty columns (Unnamed)
        df = df.dropna(axis=1, how='all')
        # Filter out header repetitions
        df = df[df['Product Name'].astype(str).str.lower() != 'product name'].reset_index(drop=True)
        # Fill empty spots
        df = df.fillna('General')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

@st.cache_resource
def build_content_engine(df):
    """
    This engine links items by Category, Usage, and Target User 
    so that Abayas relate to Clothing, not Sugar.
    """
    # Select columns to use for "relatedness"
    features = ['CATEGORY', 'USAGE', 'TARGET USER']
    
    # Convert text features into numbers (One-Hot Encoding)
    # This creates a matrix where similar categories have similar values
    feature_df = pd.get_dummies(df[features])
    
    # --- MANUAL COSINE SIMILARITY (Numpy) ---
    matrix = feature_df.values
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9 # Prevent division by zero
    normalized_matrix = matrix / norms
    
    # Calculate similarity between all products
    similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
    
    return pd.DataFrame(similarity_matrix, index=df['Product Name'], columns=df['Product Name'])

# 2. APP UI
st.title("🛍️ Nova Cart Smart Recommender")
st.markdown("### Discover Related Products based on Category & Usage")

df = load_and_clean_data()

if df is not None:
    # Build engine
    sim_df = build_content_engine(df)
    
    # Sidebar search
    st.sidebar.header("Find Recommendations")
    product_list = sorted(df['Product Name'].unique())
    selected_item = st.sidebar.selectbox("Select a product you like:", product_list)
    num_recs = st.sidebar.slider("How many recommendations?", 1, 10, 5)

    if st.sidebar.button("Show Related Items"):
        st.subheader(f"Items related to '{selected_item}':")
        
        if selected_item in sim_df.index:
            # Get the top similar items (excluding itself)
            # We use .iloc[1:] to skip the selected product
            # We group by name to avoid showing duplicate entries if the CSV has them
            item_scores = sim_df[selected_item].sort_values(ascending=False)
            recommendations = item_scores[item_scores.index != selected_item].head(num_recs)
            
            # Display results in columns
            cols = st.columns(len(recommendations))
            for i, (name, score) in enumerate(recommendations.items()):
                with cols[i]:
                    # Get product details for the recommended item
                    details = df[df['Product Name'] == name].iloc[0]
                    st.success(f"**{name}**")
                    st.caption(f"Category: {details['CATEGORY']}")
                    st.write(f"Usage: {details['USAGE']}")
                    st.metric("Match Score", f"{int(score*100)}%")
        else:
            st.error("Could not find this product in the similarity matrix.")

    # Catalog View
    with st.expander("View Full Product Catalog"):
        st.dataframe(df)
else:
    st.info("Please ensure 'product_ratings.csv' is uploaded to your GitHub repository.")
