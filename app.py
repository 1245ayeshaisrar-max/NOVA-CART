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
    to ensure relatedness (e.g., Abayas relate to Clothing).
    """
    features = ['CATEGORY', 'USAGE', 'TARGET USER']
    # One-Hot Encoding features for mathematical comparison
    feature_df = pd.get_dummies(df[features])
    
    # Manual Cosine Similarity using Numpy
    matrix = feature_df.values
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9 
    normalized_matrix = matrix / norms
    similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
    
    return pd.DataFrame(similarity_matrix, index=df['Product Name'], columns=df['Product Name'])

# 2. APP UI
st.title("🛍️ Nova Cart Smart Recommender")

df = load_and_clean_data()

if df is not None:
    # Build engine
    sim_df = build_content_engine(df)
    
    # Sidebar
    st.sidebar.header("Search Product")
    product_list = sorted(df['Product Name'].unique())
    selected_item = st.sidebar.selectbox("Select a product:", product_list)
    num_recs = st.sidebar.slider("Number of recommendations:", 1, 10, 5)

    # --- TASK: DISPLAY DETAILS OF SEARCHED ITEM ---
    st.subheader(f"🔍 Details for: {selected_item}")
    
    # Get the row for the selected item
    selected_details = df[df['Product Name'] == selected_item].iloc[0]
    
    # Display details in cards/columns
    det_col1, det_col2, det_col3, det_col4 = st.columns(4)
    with det_col1:
        st.metric("Category", selected_details['CATEGORY'])
    with det_col2:
        st.metric("Target User", selected_details['TARGET USER'])
    with det_col3:
        st.metric("Usage", selected_details['USAGE'])
    with det_col4:
        # Check if Price Range exists in the dataset
        price = selected_details.get('PRICE RANGE', 'N/A')
        st.metric("Price Range", price)

    st.markdown("---")

    # --- RECOMMENDATIONS SECTION ---
    if st.sidebar.button("Find Related Items"):
        st.subheader(f"✨ Items similar to '{selected_item}'")
        
        if selected_item in sim_df.index:
            # Get similarity scores and filter out the selected item itself
            item_scores = sim_df[selected_item].sort_values(ascending=False)
            recommendations = item_scores[item_scores.index != selected_item].head(num_recs)
            
            # Display results in columns
            cols = st.columns(len(recommendations))
            for i, (name, score) in enumerate(recommendations.items()):
                with cols[i]:
                    # Pull details for each recommendation
                    rec_details = df[df['Product Name'] == name].iloc[0]
                    st.success(f"**{name}**")
                    st.caption(f"Category: {rec_details['CATEGORY']}")
                    st.write(f"Usage: {rec_details['USAGE']}")
                    st.metric("Match Score", f"{int(score*100)}%")
        else:
            st.error("Similarity data not available for this item.")

    # Catalog View
    with st.expander("Explore Full Data Catalog"):
        st.dataframe(df)
else:
    st.info("Please upload 'product_ratings.csv' to your GitHub repository.")
