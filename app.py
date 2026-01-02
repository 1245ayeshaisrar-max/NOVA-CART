import streamlit as st
import pandas as pd
import numpy as np

# 1. SETUP PAGE
st.set_page_config(page_title="Nova Cart Recommender", layout="wide", page_icon="🛒")

@st.cache_data
def load_and_clean_data():
    try:
        # Load the product CSV
        df = pd.read_csv('product_ratings.csv')
        # Remove the empty columns (Unnamed) often found in CSV exports
        df = df.dropna(axis=1, how='all')
        # Remove rows that repeat the header
        df = df[df['Product Name'].astype(str).str.lower() != 'product name'].reset_index(drop=True)
        # Fill missing values
        df = df.fillna('Unknown')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def get_recommendations(target_product, df, top_n=5):
    """
    Finds products similar to the target based on Category, Usage, and Target User.
    """
    # Get features of the selected product
    selected_row = df[df['Product Name'] == target_product].iloc[0]
    
    # Create a simple scoring system
    # We give points for matching Category, Usage, and Target User
    scores = []
    
    for idx, row in df.iterrows():
        if row['Product Name'] == target_product:
            scores.append(-1) # Don't recommend itself
            continue
            
        score = 0
        if row['CATEGORY'] == selected_row['CATEGORY']:
            score += 5  # Highest weight for same category
        if row['USAGE'] == selected_row['USAGE']:
            score += 3  # Medium weight for same usage
        if row['TARGET USER'] == selected_row['TARGET USER']:
            score += 2  # Weight for same target audience
            
        scores.append(score)
    
    # Get indices of top scorers
    df['Score'] = scores
    recommendations = df.sort_values(by='Score', ascending=False).head(top_n)
    return recommendations

# 2. APP INTERFACE
st.title("🛒 Nova Cart Smart Recommender")
st.markdown("### Accurate Product Recommendations based on Category & Usage")

df = load_and_clean_data()

if df is not None:
    # Sidebar
    st.sidebar.header("Product Search")
    all_products = sorted(df['Product Name'].unique())
    user_choice = st.sidebar.selectbox("What product are you interested in?", all_products)
    num_recs = st.sidebar.slider("Number of recommendations:", 1, 10, 5)

    if st.sidebar.button("Find Similar Products"):
        st.write(f"#### Because you liked: **{user_choice}**")
        
        # Get recommendations
        recs = get_recommendations(user_choice, df, num_recs)
        
        # Display results in a nice grid
        cols = st.columns(len(recs))
        for i, (idx, row) in enumerate(recs.iterrows()):
            with cols[i]:
                # Use st.container for a card-like look
                with st.container():
                    st.success(f"**{row['Product Name']}**")
                    st.markdown(f"**Category:** {row['CATEGORY']}")
                    st.markdown(f"**Best for:** {row['USAGE']}")
                    st.markdown(f"**Target:** {row['TARGET USER']}")
                    if 'PRICE RANGE' in df.columns:
                        st.caption(f"Price: {row['PRICE RANGE']}")

    # Show the catalog
    with st.expander("Explore Full Product Catalog"):
        st.dataframe(df[['Product Name', 'CATEGORY', 'USAGE', 'TARGET USER', 'PRICE RANGE']])
else:
    st.warning("Please make sure 'product_ratings.csv' is uploaded to your GitHub repository.")

# 3. FOOTER
st.markdown("---")
st.caption("Algorithm: Content-Based Feature Matching (Category + Usage + Target User)")
