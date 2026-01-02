import streamlit as st
import pandas as pd
import numpy as np

# 1. Page Config
st.set_page_config(page_title="Collaborative Recommender", layout="wide")

@st.cache_data
def load_and_clean_data():
    # Load your product_ratings.csv
    df = pd.read_csv('product_ratings.csv')
    # Remove empty columns (Unnamed columns)
    df = df.dropna(axis=1, how='all')
    # Filter out header repetitions
    df = df[df['Product Name'].astype(str).str.lower() != 'product name'].reset_index(drop=True)
    return df

@st.cache_resource
def build_collaborative_engine(df):
    """
    Collaborative filtering needs Users and Ratings. 
    Since the CSV is content-only, we generate a synthetic interaction matrix.
    """
    products = df['Product Name'].unique()
    np.random.seed(42)
    
    # Simulate 100 users and 2000 ratings
    num_users = 100
    user_ids = np.random.randint(1, num_users + 1, size=2000)
    product_names = np.random.choice(products, size=2000)
    ratings = np.random.randint(1, 6, size=2000)
    
    rdf = pd.DataFrame({'User': user_ids, 'Item': product_names, 'Rating': ratings})
    rdf = rdf.drop_duplicates(['User', 'Item'])
    
    # Pivot to create User-Item Matrix
    pivot = rdf.pivot(index='Item', columns='User', values='Rating').fillna(0)
    
    # Calculate Similarity (Cosine) manually using Numpy
    vals = pivot.values
    norms = np.linalg.norm(vals, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9 # Prevent division by zero
    normalized_vals = vals / norms
    similarity_matrix = np.dot(normalized_vals, normalized_vals.T)
    
    return pd.DataFrame(similarity_matrix, index=pivot.index, columns=pivot.index)

# --- APP UI ---
st.title("🛒 Collaborative Recommender System")
st.markdown("This app uses **Collaborative Filtering** to recommend items based on user behavior.")

try:
    # Load Data
    data = load_and_clean_data()
    sim_df = build_collaborative_engine(data)
    
    # Sidebar Search
    st.sidebar.header("Find Recommendations")
    product_list = sorted(data['Product Name'].unique())
    selected_item = st.sidebar.selectbox("Choose a product you like:", product_list)
    num_recs = st.sidebar.slider("Number of results:", 1, 10, 5)

    if st.sidebar.button("Recommend"):
        st.subheader(f"People who liked '{selected_item}' also liked:")
        
        if selected_item in sim_df.index:
            # Get similarity scores and sort
            results = sim_df[selected_item].sort_values(ascending=False).iloc[1:num_recs+1]
            
            # Create columns for the display
            cols = st.columns(num_recs)
            for i, (name, score) in enumerate(results.items()):
                with cols[i]:
                    # Pull category from the original dataframe
                    category_info = data[data['Product Name'] == name]['CATEGORY'].values
                    category = category_info[0] if len(category_info) > 0 else "General"
                    
                    st.info(f"**{name}**")
                    st.write(f"Category: {category}")
                    st.metric("Similarity", f"{int(score*100)}%")
        else:
            st.error("Item not found in the recommendation engine.")

    # Show data table
    with st.expander("Show Product Catalog"):
        st.dataframe(data)

except Exception as e:
    st.error("Error: Please make sure 'product_ratings.csv' is in your GitHub repository.")
    st.write(f"Technical details: {e}")
