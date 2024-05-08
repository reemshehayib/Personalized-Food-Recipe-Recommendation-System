import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Add CSS for background image
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://wallpapers.com/images/hd/food-4k-1vrcb0mw76zcg4qf.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .input-field {
        margin-bottom: 20px;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #2b2b2b;
        font-size: 18px;
        width: 70%;
        box-sizing: border-box;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add titles with color and styling
st.markdown(
    """
    <style>
    .header {
        color: #2b2b2b;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #2b2b2b;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 40px;
    }
    </style>
    """
    , unsafe_allow_html=True)

st.markdown("<h1 class='header'>Welcome to Foodie's Delight</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>Pick Your Favorite Ingredients</h2>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>The Ultimate Food Recommendation System</h2>", unsafe_allow_html=True)

# Define a beautiful background color
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom right, #ffcccb, #f9f8e6);
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load dataset
def load_data():
    data = pd.read_csv("data.csv")
    return data

data = load_data()

# Load pre-trained model
kmeans = joblib.load("kmeans_model.pkl")
vectorizer = TfidfVectorizer()
ingredients_matrix = vectorizer.fit_transform(data['Cleaned_Ingredients'])
kmeans = KMeans(n_clusters=4, random_state=0)
data['Ingredients_Cluster'] = kmeans.fit_predict(ingredients_matrix)
# Recommendation function
def recommend_dishes(data, user_input, n_recommendations):
    user_input = user_input.lower()
    user_vector = vectorizer.transform([user_input])

    user_cluster = kmeans.predict(user_vector)[0]
    cluster_data = data[data['Cluster'] == user_cluster]
    cluster_matrix = vectorizer.transform(cluster_data['Cleaned_Ingredients'])

    similarities = cosine_similarity(user_vector, cluster_matrix)
    top_indices = similarities[0].argsort()[-n_recommendations:][::-1]
    recommended_dishes = cluster_data.iloc[top_indices]

    return recommended_dishes[['Title', 'Cleaned_Ingredients', 'Instructions']]

# User input for number of recommendations
with st.container():
    n_recommendations = st.number_input(
        "Enter the number of recommendations:",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

# User input and recommendation button
with st.container():
    user_input = st.text_input("Enter ingredients separated by commas:", key="user_input")
    st.markdown(
        """
        <style>
        .input-field {
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 5px;
            border: 2px solid #2b2b2b;
            font-size: 18px;
            width: 70%;
            box-sizing: border-box;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Recommend"):
        if user_input:
            recommended_dishes = recommend_dishes(data, user_input, n_recommendations)
            if not recommended_dishes.empty:
                st.subheader("Recommended Dishes:")
                for idx, row in recommended_dishes.iterrows():
                    title = row["Title"]
                    ingredients = row["Cleaned_Ingredients"]
                    instructions = row["Instructions"]

                    # Convert ingredients string to list
                    ingredients_list = ingredients.strip("[]").replace("'", "").split(", ")

                    with st.expander(title):
                        st.markdown("<h3>Ingredients:</h3>", unsafe_allow_html=True)
                        for ingredient in ingredients_list:
                            st.markdown(f"- {ingredient.strip()}")
                        st.markdown("<h3>Instructions:</h3>", unsafe_allow_html=True)
                        instructions_list = [instruction.strip() for instruction in instructions.split(".") if instruction]
                        for instruction in instructions_list:
                            st.markdown(f"- {instruction.strip()}")
            else:
                st.write("No recommended dishes found. Please try a different combination of ingredients.")
        else:
            st.warning("Please enter ingredients to get recommendations.")

# Sidebar information
st.sidebar.header("About This App")
st.sidebar.info("Welcome to the Food Recommendation System! This web app suggests dishes based on the ingredients you provide.")
st.sidebar.info("The more ingredients you specify, the more accurate the recommendations will be.")
st.sidebar.info("To get your recommended dishes, simply enter the ingredients you'd like to use and click on the 'Recommend' button.")
st.sidebar.info("Example: chicken, salt, rice, tomato, lettuce, pepper, cucumber")
st.sidebar.info("Be creative!")

# Footer
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #2b2b2b;
        background-color: #f3f3f3;
        padding: 20px;
        border-top: 2px solid #ddd;
        width: 100%; /* Set width to 100% */
    }
    .footer-content {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        max-width: 1200px;
        margin: 0 auto;
    }
    .footer-section {
        flex: 1;
        margin: 10px;
    }
    .footer-section h3 {
        font-size: 18px;
        margin-bottom: 10px;
        color: #555;
    }
    .footer-section p {
        font-size: 14px;
        color: #777;
    }
    .footer-section a {
        color: #555;
        text-decoration: none;
    }
    </style>
    """
    , unsafe_allow_html=True
)

st.markdown(
    """
    <div class='footer'>
        <div class='footer-content'>
            <div class='footer-section'>
                <h3>Contact Us</h3>
                <p><a href="mailto:info@foodiesdelight.com">Email: info@foodiesdelight.com</a></p>
                <p><a href="tel:+96176675173">Phone: +961 76 675173</a></p>
            </div>
            <div class='footer-section'>
                <h3>Address</h3>
                <p>Bazerkan Bldg. Floor 7, Beirut, Lebanon</p>
            </div>
            <div class='footer-section'>
                <h3>Follow Us</h3>
                <p><a href="https://www.facebook.com/">Facebook</a></p>
                <p><a href="https://www.twitter.com/">Twitter</a></p>
                <p><a href="https://www.instagram.com/">Instagram</a></p>
            </div>
        </div>
    </div>
    """
    , unsafe_allow_html=True
)
