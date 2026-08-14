import streamlit as st
import pandas as pd
import plotly.express as px
import re
from collections import Counter
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from outscraper import OutscraperClient

# Page Config
st.set_page_config(page_title="Review Sentiment Dashboard", layout="wide")
st.title("📊 Google Reviews Sentiment & Interaction Dashboard")

# Download NLTK Lexicons
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------
# 1. SIDEBAR CONFIGURATION
# ---------------------------------------------------------
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Outscraper API Key", type="password")
search_query = st.sidebar.text_input("Business Location", "Westfield Washington Public Library, IN, USA")
fetch_btn = st.sidebar.button("Fetch & Analyze Reviews")

# Session state to store processed data
if "df" not in st.session_state:
    st.session_state.df = None

# ---------------------------------------------------------
# 2. DATA SCRAPING & PROCESSING
# ---------------------------------------------------------
if fetch_btn and api_key:
    with st.spinner("Scraping reviews and running sentiment analysis..."):
        client = OutscraperClient(api_key=api_key)
        
        results = client.google_maps_business_reviews(
            search_query,
            reviews_limit=0,
            limit=1,
            sort='newest',
            ignore_empty=False
        )
        
        processed_reviews = []
        sia = SentimentIntensityAnalyzer()
        stop_words = set(stopwords.words('english'))
        
        category_keywords = {
            'Service': ['staff', 'librarian', 'help', 'friendly', 'patient', 'support', 'service', 'event', 'program', 'kids', 'attentive'],
            'Quality': ['great', 'clean', 'excellent', 'best', 'modern', 'top', 'wonderful', 'amazing', 'quiet', 'nice', 'dirty', 'bad'],
            'Price':   ['free', 'cost', 'price', 'fee', 'expensive', 'cheap', 'value', 'card', 'fine', 'tax', 'money', 'affordable']
        }

        for place in results:
            reviews_data = place.get('reviews_data', []) or []
            for r in reviews_data:
                text = r.get('review_text', '')
                if not text or not text.strip():
                    continue

                # Calculate Sentiment
                score = sia.polarity_scores(text)['compound']
                if score >= 0.05:
                    sentiment = 'Positive'
                elif score <= -0.05:
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'

                # Clean words
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                words_clean = [w for w in words if w not in stop_words]

                # Match Categories
                matched_categories = []
                for cat, kws in category_keywords.items():
                    if any(kw in words_clean for kw in kws):
                        matched_categories.append(cat)

                # Extract Owner Response Info
                owner_reply = r.get('owner_answer')
                has_owner_response = "Yes" if owner_reply else "No"
                owner_reply_date = r.get('owner_answer_timestamp_datetime_utc', 'N/A')
                
                # Extract Review Metadata
                review_date = r.get('review_datetime_utc', 'N/A')
                author = r.get('author_title', 'Anonymous')
                rating = r.get('review_rating', 'N/A')

                processed_reviews.append({
                    'Author': author,
                    'Rating': rating,
                    'Review Date': review_date,
                    'Review Text': text,
                    'Sentiment': sentiment,
                    'Sentiment Score': score,
                    'Categories': matched_categories if matched_categories else ['Uncategorized'],
                    'Has Owner Response': has_owner_response,
                    'Owner Response': owner_reply if owner_reply else "No response provided",
                    'Owner Response Date': owner_reply_date,
                    'Words': words_clean
                })

        st.session_state.df = pd.DataFrame(processed_reviews)
        st.success(f"Successfully loaded {len(st.session_state.df)} text reviews!")

# ---------------------------------------------------------
# 3. INTERACTIVE DASHBOARD & DRILL-DOWN
# ---------------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df

    st.markdown("---")
    
    # Category Selection Tabs
    tab_service, tab_quality, tab_price = st.tabs(["Service", "Quality", "Price"])

    def render_category_tab(category_name):
        # Filter dataframe for category
        cat_df = df[df['Categories'].apply(lambda cats: category_name in cats)]
        
        col1, col2 = st.columns(2)

        # Count words for Positive and Negative reviews
        def get_top_words(sentiment_filter):
            words_list = []
            sub_df = cat_df[cat_df['Sentiment'] == sentiment_filter]
            for words in sub_df['Words']:
                words_list.extend(words)
            counter = Counter(words_list)
            return pd.DataFrame(counter.most_common(10), columns=['Word', 'Count'])

        pos_words_df = get_top_words('Positive')
        neg_words_df = get_top_words('Negative')

        with col1:
            st.subheader(f"Top Positive Words ({category_name})")
            if not pos_words_df.empty:
                fig_pos = px.bar(
                    pos_words_df, x='Count', y='Word', orientation='h',
                    color_discrete_sequence=['#2ecc71'],
                    title=f"Positive Words - {category_name}"
                )
                fig_pos.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_pos, use_container_width=True)
            else:
                st.info("No positive reviews found for this category.")

        with col2:
            st.subheader(f"Top Negative Words ({category_name})")
            if not neg_words_df.empty:
                fig_neg = px.bar(
                    neg_words_df, x='Count', y='Word', orientation='h',
                    color_discrete_sequence=['#e74c3c'],
                    title=f"Negative Words - {category_name}"
                )
                fig_neg.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_neg, use_container_width=True)
            else:
                st.info("No negative reviews found for this category.")

        # DRILL-DOWN INTERACTIVE FILTERS
        st.markdown(f"### 🔍 Drill Down Reviews for **{category_name}**")
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            selected_sentiment = st.selectbox(
                "Filter Sentiment", ["All", "Positive", "Negative"], key=f"sent_{category_name}"
            )
        with filter_col2:
            selected_response = st.selectbox(
                "Filter Owner Response", ["All", "Yes", "No"], key=f"resp_{category_name}"
            )
        with filter_col3:
            all_cat_words = list(set([w for sublist in cat_df['Words'] for w in sublist]))
            selected_word = st.selectbox(
                "Filter by Word Mentioned", ["All"] + sorted(all_cat_words), key=f"word_{category_name}"
            )

        # Apply Interactive Filters
        filtered_df = cat_df.copy()
        if selected_sentiment != "All":
            filtered_df = filtered_df[filtered_df['Sentiment'] == selected_sentiment]
        if selected_response != "All":
            filtered_df = filtered_df[filtered_df['Has Owner Response'] == selected_response]
        if selected_word != "All":
            filtered_df = filtered_df[filtered_df['Words'].apply(lambda w_list: selected_word in w_list)]

        st.write(f"Showing **{len(filtered_df)}** matching reviews:")

        # Display Interactive Data Table with Timestamps and Owner Answers
        for _, row in filtered_df.iterrows():
            with st.expander(f"⭐ {row['Rating']} Stars | {row['Author']} ({row['Review Date']}) - Sentiment: {row['Sentiment']}"):
                st.markdown(f"**Review Text:**\n> {row['Review Text']}")
                st.markdown(f"**Review Timestamp:** `{row['Review Date']}`")
                
                if row['Has Owner Response'] == "Yes":
                    st.success(f"**Owner Response ({row['Owner Response Date']}):**\n\n{row['Owner Response']}")
                else:
                    st.warning("⚠️ No owner response recorded for this review.")

    with tab_service:
        render_category_tab("Service")

    with tab_quality:
        render_category_tab("Quality")

    with tab_price:
        render_category_tab("Price")

else:
    st.info("Enter your Outscraper API Key in the sidebar and click 'Fetch & Analyze Reviews' to start.")