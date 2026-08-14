import streamlit as st
import pandas as pd
import plotly.express as px
import re
from collections import Counter
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from outscraper import OutscraperClient

# Page Setup
st.set_page_config(page_title="Review Sentiment Dashboard", layout="wide")
st.title("📊 Customer Feedback & Sentiment Analytics")

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------
# 1. API KEY & SECRETS CONFIGURATION
# ---------------------------------------------------------
# Pulls key automatically from Streamlit Secrets or environment
if "OUTSCRAPER_API_KEY" in st.secrets:
    API_KEY = st.secrets["OUTSCRAPER_API_KEY"]
else:
    # Fallback to direct key string if secrets file is not used
    API_KEY = "ZXAjNzVkMGQ1ZjhjMmRhNDdlYWEyZGI3NTdlZjYwZjViYjR8ZTIzNWJjY2I3MA"

# Sidebar controls for location and safety limits
st.sidebar.header("🔍 Analysis Settings")
search_query = st.sidebar.text_input("Business Location", "Westfield Washington Public Library, IN, USA")
review_limit = st.sidebar.number_input("Review Limit (0 = All)", min_value=0, max_value=500, value=100)

fetch_btn = st.sidebar.button("🚀 Run Sentiment Analysis")

# Session state to hold analyzed data
if "df" not in st.session_state:
    st.session_state.df = None

# ---------------------------------------------------------
# 2. DATA FETCHING & PROCESSING
# ---------------------------------------------------------
if fetch_btn:
    with st.spinner("Scraping reviews and processing sentiment scores..."):
        try:
            client = OutscraperClient(api_key=API_KEY)
            
            results = client.google_maps_business_reviews(
                search_query,
                reviews_limit=review_limit,
                limit=1,
                sort='newest',
                ignore_empty=False
            )
            
            processed_reviews = []
            sia = SentimentIntensityAnalyzer()
            stop_words = set(stopwords.words('english'))
            
            # Key category definitions
            category_keywords = {
                'Service': ['staff', 'librarian', 'help', 'friendly', 'patient', 'support', 'service', 'event', 'program', 'kids', 'attentive', 'kind'],
                'Quality': ['great', 'clean', 'excellent', 'best', 'modern', 'top', 'wonderful', 'amazing', 'quiet', 'nice', 'dirty', 'bad'],
                'Price':   ['free', 'cost', 'price', 'fee', 'expensive', 'cheap', 'value', 'card', 'fine', 'tax', 'money', 'affordable', 'worth']
            }

            for place in results:
                reviews_data = place.get('reviews_data', []) or []
                for r in reviews_data:
                    text = r.get('review_text', '')
                    if not text or not text.strip():
                        continue

                    # VADER Sentiment Scoring
                    score = sia.polarity_scores(text)['compound']
                    if score >= 0.05:
                        sentiment = 'Positive'
                    elif score <= -0.05:
                        sentiment = 'Negative'
                    else:
                        sentiment = 'Neutral'

                    # Word Cleaning
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                    words_clean = [w for w in words if w not in stop_words]

                    # Category Keyword Mapping
                    matched_categories = []
                    for cat, kws in category_keywords.items():
                        if any(kw in words_clean for kw in kws):
                            matched_categories.append(cat)

                    # Extract Response Metadata
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
            st.success(f"Analysis complete! Loaded {len(st.session_state.df)} reviews.")

        except Exception as e:
            st.error(f"Error fetching reviews: {e}")

# ---------------------------------------------------------
# 3. DASHBOARD & DATA AUDIT
# ---------------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df

    # --- TOP METRICS AUDIT TRAIL ---
    st.markdown("### 📈 Data Audit Summary")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Reviews Analyzed", len(df))
    a2.metric("Positive Reviews", len(df[df['Sentiment'] == 'Positive']))
    a3.metric("Negative Reviews", len(df[df['Sentiment'] == 'Negative']))
    a4.metric("Neutral (Excluded)", len(df[df['Sentiment'] == 'Neutral']))

    # --- CSV EXPORT SIDEBAR BUTTON ---
    @st.cache_data
    def convert_df_to_csv(data_frame):
        # Drop list object 'Words' for clean CSV export
        export_df = data_frame.drop(columns=['Words'])
        return export_df.to_csv(index=False).encode('utf-8')

    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Download Audit Data (CSV)",
        data=convert_df_to_csv(df),
        file_name="review_sentiment_audit.csv",
        mime="text/csv"
    )

    # --- METHODOLOGY EXPLAINER ---
    with st.expander("ℹ️ How are these counts and sentiment categories calculated?"):
        st.write("""
        * **Sentiment Classification:** Powered by NLTK VADER NLP. Sentiment score $\ge 0.05$ = Positive, $\le -0.05$ = Negative, between = Neutral.
        * **Category Placement:** Reviews containing category-specific terms (e.g., *staff, clean, cost*) are tagged accordingly. Multi-topic reviews appear in all matching category tabs.
        * **Word Frequencies vs. Review Counts:** The bar charts highlight the **frequency of key words**, not the total count of individual reviews.
        * **Neutral Exclusion:** Neutral reviews are listed in the total audit count above, but excluded from positive vs. negative word histograms.
        """)

    st.markdown("---")

    # --- CATEGORY TABS ---
    tab_service, tab_quality, tab_price = st.tabs(["🤝 Service", "⭐ Quality", "💵 Price"])

    def render_category_tab(category_name):
        cat_df = df[df['Categories'].apply(lambda cats: category_name in cats)]
        
        col1, col2 = st.columns(2)

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
                    title=f"Positive Keywords ({category_name})"
                )
                fig_pos.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_pos, use_container_width=True)
            else:
                st.info("No positive reviews found matching this category.")

        with col2:
            st.subheader(f"Top Negative Words ({category_name})")
            if not neg_words_df.empty:
                fig_neg = px.bar(
                    neg_words_df, x='Count', y='Word', orientation='h',
                    color_discrete_sequence=['#e74c3c'],
                    title=f"Negative Keywords ({category_name})"
                )
                fig_neg.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_neg, use_container_width=True)
            else:
                st.info("No negative reviews found matching this category.")

        # --- DRILL-DOWN FILTERS ---
        st.markdown(f"### 🔍 Drill Down Reviews for **{category_name}**")
        
        f1, f2, f3 = st.columns(3)
        with f1:
            selected_sentiment = st.selectbox("Sentiment", ["All", "Positive", "Negative"], key=f"s_{category_name}")
        with f2:
            selected_response = st.selectbox("Owner Responded?", ["All", "Yes", "No"], key=f"r_{category_name}")
        with f3:
            all_cat_words = list(set([w for sublist in cat_df['Words'] for w in sublist]))
            selected_word = st.selectbox("Contains Keyword", ["All"] + sorted(all_cat_words), key=f"w_{category_name}")

        filtered_df = cat_df.copy()
        if selected_sentiment != "All":
            filtered_df = filtered_df[filtered_df['Sentiment'] == selected_sentiment]
        if selected_response != "All":
            filtered_df = filtered_df[filtered_df['Has Owner Response'] == selected_response]
        if selected_word != "All":
            filtered_df = filtered_df[filtered_df['Words'].apply(lambda w_list: selected_word in w_list)]

        st.write(f"Displaying **{len(filtered_df)}** matching reviews:")

        # --- EXPANDABLE REVIEW CARDS ---
        for _, row in filtered_df.iterrows():
            with st.expander(f"⭐ {row['Rating']} Stars | {row['Author']} ({row['Review Date']}) - {row['Sentiment']}"):
                st.markdown(f"**Review:**\n> {row['Review Text']}")
                st.markdown(f"**Timestamp:** `{row['Review Date']}`")
                
                if row['Has Owner Response'] == "Yes":
                    st.success(f"**Owner Response ({row['Owner Response Date']}):**\n\n{row['Owner Response']}")
                else:
                    st.warning("⚠️ No owner response recorded.")

    with tab_service:
        render_category_tab("Service")

    with tab_quality:
        render_category_tab("Quality")

    with tab_price:
        render_category_tab("Price")

else:
    st.info("Click '🚀 Run Sentiment Analysis' in the sidebar to fetch data and generate charts.")