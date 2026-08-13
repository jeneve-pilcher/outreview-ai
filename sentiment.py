import re
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from outscraper import OutscraperClient

# 1. Download required NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# 2. Initialize API and fetch reviews
client = OutscraperClient(api_key='ZXAjNzVkMGQ1ZjhjMmRhNDdlYWEyZGI3NTdlZjYwZjViYjR8ZTIzNWJjY2I3MA')

results = client.google_maps_business_reviews(
    'Westfield Washington Public Library, IN, USA',
    reviews_limit=0,    # Scrape all available reviews
    limit=1,
    sort='newest',
    ignore_empty=False,
)

reviews_list = []
for place in results:
    reviews_data = place.get('reviews_data', []) or []
    for review in reviews_data:
        text = review.get('review_text')
        if text and text.strip():
            reviews_list.append({'text': text})

print(f"Extracted {len(reviews_list)} text reviews for analysis.\n")

# 3. Setup Sentiment Analyzer and Category Dictionary
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Keywords defining each category
category_keywords = {
    'Product': ['facility', 'room', 'space', 'wifi', 'books', 'coffee', 'equipment', 'building', 'parking', 'desk', 'computer'],
    'Service': ['staff', 'librarian', 'help', 'friendly', 'patient', 'support', 'service', 'event', 'program', 'kids'],
    'Quality': ['great', 'clean', 'excellent', 'best', 'modern', 'top', 'wonderful', 'amazing', 'quiet', 'nice', 'dirty', 'bad']
}

# Storage for word frequencies: Category -> Positive/Negative -> Counter
category_counts = {
    'Product': {'positive': Counter(), 'negative': Counter()},
    'Service': {'positive': Counter(), 'negative': Counter()},
    'Quality': {'positive': Counter(), 'negative': Counter()}
}

# 4. Process Reviews: Sentiment Scoring + Word Frequency Analysis
for item in reviews_list:
    text = item['text']
    
    # Calculate Sentiment Score (-1.0 to +1.0)
    sentiment_score = sia.polarity_scores(text)['compound']
    
    if sentiment_score >= 0.05:
        sentiment_label = 'positive'
    elif sentiment_score <= -0.05:
        sentiment_label = 'negative'
    else:
        continue  # Skip neutral reviews for quadrant comparison

    # Tokenize words and remove stop words / short characters
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered_words = [w for w in words if w not in stop_words]

    # Assign words to matching categories
    for category, keywords in category_keywords.items():
        if any(kw in words for kw in keywords):
            category_counts[category][sentiment_label].update(filtered_words)

# 5. Generate and Export Side-by-Side Positive/Negative JPEG Charts
for category, sentiments in category_counts.items():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{category} Category - Positive vs. Negative Word Frequencies', fontsize=16, fontweight='bold')
    
    # Left Side: Top Positive Words
    top_pos = dict(sentiments['positive'].most_common(10))
    if top_pos:
        ax1.barh(list(top_pos.keys()), list(top_pos.values()), color='#2ecc71')
        ax1.set_title('Top Positive Words Count', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Frequency')
        ax1.invert_yaxis()
    else:
        ax1.text(0.5, 0.5, 'No Positive Reviews Found', ha='center', va='center')
        ax1.set_title('Top Positive Words')

    # Right Side: Top Negative Words
    top_neg = dict(sentiments['negative'].most_common(10))
    if top_neg:
        ax2.barh(list(top_neg.keys()), list(top_neg.values()), color='#e74c3c')
        ax2.set_title('Top Negative Words Count', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Frequency')
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'No Negative Reviews Found', ha='center', va='center')
        ax2.set_title('Top Negative Words')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 1. Set the file name
    output_filename = f'{category.lower()}_sentiment_quadrants.jpg'

    # 2. Save the figure FIRST
    plt.savefig(output_filename, format='jpeg', dpi=300)

    # 3. Close the figure AFTER saving
    plt.close()
    
    print(f"Exported chart: {output_filename}")