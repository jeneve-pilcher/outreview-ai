#79 reviews
from outscraper import OutscraperClient
client = OutscraperClient(api_key='ZXAjNzVkMGQ1ZjhjMmRhNDdlYWEyZGI3NTdlZjYwZjViYjR8ZTIzNWJjY2I3MA')
# Fetch business reviews by name, location, or Google Maps URL
results = client.google_maps_business_reviews(
    'Westfield Washington Public Library, IN, USA',
    reviews_limit=0,    # 0 tells Outscraper to scrape ALL available historical reviews
    limit=1,
    sort='newest',      # Chronological ordering forces systematic pagination
    ignore_empty=False,
)
total_reviews_count = 0
text_reviews_count = 0
reviews_list = []

for place in results:
    print(f"\nProcessing Location: {place.get('name')}")
    reviews_data = place.get('reviews_data', [])
    
    # 1. Count total reviews fetched from the API
    total_reviews_count += len(reviews_data)
    
    for review in reviews_data:
        text = review.get('review_text')
        
        # 2. Check if the review has actual written text
        if text and text.strip():
            text_reviews_count += 1
            reviews_list.append({
                'text': text
            })

# Output your target counts
print("\n--- Summary ---")
print(f"Total Reviews:     {total_reviews_count}")
print(f"Reviews with Text: {text_reviews_count}")

# Output all of the extracted text reviews
print("\n--- Text Reviews ---")
for index, item in enumerate(reviews_list, start=1):
    print(f"{index}. {item['text']}")

