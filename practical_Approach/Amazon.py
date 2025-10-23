import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample Amazon product reviews
reviews = [
    "I love my new iPhone from Apple! The camera is amazing.",
    "Samsung Galaxy has terrible battery life. Very disappointed.",
    "The Sony headphones broke after one week of use.",
    "Microsoft Surface Pro is perfect for work and entertainment.",
    "Google Pixel camera quality is outstanding but expensive."
]

# Analyze each review
for review in reviews:
    doc = nlp(review)
    
    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Rule-based sentiment analysis
    positive_words = ['love', 'amazing', 'perfect', 'outstanding', 'good', 'great']
    negative_words = ['terrible', 'disappointed', 'broke', 'bad', 'expensive']
    
    sentiment = "neutral"
    if any(word in review.lower() for word in positive_words):
        sentiment = "positive"
    if any(word in review.lower() for word in negative_words):
        sentiment = "negative" if sentiment == "neutral" else "mixed"
    
    # Print results
    print(f"Review: {review}")
    print(f"Entities: {entities}")
    print(f"Sentiment: {sentiment}\n")