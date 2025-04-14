# Hospital Finder with Natural Language Input and Review Analysis (Improved Abstract Summary)


import nltk  # NLP library for text processing
nltk.download('stopwords') ## Download stopwords for text processing
nltk.download('punkt')       # Required for sentence/word tokenization
nltk.download('wordnet')
import requests ## HTTP library for API requests
from textblob import TextBlob ## Text processing library
import time ## Time library for delays
from fpdf import FPDF ## PDF generation library
import re ## Regular expressions for text processing
import html ## HTML processing library
from collections import Counter ## Counter for counting word frequencies
from nltk.corpus import stopwords ## Stopwords for text processing

GOOGLE_API_KEY = "AIzaSyCso3qJe805uSimr_SArx2NO4JT3lB6z3U"  # Replace with your actual key

stop_words = set(stopwords.words("english")) ## Set of English stopwords for text processing

# ===================== STEP 1: Get Coordinates using Google =====================
### Retrieves location coordinates using Google Geocoding API and sets up NLP and PDF libraries for text analysis and report generation
def get_coordinates(location_name):
    endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location_name, "key": GOOGLE_API_KEY}
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            location = data["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
    return None, None

# ===================== STEP 2: Get Hospitals =====================
# Fetches nearby hospitals within a specified radius using Google Places API based on given latitude and longitude
# Returns a list of hospital details if the API call is successful, otherwise returns an empty list
def get_nearby_hospitals(lat, lng, radius=5000):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius={radius}&type=hospital&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data.get("status") != "OK":
        return []
    return data.get("results", [])

# ===================== STEP 3: Get Hospital Reviews =====================
# Retrieves hospital details using Google Place Details API based on the given place_id
# Returns the list of reviews, overall rating, and hospital name; defaults to empty or placeholder values if not available
def get_hospital_reviews(place_id):
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=name,rating,reviews&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    result = response.json().get("result", {})
    return result.get("reviews", []), result.get("rating", 0), result.get("name", "Unknown")

# ===================== STEP 4: Extract Sentiment and Abstract Summary =====================
# Cleans summary text by decoding HTML entities, removing non-ASCII characters,
# reducing repeated punctuation, and normalizing whitespace
def clean_summary_text(text):
    text = html.unescape(text) # Convert HTML entities to normal characters (e.g., &amp; -> &)
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-ASCII characters
    text = re.sub(r"[.]{2,}", ".", text)  # Replace multiple periods with a single period
    text = re.sub(r"\s+", " ", text) # Replace multiple spaces or line breaks with a single space
    return text.strip()   # Trim leading and trailing whitespace

# Analyzes sentiment and extracts key noun tokens from a list of hospital reviews
def extract_sentiment(reviews):
    sentiments = [] # List to store sentiment scores
    noun_tokens = [] # List to store noun tokens
    for review in reviews: # Extracts sentiment and noun tokens from each review
        text = review.get("text", "") # Get the review text
        if text: # If the review text is not empty
            blob = TextBlob(text) # Create a TextBlob object for sentiment analysis
            sentiments.append(blob.sentiment.polarity) # Append sentiment score to the list
            words = [word.lower() for word in blob.words if word.isalpha() and word.lower() not in stop_words] # Filter out non-alphabetic words and stopwords
            noun_tokens.extend(words) # Append noun tokens to the list
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0 # Calculate average sentiment score

    if not noun_tokens:
        return avg_sentiment, "No summary available." # If no noun tokens are found, return average sentiment and a default message

    common = Counter(noun_tokens).most_common(5)  # Get the 5 most common noun tokens
    common_nouns = [word for word, freq in common] # Extract the words from the tuples

    summary = (
        f"Reviews often highlight aspects like {', '.join(common_nouns)}. "
        f"Overall sentiment from patients was {'positive' if avg_sentiment > 0 else 'mixed' if avg_sentiment == 0 else 'negative'}."
    )

    return avg_sentiment, clean_summary_text(summary)

# ===================== STEP 5: Parse User Preferences =====================
# Parses the user's natural language input to extract search preferences
def parse_user_input(user_input):
    blob = TextBlob(user_input) # Create a TextBlob object for potential NLP use (not actively used here)
    # Set minimum rating based on keywords: assume higher standards if user mentions "best" or "high"
    min_rating = 4.0 if "best" in user_input or "high" in user_input else 3.5
    # Set maximum acceptable wait time: prefer shorter wait if user says "fast" or "quick"
    max_wait_time = 30 if "fast" in user_input or "quick" in user_input else 999 # 999 acts as a default (no constraint)
    # Return user preferences as a dictionary
    return {
        "min_rating": min_rating,
        "max_wait_time": max_wait_time
    }

# ===================== STEP 6: Rank All Hospitals =====================
# Ranks hospitals based on a weighted score of user preferences, ratings, and sentiment scores
def rank_hospitals(hospitals_data, user_preferences):
    ranked = []  # List to hold hospitals that meet the minimum rating and their calculated score
    for hospital in hospitals_data:
        rating = hospital.get("rating", 0) # Get hospital rating (default to 0 if missing)
        sentiment = hospital.get("sentiment", 0) # Get sentiment score (default to 0 if missing)
        # Filter hospitals based on user's minimum rating preference
        if rating >= user_preferences["min_rating"]:
            # Calculate a weighted score: 60% rating, 40% sentiment
            score = (rating * 0.6) + (sentiment * 0.4)
            # Append hospital details and score to the ranked list
            ranked.append({"name": hospital["name"], "score": score, "rating": rating, "sentiment": sentiment, "summary": hospital["summary"]})
    # Sort hospitals by score in descending order (highest-ranked first)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked # Return the sorted list of ranked hospitals

# ===================== STEP 7: Generate PDF =====================
# Generates a PDF report summarizing hospital insights based on location, recommendations, and review analysis
def generate_pdf_report(location, total_count, top5, all_hospitals):
    pdf = FPDF()  # Initialize a new PDF document
    pdf.add_page()  # Add a page to the PDF

    # Set main header font and write the title of the report with the specified location
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Hospital Insights Report - {location}", ln=True)

    # Add total hospital count
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Total hospitals found: {total_count}", ln=True)
    pdf.ln(5)  # Add vertical space

    # Section: Top 5 recommended hospitals based on rating & sentiment
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Top 5 Recommended Hospitals:", ln=True)

    pdf.set_font("Arial", "", 12)
    for i, h in enumerate(top5, 1):
        # Format each hospital’s name, rating, sentiment, score, and summary
        text = f"{i}. {h['name']} - Rating: {h['rating']}, Sentiment: {round(h['sentiment'], 2)}, Score: {round(h['score'], 2)}\n   Summary: {h['summary']}"

        # Use multi_cell to allow text wrapping and handle special characters
        pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'))

    pdf.ln(5)  # Add vertical space

    # Section: Full list of all hospitals processed
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "All Hospitals Processed:", ln=True)

    pdf.set_font("Arial", "", 11)
    for h in all_hospitals:
        # Format full details for each hospital
        full_text = f"- {h['name']}\n  Rating: {h['rating']}, Sentiment: {round(h['sentiment'], 2)}, Score: {round(h['score'], 2)}\n  Summary: {h['summary']}"

        # Ensure special characters are handled and display text in wrapped lines
        pdf.multi_cell(0, 8, full_text.encode('latin-1', 'replace').decode('latin-1'))

    # Save the final PDF report to file
    pdf.output("hospital_report.pdf")


# ===================== MAIN FLOW =====================
# Main function to drive the hospital search, analysis, and reporting process
def main():
    # Prompt user for location and hospital preferences
    location_input = input("Enter a location (e.g., 'Los Angeles, CA'): ")
    user_query = input("Describe what you're looking for in a hospital: ")

    # Parse user preferences from input query (e.g., min_rating, wait time)
    user_preferences = parse_user_input(user_query)

    # Get coordinates (latitude & longitude) for the given location
    lat, lng = get_coordinates(location_input)
    if not lat:
        print("Could not find the location.")
        return  # Exit if location is invalid or not found

    # Retrieve hospitals near the specified coordinates using Google Places API
    hospitals = get_nearby_hospitals(lat, lng)
    if not hospitals:
        print("\nNo hospitals found for the given location. Please check the location input.")
        return  # Exit if no hospitals are found

    print(f"\nFound {len(hospitals)} hospitals within your search radius. Analyzing reviews...\n")

    # Enrich each hospital with reviews, sentiment score, and a summary
    enriched_hospitals = []
    for h in hospitals:
        place_id = h.get("place_id")
        reviews, rating, name = get_hospital_reviews(place_id)
        sentiment, summary = extract_sentiment(reviews)
        enriched_hospitals.append({
            "name": name,
            "rating": rating,
            "sentiment": sentiment,
            "summary": summary
        })
        time.sleep(1)  # Delay to respect API rate limits

    # Rank hospitals based on user preferences, rating, and sentiment
    ranked_hospitals = rank_hospitals(enriched_hospitals, user_preferences)
    top5 = ranked_hospitals[:5]  # Get top 5 hospitals

    # Display the top recommended hospital to the user
    if top5:
        print(f"\nRecommended Top Hospital: {top5[0]['name']}")
        print(f"Rating: {top5[0]['rating']}")
        print(f"Sentiment: {round(top5[0]['sentiment'], 2)}")
        print(f"Overall Score: {round(top5[0]['score'], 2)}")
        print(f"Summary: {top5[0]['summary']}")
    else:
        print("\nNo hospital matched your preferences.")

    # Generate a PDF report with hospital insights and rankings
    generate_pdf_report(location_input, len(hospitals), top5, ranked_hospitals)
    print("\nPDF report 'hospital_report.pdf' generated successfully!")

# Entry point for the script
if __name__ == "__main__":
    main()