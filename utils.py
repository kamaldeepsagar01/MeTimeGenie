# This is the complete utils.py file

import datetime
import random
import pandas as pd
import openrouteservice
import googlemaps
import requests
import streamlit as st
import google.generativeai as genai
import os
import json
from pathlib import Path
import traceback
import logging
import re
from PIL import UnidentifiedImageError
import openai

# Make sure to set OPENAI_API_KEY in your environment or replace here.
#openai.api_key = os.getenv("OPENAI_API_KEY", "your-default-api-key")

UNSPLASH_ACCESS_KEY = "rVvxvkYuJREpI8wMn9GvJUGhj5bZVlVFBkKMx1QquQA"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('activity_suggester')

# Custom exception classes
class AppError(Exception):
    """Base exception class for application errors"""
    def __init__(self, message, error_type="general", original_exception=None):
        self.message = message
        self.error_type = error_type
        self.original_exception = original_exception
        super().__init__(self.message)

class APIError(AppError):
    """Exception raised for errors in the API calls"""
    def __init__(self, message, api_name, original_exception=None):
        super().__init__(message, f"api_{api_name.lower()}", original_exception)
        self.api_name = api_name

class LLMError(AppError):
    """Exception raised for errors in LLM processing"""
    def __init__(self, message, original_exception=None):
        super().__init__(message, "llm_error", original_exception)

class ImageError(AppError):
    """Exception raised for errors in image handling"""
    def __init__(self, message, original_exception=None):
        super().__init__(message, "image_error", original_exception)

def safe_api_call(func):
    """Decorator for safely calling API functions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            api_name = func.__name__
            error_msg = f"Error in {api_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise APIError(error_msg, api_name, e)
    return wrapper

# Set up clients
def init_clients(openroute_api_key, google_maps_api_key):
    ors_client = openrouteservice.Client(key=openroute_api_key)
    gmaps_client = googlemaps.Client(key=google_maps_api_key)
    return ors_client, gmaps_client

# Generate synthetic user context
def get_synthetic_user():
    # This is a placeholder function that returns synthetic user data
    # In a real app, you would get this data from the user's actual context
    """
    Return synthetic user data with automatically calculated free hours
    based on calendar and current time.
    """
    # Define the user's base information
    user_data = {
        "location": {
            "city": "Bangalore",
            "lat": 12.9716,
            "lon": 77.5946
        },
        "weather": "Rainy",
        "current_time": "Saturday 8 AM",
        #"free_hours": 4,
        "calendar": [
            {"event": "Lunch with friend", "start": "1 PM", "end": "2 PM"},
            {"event": "Office Meeting", "start": "6 PM", "end": "7 PM"}
        ],
        "interests": {
            "travel": 0.91,
            "food": 0.18,
            "news": 0.15,
            "shopping": 0.13,
            "gaming": 0.24
        }
    }
    
    # Calculate free hours based on current time and calendar
    user_data["free_hours"] = calculate_free_time(
        user_data["current_time"], 
        user_data["calendar"]
    )
    
    return user_data

def extract_main_keywords(text):
    """
    Extract main keywords from indoor activity description
    """
    try:
        # If the text is empty or None, return a generic keyword
        if not text:
            return "indoor activity"

        # List of food and activity keywords to look for
        food_keywords = [
            "dosa", "cooking", "baking", "food", "recipe", "cuisine", "dish",
            "meal", "restaurant", "café", "bakery", "pizza", "burger", "pasta",
            "sushi", "curry", "breakfast", "lunch", "dinner", "snack",
            "dessert", "coffee", "tea", "smoothie", "cocktail", "pasta", "truffles"
        ]

        activity_keywords = [
            "yoga", "meditation", "painting", "drawing", "art", "craft",
            "reading", "book", "game", "gaming", "movie", "film", "music",
            "dance", "workout", "exercise", "pottery", "chess", "board game",
            "puzzle", "knitting", "photography", "baking", "cooking"
        ]

        # Combine all keywords
        all_keywords = food_keywords + activity_keywords

        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()

        # Find matching keywords
        matches = []
        for keyword in all_keywords:
            if keyword in text_lower:
                matches.append(keyword)

        # If we found any matches, return the longest one (likely most specific)
        if matches:
            return max(matches, key=len)

        # If no specific matches, use regex to find nouns (imperfect but useful fallback)
        words = re.findall(r'\b[A-Za-z]{4,}\b', text)
        if words:
            # Return the longest word as a fallback
            return max(words, key=len)

        # Last resort
        return "indoor activity"

    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return "indoor activity"  # Fallback

def extract_keywords_from_prompt(prompt):
    """
    Extract the most relevant keywords from an activity description for image search
    """
    try:
        # If we have OpenAI access, use it for best results
        if openai.api_key and openai.api_key != "Test":
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "Extract 3 important keywords for image search from the text, ordered from most to least specific. Return only the keywords separated by commas, nothing else.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=30,
                )
                extracted_text = response["choices"][0]["message"]["content"].strip()
                keywords = [kw.strip() for kw in extracted_text.split(',')]
                return keywords
            except Exception as e:
                logging.warning(f"OpenAI extraction failed: {e}")
                # Continue to fallback methods
        
        # Fallback 1: Pattern-based extraction for specific activities
        # Look for bold or asterisk-emphasized text which often contains the activity name
        emphasized = re.findall(r'\*\*(.*?)\*\*|\*(.*?)\*', prompt)
        emphasized_keywords = []
        for match_pair in emphasized:
            # Each match is a tuple with one empty value
            for match in match_pair:
                if match:
                    emphasized_keywords.append(match)
        
        if emphasized_keywords:
            return emphasized_keywords + extract_food_keywords(prompt)
        
        # Fallback 2: Look for food/activity specific patterns
        food_keywords = extract_food_keywords(prompt)
        if food_keywords:
            return food_keywords
            
        # Fallback 3: Extract phrases that might be activities
        activity_phrases = re.findall(r'(making|cooking|baking|playing|watching|trying) ([a-zA-Z\s]+)', prompt)
        if activity_phrases:
            return [f"{verb} {obj}" for verb, obj in activity_phrases[:2]] + extract_nouns(prompt)[:1]
        
        # Final fallback: Just extract potential nouns
        return extract_nouns(prompt)
        
    except Exception as e:
        logging.error(f"All keyword extraction methods failed: {str(e)}")
        # Last resort - split by spaces and take longest words (likely nouns)
        words = prompt.split()
        words.sort(key=len, reverse=True)
        return words[:3] if words else ["activity"]

def extract_food_keywords(text):
    """Extract food-related keywords which are common in indoor activities"""
    # Common food patterns
    food_patterns = [
        r"(?:making|cooking|baking|prepare|preparing|homemade) ([a-zA-Z\s]+)",  # cooking X
        r"(?:make|cook|bake|try) ([a-zA-Z\s]+)",  # make X
        r"([a-zA-Z\s]+) recipe",  # X recipe
        r"([a-zA-Z\s]+) from scratch"  # X from scratch
    ]
    
    matches = []
    for pattern in food_patterns:
        found = re.findall(pattern, text.lower())
        if found:
            matches.extend(found)
    
    # Clean up matches
    cleaned = []
    for match in matches:
        # Remove articles and filler words
        for word in ["a", "the", "some", "your", "own"]:
            match = re.sub(r'\b' + word + r'\b', '', match)
        match = re.sub(r'\s+', ' ', match).strip()
        if match and len(match) > 3:
            cleaned.append(match)
    
    return cleaned[:3] if cleaned else []

def extract_nouns(text):
    """Extract potential nouns from text"""
    # Simple regex-based noun extraction
    # Words that are capitalized or 4+ characters and not in stop list
    stop_words = ["that", "this", "with", "from", "your", "have", "will", "what", 
                 "about", "which", "when", "make", "like", "how", "can", "time",
                 "just", "being", "some", "take", "into", "spicy", "delicious", "easy"]
    
    # First look for noun phrases
    noun_phrases = re.findall(r'([a-zA-Z]{3,}(?:\s+[a-zA-Z]{3,}){1,2})', text)
    
    # Then individual potential nouns
    words = re.findall(r'\b[A-Za-z]{4,}\b', text)
    
    # Filter and combine
    result = []
    
    for phrase in noun_phrases:
        if all(word.lower() not in stop_words for word in phrase.split()):
            result.append(phrase)
    
    for word in words:
        if word.lower() not in stop_words:
            result.append(word)
    
    # Deduplicate and limit
    unique_results = []
    for item in result:
        if item not in unique_results:
            unique_results.append(item)
    
    return unique_results[:3]

@safe_api_call
def fetch_unsplash_image(keyword):
    """
    Fetch an image from Unsplash API for a given keyword with improved reliability
    """
    try:
        # Check if we have an Unsplash API key in the environment or secrets
        access_key = os.environ.get("UNSPLASH_ACCESS_KEY") or st.secrets.get("UNSPLASH_ACCESS_KEY", UNSPLASH_ACCESS_KEY)
        
        # Simplify the keyword to improve hit rate
        original_keyword = keyword
        simplified_keyword = simplify_keyword(keyword)
        core_keyword = extract_core_keyword(keyword)
        
        # List of keywords to try, in order of specificity
        keywords_to_try = [
            simplified_keyword,
            core_keyword,
            # Add some category-specific generic terms
            f"{core_keyword} activity",
            original_keyword
        ]
        
        # Remove duplicates while preserving order
        unique_keywords = []
        for kw in keywords_to_try:
            if kw and kw not in unique_keywords:
                unique_keywords.append(kw)
        
        # Log the keywords we'll try
        logging.info(f"Trying Unsplash with keywords: {unique_keywords}")
        
        # Try each keyword with the API method first
        if access_key:
            for kw in unique_keywords:
                try:
                    response = requests.get(
                        "https://api.unsplash.com/search/photos",
                        params={
                            "query": kw,
                            "client_id": access_key,
                            "per_page": 3
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data["results"] and len(data["results"]) > 0:
                            # Return the regular sized image URL
                            image_url = data["results"][0]["urls"]["regular"]
                            logging.info(f"Found Unsplash image with API for '{kw}'")
                            return image_url
                except Exception as api_err:
                    logging.warning(f"API method failed for '{kw}': {api_err}")
                    continue
        
        # Fall back to the direct URL method which is more reliable
        for kw in unique_keywords:
            try:
                sanitized_keyword = kw.replace(" ", "+")
                direct_url = f"https://source.unsplash.com/1600x900/?{sanitized_keyword}"
                
                # Check if URL returns a valid image
                response = requests.head(direct_url, allow_redirects=True)
                if response.status_code == 200:
                    logging.info(f"Found Unsplash image with direct URL for '{kw}'")
                    return direct_url
            except Exception as direct_err:
                logging.warning(f"Direct URL method failed for '{kw}': {direct_err}")
                continue
        
        # Final fallback to a very generic term
        return "https://source.unsplash.com/1600x900/?activity"
            
    except Exception as e:
        logger.error(f"All Unsplash methods failed for '{keyword}': {str(e)}")
        # Ultimate fallback
        return "https://source.unsplash.com/1600x900/?activity"

def simplify_keyword(keyword):
    """
    Simplify a complex keyword phrase to increase hit rate with image APIs
    """
    # If keyword is already simple, return as is
    if len(keyword.split()) <= 2:
        return keyword
        
    # Remove common phrases that make keywords too specific
    removable_phrases = [
        "from scratch", "homemade", "a batch of", "watching", "making", "cooking",
        "baking", "playing", "trying", "batch of", "going to", "session of"
    ]
    
    result = keyword.lower()
    for phrase in removable_phrases:
        result = result.replace(phrase, "")
    
    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def extract_core_keyword(keyword):
    """
    Extract the core subject from a keyword phrase
    Example: "batch of homemade pasta from scratch" -> "pasta"
    """
    # List of common subjects
    common_subjects = [
        "pasta", "pizza", "movie", "film", "game", "book", "yoga", "meditation",
        "painting", "drawing", "music", "coffee", "tea", "cake", "cookie", "bread",
        "soup", "salad", "dessert", "craft", "puzzle", "chess", "board game"
    ]
    
    # First check if any common subject is in the keyword
    keyword_lower = keyword.lower()
    for subject in common_subjects:
        if subject in keyword_lower:
            return subject
    
    # If not found, just take the last word (often the main subject)
    words = keyword_lower.split()
    if words:
        # Remove common modifiers if they're the last word
        if words[-1] in ["recipe", "activity", "project", "session"]:
            if len(words) > 1:
                return words[-2]
        return words[-1]
    
    return keyword  # Fallback to original

@safe_api_call
def fetch_google_images(keyword, GOOGLE_CSE_ID, GOOGLE_API_KEY):
    """
    Fetch images using Google Custom Search API as a more robust alternative
    """
    try:
        if not GOOGLE_CSE_ID or not GOOGLE_API_KEY:
            logger.warning("Missing Google CSE credentials")
            return None
            
        # Simplify the keyword to improve hit rate
        simplified_keyword = simplify_keyword(keyword)
            
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': simplified_keyword,
            'cx': GOOGLE_CSE_ID,
            'key': GOOGLE_API_KEY,
            'searchType': 'image',
            'num': 1
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'items' in data and len(data['items']) > 0:
                return data['items'][0]['link']
        else:
            logger.warning(f"Google CSE error: {response.status_code}")
            
        return None
    except Exception as e:
        logger.error(f"Error fetching Google image for '{keyword}': {str(e)}")
        return None


@safe_api_call
def fetch_image_for_keyword(keyword, GOOGLE_MAPS_API_KEY, GOOGLE_CSE_ID=None, GOOGLE_CSE_API_KEY=None):
    """
    Fetch an image for a specific keyword using multiple services with improved fallbacks
    """
    try:
        if not keyword:
            return None
        
        logging.info(f"Fetching image for keyword: {keyword}")
        original_keyword = keyword
        simplified_keyword = simplify_keyword(keyword)
        core_keyword = extract_core_keyword(keyword)
        
        logging.info(f"Original: '{original_keyword}', Simplified: '{simplified_keyword}', Core: '{core_keyword}'")

        # Try Unsplash first for indoor activities
        logging.info(f"Trying Unsplash with original keyword: {original_keyword}")
        unsplash_url = fetch_unsplash_image(original_keyword)
        if unsplash_url:
            logging.info("Got image from Unsplash with original keyword")
            return unsplash_url
            
        # Try simplified keyword
        logging.info(f"Trying Unsplash with simplified keyword: {simplified_keyword}")
        unsplash_url = fetch_unsplash_image(simplified_keyword)
        if unsplash_url:
            logging.info("Got image from Unsplash with simplified keyword")
            return unsplash_url
            
        # Try core keyword
        logging.info(f"Trying Unsplash with core keyword: {core_keyword}")
        unsplash_url = fetch_unsplash_image(core_keyword)
        if unsplash_url:
            logging.info("Got image from Unsplash with core keyword")
            return unsplash_url

        # Only if Unsplash fails, try Google Maps API
        try:
            gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

            # Search for places related to the keyword
            places_result = gmaps.places(
                query=simplified_keyword,
                language="en",
            )

            places_with_photos = [place for place in places_result.get("results", [])
                              if place.get("photos")]

            if places_with_photos:
                # Select a random place with photos
                selected_place = random.choice(places_with_photos)

                # Get the photo reference
                photo_reference = selected_place["photos"][0]["photo_reference"]

                # Build the URL for the photo
                image_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_MAPS_API_KEY}"
                
                logging.info("Got image from Google Places API")
                return image_url
        except Exception as e:
            logging.warning(f"Google Places image fetch failed: {str(e)}")
        
        # Try Google Custom Search API if available
        if GOOGLE_CSE_ID and GOOGLE_CSE_API_KEY:
            try:
                google_image = fetch_google_images(original_keyword, GOOGLE_CSE_ID, GOOGLE_CSE_API_KEY)
                if google_image:
                    logging.info("Got image from Google Custom Search API")
                    return google_image
                    
                # Try with simplified keyword
                google_image = fetch_google_images(simplified_keyword, GOOGLE_CSE_ID, GOOGLE_CSE_API_KEY)
                if google_image:
                    logging.info("Got image from Google Custom Search API with simplified keyword")
                    return google_image
                    
                # Try with core keyword
                google_image = fetch_google_images(core_keyword, GOOGLE_CSE_ID, GOOGLE_CSE_API_KEY)
                if google_image:
                    logging.info("Got image from Google Custom Search API with core keyword")
                    return google_image
            except Exception as e:
                logging.warning(f"Google CSE image fetch failed: {str(e)}")
            
        return None

    except Exception as e:
        logging.error(f"All image fetching methods failed for keyword '{keyword}': {str(e)}")
        return None

def calculate_interest_adjustments(prefs):
    """Calculate interest adjustments based on user feedback"""
    # Count feedback by category
    category_counts = {}
    for feedback in prefs["feedback_history"]:
        category = feedback["category"]
        feedback_type = feedback["type"]
        
        if category not in category_counts:
            category_counts[category] = {"like": 0, "dislike": 0, "view_details": 0}
            
        if feedback_type in category_counts[category]:
            category_counts[category][feedback_type] += 1
    
    # Calculate adjustments
    adjustments = {}
    for category, counts in category_counts.items():
        # Simple formula: likes + (views * 0.3) - dislikes
        score = counts.get("like", 0) + (counts.get("view_details", 0) * 0.3) - counts.get("dislike", 0)
        adjustments[category] = min(max(score * 0.1, -0.5), 0.5)  # Limit adjustment between -0.5 and 0.5
    
    prefs["interest_adjustments"] = adjustments

def get_adjusted_interests(user):
    """Get user interests adjusted by feedback"""
    prefs = get_user_preferences_db()
    updated_scores = prefs.get("category_preferences", {})
    if updated_scores:
        return updated_scores

    # Fallback to original interests from user
    return user.get("interests", {})





def top_activity_interest_llm(user):
    model = st.session_state.model
    
    # Get adjusted interests based on feedback
    adjusted_interests = get_adjusted_interests(user)
    
    prompt = f"""
    You are a smart assistant that ranks user interests in the context of the moment.

    User Context:
    - City: {user['location']['city']}
    - Weather: {user['weather']}
    - Current Time: {user['current_time']}
    - Free Hours: {user['free_hours']}
    - Interests (with scores): {adjusted_interests}

    Based on this context, rank the categories from most to least relevant **for recommending an activity right now**.

    Return a ranked list like this:
    1. travel
    2. gaming
    3. shopping
    ...

    Only return the list — no explanations.
    """
    response = model.generate_content(prompt)
    ranked_categories = response.text.strip().split('\n')
    top_interest = ranked_categories[0].split(".")[1].strip()
    return top_interest


def build_llm_decision_prompt(user, top_interest):
    """
    Build a prompt to decide between indoor and outdoor activity
    """
    weather = user.get("weather", "Unknown")
    time = user.get("current_time", "Unknown")
    
    prompt = f"""
    Based on this context, decide if I should suggest an indoor or outdoor activity.
    Just respond with "indoor" or "outdoor".
    
    User context:
    - Current weather: {weather}
    - Current time: {time}
    - Their top interest: {top_interest}
    - Free hours: {user.get("free_hours", "Unknown")}
    - Location: {user['location']['city']}
    
    Consider:
    - If it's late evening, raining, or very hot, indoor might be better
    - If it's morning or daytime with good weather, outdoor might be better
    - Also consider the interest - some activities like gaming are typically indoor
    - Also consider the location of the user
    """
    return prompt.strip()

def build_llm_prompt_indoor(user, top_interest, user_feedback=None):
    """
    Build a prompt for indoor activity suggestion
    """
    # Include user feedback if available
    feedback_note = "" if not user_feedback else f"{user_feedback} "
    
    prompt = f"""
    {feedback_note}Suggest a specific indoor activity related to {top_interest} that I can do at home or nearby.
    
    My context:
    - Current time: {user.get("current_time", "Unknown")}
    - I have {user.get("free_hours", "Unknown")} free hours
    - My top interest right now: {top_interest}
    - My city right now: {user['location']['city']}

    ❗ Choose only one activity. Do not list or compare options. 
    Make your response in 1-2 short, fun, personal sentences that help me decide what to do right now.
    Be specific and practical. Recommend something realistic, not generic. Your output would be displayed on the lockscreen of the users phone
    """
    return prompt.strip()

@safe_api_call
def fetch_places(user, interest_type, api_key):
    """
    Fetch places from Google Maps Places API based on user context and interest
    """
    try:
        gmaps = googlemaps.Client(key=api_key)
        
        # Get location from user
        lat = user.get('location', {}).get('lat')
        lon = user.get('location', {}).get('lon')
        
        if not lat or not lon:
            logger.warning("Missing user location coordinates")
            return []
        
        # Map interest types to Google Maps place types
        place_type_mapping = {
            'food': 'restaurant',
            'shopping': 'shopping_mall',
            'travel': 'tourist_attraction',
            'news': 'library',
            'gaming': 'amusement_park',
            'cooking': 'kitchen'
        }
        
        # Get place type from interest
        place_type = place_type_mapping.get(interest_type, 'point_of_interest')
        
        # Search for places
        places_result = gmaps.places_nearby(
            location=(lat, lon),
            radius=20000,  # 20km radius
            type=place_type,
            open_now=True
        )
        
        return places_result.get('results', [])
    except Exception as e:
        logger.error(f"Error fetching places: {str(e)}")
        return []

def build_personalized_context(user, top_interest):
    """
    Build personalized context string based on user preferences
    """
    try:
        # Get user preferences from database (or default to empty)
        prefs = get_user_preferences_db()
        
        context = []
        
        # Add information about category preferences
        if prefs["category_preferences"]:
            top_categories = sorted(
                prefs["category_preferences"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            categories_text = ", ".join([f"{cat} ({score:.1f})" for cat, score in top_categories])
            context.append(f"Top categories: {categories_text}")
        
        # Add information about liked places
        if prefs["liked_places"]:
            recent_likes = [item['name'] for item in prefs["liked_places"][-3:]]
            context.append(f"Recently liked: {', '.join(recent_likes)}")
        
        # Add information about disliked places
        if prefs["disliked_places"]:
            recent_dislikes = [item['name'] for item in prefs["disliked_places"][-3:]]
            context.append(f"Recently disliked: {', '.join(recent_dislikes)}")
        
        # Return the combined context
        if context:
            return "\n- " + "\n- ".join(context)
        return "No preference history available."
    except Exception as e:
        logger.error(f"Error building personalized context: {str(e)}")
        return "No preference history available."

@safe_api_call
def get_route_duration(origin, destination, ors_client):
    """
    Get the route duration between two points using OpenRouteService
    Returns time in minutes
    """
    try:
        # Make sure coordinates are valid
        if not all(origin) or not all(destination):
            return None
        
        # Request route from ORS API
        route = ors_client.directions(
            coordinates=[origin, destination],
            profile='driving-car',
            format='geojson'
        )
        
        # Extract duration in seconds and convert to minutes
        if route and 'features' in route and len(route['features']) > 0:
            duration_seconds = route['features'][0]['properties']['summary']['duration']
            return round(duration_seconds / 60)  # Convert to minutes
        
        return None
    except Exception as e:
        logger.error(f"Error getting route duration: {str(e)}")
        return None

@safe_api_call
def fetch_place_image(place, api_key):
    """
    Fetch an image for a place using Google Places API
    """
    try:
        if not place or 'photos' not in place or not place['photos']:
            return None
        
        # Get the photo reference
        photo_reference = place['photos'][0]['photo_reference']
        
        # Build the URL for the photo
        image_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={api_key}"
        
        return image_url
    except Exception as e:
        logger.error(f"Error fetching place image: {str(e)}")
        return None

def get_detailed_suggestion(user, model, short_description, interest_type, recommendation_data=None):
    """
    Get detailed information about a suggestion
    """
    try:
        prompt = f"""
        Please provide more detailed information about this activity suggestion:
        "{short_description}"
        
        The user's main interest is: {interest_type}
        Current time: {user.get("current_time", "Unknown")}
        Free hours: {user.get("free_hours", "Unknown")}
        
        Provide 5-6 sentences with:
        1. More details about this specific activity
        2. Why it's a good fit for the user now
        3. Specific things to look for or enjoy
        
        Be specific, practical and personal. Make it sound exciting but realistic.
        """
        
        response = model.generate_content(prompt)
        detailed_response = response.text.strip()
        
        # If it's an outdoor activity and we have place data, add Google Maps link
        maps_link = ""
        if recommendation_data and recommendation_data.get("type") == "outdoor" and recommendation_data.get("place"):
            place = recommendation_data.get("place")
            if place.get("place_id"):
                place_id = place.get("place_id")
                place_name = place.get("name", "Location")
                maps_link = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
                
                # Create a button to open Google Maps
                maps_html = f"""
                <div style="margin-top: 20px; margin-bottom: 20px;">
                    <h4>📍 Map Location</h4>
                    <a href="{maps_link}" target="_blank">
                        <button style="background-color: #4285F4; color: white; padding: 10px 15px; 
                        border: none; border-radius: 5px; cursor: pointer;">
                            Open {place_name} in Google Maps
                        </button>
                    </a>
                </div>
                """
                return detailed_response, maps_html
        
        return detailed_response, ""
    except Exception as e:
        logging.error(f"Error getting detailed suggestion: {str(e)}")
        return "I'm sorry, I couldn't generate additional details right now.", ""

# User preferences functions
def get_user_preferences_db():
    """
    Get user preferences from "database" (session state in this MVP)
    """
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {
            "category_preferences": {
                "food": 0.8,
                "travel": 0.5,
                "shopping": 0.5,
                "gaming": 0.5,
                "news": 0.5,
                "fitness": 0.7,
                "cooking": 0.9
            },
            "liked_places": [],
            "disliked_places": []
        }
    
    return st.session_state.user_preferences

def update_preferences_from_feedback(feedback_type, item_data):
    """
    Update user preferences based on feedback
    """
    prefs = get_user_preferences_db()
    
    # Add to liked or disliked places
    if feedback_type == "like":
        prefs["liked_places"].append(item_data)
        
        # Increase category preference
        category = item_data.get("type", "")
        if category in prefs["category_preferences"]:
            prefs["category_preferences"][category] = min(
                1.0, prefs["category_preferences"][category] + 0.1
            )
            
    elif feedback_type == "dislike":
        prefs["disliked_places"].append(item_data)
        
        # Decrease category preference
        category = item_data.get("type", "")
        if category in prefs["category_preferences"]:
            prefs["category_preferences"][category] = max(
                0.1, prefs["category_preferences"][category] - 0.1
            )
            
    elif feedback_type == "view_details":
        # Slightly increase category preference when viewing details
        category = item_data.get("type", "")
        if category in prefs["category_preferences"]:
            prefs["category_preferences"][category] = min(
                1.0, prefs["category_preferences"][category] + 0.05
            )
    
    # Trim lists if they get too long
    if len(prefs["liked_places"]) > 20:
        prefs["liked_places"] = prefs["liked_places"][-20:]
    if len(prefs["disliked_places"]) > 20:
        prefs["disliked_places"] = prefs["disliked_places"][-20:]
    
    # Save back to session state
    st.session_state.user_preferences = prefs

# Add these functions to utils.py

def get_suggestion_history():
    """
    Retrieve suggestion history from session state
    """
    if "suggestion_history" not in st.session_state:
        st.session_state.suggestion_history = {
            "indoor": [],  # List of indoor suggestions
            "outdoor": [],  # List of outdoor place IDs
            "total_shown": 0  # Total count of suggestions shown
        }
    
    return st.session_state.suggestion_history

def is_duplicate_suggestion(suggestion_text, suggestion_type):
    """
    Check if a suggestion has been shown before
    
    Args:
        suggestion_text: The text of the suggestion or place_id for outdoor places
        suggestion_type: Either "indoor" or "outdoor"
        
    Returns:
        Boolean: True if it's a duplicate, False otherwise
    """
    history = get_suggestion_history()
    
    # For indoor activities, we check the actual text
    if suggestion_type == "indoor":
        # Normalize the text for comparison
        normalized_text = suggestion_text.lower().strip()
        
        # Check for exact matches
        for past_suggestion in history["indoor"]:
            past_normalized = past_suggestion.lower().strip()
            if past_normalized == normalized_text:
                return True
            
            # Also check for high similarity (suggestion with the same core activity)
            # We consider >70% word overlap as a duplicate
            words1 = set(past_normalized.split())
            words2 = set(normalized_text.split())
            if words1 and words2:  # Avoid division by zero
                overlap = len(words1.intersection(words2)) / min(len(words1), len(words2))
                if overlap > 0.7:
                    return True
    
    # For outdoor, we check place_id
    elif suggestion_type == "outdoor" and suggestion_text:
        return suggestion_text in history["outdoor"]
    
    return False

def add_to_suggestion_history(suggestion, suggestion_type, place_id=None):
    """
    Add a suggestion to the history
    
    Args:
        suggestion: The suggestion text (for indoor) or place info (for outdoor)
        suggestion_type: Either "indoor" or "outdoor"
        place_id: The Google Place ID (for outdoor activities)
    """
    history = get_suggestion_history()
    
    if suggestion_type == "indoor":
        # Keep only the most recent suggestions (limit to 20)
        if len(history["indoor"]) >= 20:
            history["indoor"] = history["indoor"][1:]  # Remove oldest
        history["indoor"].append(suggestion)
    
    elif suggestion_type == "outdoor" and place_id:
        # Keep track of up to 50 place IDs to avoid repeating
        if len(history["outdoor"]) >= 50:
            history["outdoor"] = history["outdoor"][1:]  # Remove oldest
        history["outdoor"].append(place_id)
    
    # Increment total suggestions counter
    history["total_shown"] += 1
    
    # Update session state
    st.session_state.suggestion_history = history

def get_llm_prompt_with_history(base_prompt, suggestion_type):
    """
    Enhance a base LLM prompt with history information
    
    Args:
        base_prompt: The original prompt
        suggestion_type: Either "indoor" or "outdoor"
        
    Returns:
        Enhanced prompt with history information
    """
    history = get_suggestion_history()
    
    if suggestion_type == "indoor" and history["indoor"]:
        # Add the last 3 indoor suggestions to avoid repetition
        recent_indoor = history["indoor"][-3:]
        history_note = "\nPreviously suggested indoor activities (DO NOT suggest these again):\n"
        history_note += "\n".join([f"- {item}" for item in recent_indoor])
        history_note += "\n\nPlease suggest something DIFFERENT from these previous recommendations."
        
        # Add the history note before the last paragraph to keep the final instructions intact
        lines = base_prompt.split("\n")
        if len(lines) >= 2:
            return "\n".join(lines[:-2]) + history_note + "\n\n" + "\n".join(lines[-2:])
        else:
            return base_prompt + history_note
    
    return base_prompt

def calculate_free_time(current_time_str, calendar_events, max_hours=6):
    """
    Calculate free time until next calendar event, with a maximum limit.
    
    Args:
        current_time_str: String representing current time (e.g., "Friday 2:30 PM")
        calendar_events: List of calendar events with start and end times
        max_hours: Maximum number of free hours to return (default: 6)
        
    Returns:
        Number of free hours (integer) until next event, capped at max_hours
    """
    try:
        # Parse the current time string
        day_parts = current_time_str.split()
        
        # Handle different time formats: "Friday 2:30 PM" or "Friday 2 PM"
        time_str = " ".join(day_parts[1:])  # Extract the time portion
        
        # Parse hour and minute
        current_hour = 0
        current_minute = 0
        
        if ":" in time_str:
            # Format like "2:30 PM"
            time_parts = time_str.split(":")
            current_hour = int(time_parts[0])
            
            # Extract minutes from the second part which may contain AM/PM
            minute_part = time_parts[1].split()[0]
            current_minute = int(minute_part)
            
            # Check for AM/PM
            if "PM" in time_str.upper() and current_hour < 12:
                current_hour += 12
            elif "AM" in time_str.upper() and current_hour == 12:
                current_hour = 0
        else:
            # Format like "2 PM"
            hour_part = time_str.split()[0]
            current_hour = int(hour_part)
            
            # Check for AM/PM
            if "PM" in time_str.upper() and current_hour < 12:
                current_hour += 12
            elif "AM" in time_str.upper() and current_hour == 12:
                current_hour = 0
        
        # Current time in minutes since midnight
        current_time_minutes = current_hour * 60 + current_minute
        
        # Track ongoing events and find next event
        is_in_event = False
        next_event_minutes = None
        
        for event in calendar_events:
            event_start = event.get("start", "")
            event_end = event.get("end", "")
            
            if not event_start:
                continue
                
            # Parse event start time
            start_minutes = parse_time_to_minutes(event_start)
            
            # Parse event end time if available
            end_minutes = 23 * 60 + 59  # Default to end of day
            if event_end:
                end_minutes = parse_time_to_minutes(event_end)
            
            # Check if user is currently in an event
            # Key fix here: Use < end_minutes instead of <= end_minutes
            # This means the user is considered free exactly at the end time of an event
            if start_minutes <= current_time_minutes < end_minutes:
                is_in_event = True
                break
            
            # If not in an event, check if this is the next upcoming event
            elif start_minutes > current_time_minutes:
                if next_event_minutes is None or start_minutes < next_event_minutes:
                    next_event_minutes = start_minutes
        
        # Calculate free hours
        if is_in_event:
            # User is currently in an event - no free time now
            return 0
        elif next_event_minutes is None:
            # No future events today
            return max_hours
        else:
            free_minutes = next_event_minutes - current_time_minutes
            free_hours = free_minutes / 60.0
            return min(max(round(free_hours), 0), max_hours)
            
    except Exception as e:
        # If anything goes wrong, log and return 0 (safer assumption than max)
        logging.error(f"Error calculating free time: {str(e)}")
        logging.error(traceback.format_exc())
        return 0

def parse_time_to_minutes(time_str):
    """Helper function to parse time strings to minutes since midnight"""
    try:
        hour, minute = 0, 0
        
        if ":" in time_str:
            # Format like "1:30 PM"
            time_parts = time_str.split(":")
            hour = int(time_parts[0])
            
            # Extract minutes from the second part which may contain AM/PM
            minute_part = time_parts[1].split()[0]
            minute = int(minute_part)
            
            # Check for AM/PM
            if "PM" in time_str.upper() and hour < 12:
                hour += 12
            elif "AM" in time_str.upper() and hour == 12:
                hour = 0
        else:
            # Format like "1 PM"
            hour_part = time_str.split()[0]
            hour = int(hour_part)
            
            # Check for AM/PM
            if "PM" in time_str.upper() and hour < 12:
                hour += 12
            elif "AM" in time_str.upper() and hour == 12:
                hour = 0
                
        return hour * 60 + minute
    except Exception:
        return 0  # Default in case of error

# Enhanced version of choose_place with better error handling
@safe_api_call
def choose_place(user, places, model, user_feedback=None):
    """Choose the best place from options and return selected_place and LLM description."""
    if not places:
        logger.warning("No places found to choose from")
        return None, "We couldn't find any interesting places nearby. Let's suggest an indoor activity instead."

    try:
        if "disliked_places_ids" not in st.session_state:
            st.session_state.disliked_places_ids = []
            
        # Get suggestion history
        history = get_suggestion_history()
        previous_places = history.get("outdoor", [])

        # Filter out previously disliked and suggested places
        filtered_places = [
            place for place in places 
            if place.get("place_id") not in st.session_state.disliked_places_ids 
            and place.get("place_id") not in previous_places
        ]
        
        if not filtered_places:
            # If we've exhausted all options, allow reusing places but mention it
            logger.warning("All nearby places have been seen before, allowing repeats")
            filtered_places = [
                place for place in places 
                if place.get("place_id") not in st.session_state.disliked_places_ids
            ]
            
            if not filtered_places:
                return None, "You've seen all nearby places. Let's suggest an indoor activity instead."

        places = filtered_places
        enriched_places = []

        lat = user.get("location", {}).get("lat")
        lon = user.get("location", {}).get("lon")
        ors_client = st.session_state.get("ors_client")
        top_interest = st.session_state.get("top_interest", "activity")

        if not lat or not lon or not ors_client:
            return None, "We couldn't get enough data to find outdoor suggestions."

        personalized_context = build_personalized_context(user, top_interest)

        for idx, place in enumerate(places[:5]):
            try:
                place_lat = place["geometry"]["location"]["lat"]
                place_lon = place["geometry"]["location"]["lng"]
                travel_time_mins = get_route_duration((lon, lat), (place_lon, place_lat), ors_client)
                travel_time_mins = travel_time_mins * 2 if travel_time_mins else "unknown"

                enriched_places.append({
                    "prominence_rank": idx + 1,
                    "place": place,
                    "name": place.get("name", "Unknown"),
                    "rating": place.get("rating", "N/A"),
                    "total_ratings": place.get("user_ratings_total", 0),
                    "address": place.get("vicinity", "Unknown location"),
                    "travel_time_mins": travel_time_mins,
                    "type": place.get("type", top_interest)
                })
            except Exception as e:
                logger.error(f"Error enriching place: {str(e)}")
                continue

        if not enriched_places:
            return None, "We couldn't enrich any places to recommend."

        # Construct LLM prompt
        feedback_note = user_feedback + " " if user_feedback else ""
        prompt = f"""
{feedback_note}You're a smart assistant helping a user decide which is the best place to visit.

User preferences:
- Weather: {user.get("weather", "Unknown")}
- Time: {user.get("current_time", "Unknown")}
- Top interest: {top_interest}
- Free hours: {user.get("free_hours", "Unknown")}

User History and Preferences:
{personalized_context}

"""
        # Add information about previous suggestions if any
        if previous_places:
            prompt += "\nI want to suggest a NEW place the user hasn't seen before.\n"
        
        prompt += "Here are some options nearby:\n"

        for place in enriched_places:
            prompt += f"\n{place['prominence_rank']}. {place['name']} - Located at {place['address']}. "
            prompt += f"Rating: {place['rating']} ({place['total_ratings']} reviews). "
            prompt += f"Round trip travel time: {place['travel_time_mins']} minutes."

        prompt += """
❗ Choose only one place. Do not list or compare options. 
Make your response in 1–2 short, fun, personal sentences that could show up on a phone lockscreen.
Mention only one place by name.
"""

        response = model.generate_content(prompt)
        description = response.text.strip()

        # Extract the name of the place mentioned from the response
        selected_place = enriched_places[0]["place"]  # Default fallback

        for place in enriched_places:
            if place["name"].lower() in description.lower():
                selected_place = place["place"]
                break

        return selected_place, description

    except LLMError as e:
        logger.error(f"LLM Error in choose_place: {str(e)}")
        return None, "Sorry, we had an issue generating personalized recommendations. Let's try an indoor activity instead."
    except Exception as e:
        logger.error(f"Error in choose_place: {str(e)}")
        logger.error(traceback.format_exc())
        return None, "We encountered an unexpected error. Let's suggest an indoor activity instead."
