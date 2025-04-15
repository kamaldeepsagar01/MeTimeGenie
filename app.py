import streamlit as st
import os
from datetime import datetime
import google.generativeai as genai
import logging
import traceback

# Import from utils.py
from utils import (
    get_synthetic_user,
    top_activity_interest_llm,
    build_llm_decision_prompt,
    build_llm_prompt_indoor,
    fetch_places,
    fetch_place_image,
    choose_place,
    get_detailed_suggestion,
    init_clients,
    update_preferences_from_feedback,
    get_user_preferences_db,
    extract_main_keywords,
    fetch_image_for_keyword,
    extract_keywords_from_prompt,  # New import
    extract_food_keywords,  # New import
    extract_nouns,  # New import
    fetch_image_for_keyword,
    fetch_unsplash_image,  # New import
    fetch_google_images, extract_core_keyword, simplify_keyword,
    # New imports for suggestion history
    get_suggestion_history,
    is_duplicate_suggestion,
    add_to_suggestion_history,
    get_llm_prompt_with_history,
    calculate_free_time,
    parse_time_to_minutes,
    AppError, APIError, LLMError, ImageError
)

st.set_page_config(page_title="Activity Suggester", layout="centered")

# Inject custom CSS to position the title and improve UI
st.markdown("""
    <style>
    .custom-title {
        position: absolute;
        top: 10px;
        right: 20px;
        font-size: 14px;
        color: gray;
    }
    .stButton button {
        width: 100%;
        border-radius: 20px;
    }
    .feedback-history {
        margin-top: 30px;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 5px;
    }
    </style>
    <div class="custom-title">My Daily Activity Planner</div>
""", unsafe_allow_html=True)

# Initialize session state variables
if "initialized" not in st.session_state:
    # Load secrets
    try:
        GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        ORS_API_KEY = st.secrets["ORS_API_KEY"]

        # Configure Gemini model
        os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Initialize clients
        ors_client, gmaps_client = init_clients(ORS_API_KEY, GOOGLE_MAPS_API_KEY)

        # Store in session state
        st.session_state.GOOGLE_MAPS_API_KEY = GOOGLE_MAPS_API_KEY
        st.session_state.model = model
        st.session_state.ors_client = ors_client
        st.session_state.gmaps_client = gmaps_client
        st.session_state.user_feedback = None
        st.session_state.initialized = True

        # Set up error tracking
        st.session_state.errors = []
    except Exception as e:
        st.error(f"Error initializing app: {e}")
        st.stop()

# Get model from session state
model = st.session_state.model

# App Header
st.title("What should I do now?")

# Generate user context if not already in session
if "user" not in st.session_state:
    user = get_synthetic_user()
    st.session_state.user = user
else:
    user = st.session_state.user

# Process recommendation when needed
if "recommendation_shown" not in st.session_state or not st.session_state.recommendation_shown:
    with st.spinner("Finding the perfect activity for you..."):
        try:
            # Call LLM to suggest top interest
            if "top_interest" not in st.session_state:
                try:
                    top_interest = top_activity_interest_llm(user)
                    st.session_state.top_interest = top_interest
                except Exception as e:
                    logging.error(f"Error getting top interest: {str(e)}")
                    st.session_state.top_interest = "food"  # Default fallback
                    st.session_state.errors.append(f"Error determining interest: {str(e)}")
            else:
                top_interest = st.session_state.top_interest

            # Call second LLM to decide indoor or outdoor
            try:
                decision_prompt = build_llm_decision_prompt(user, top_interest)
                decision_response = model.generate_content(decision_prompt)
                decision = decision_response.text.strip().lower()
                st.session_state.activity_type = decision
            except Exception as e:
                logging.error(f"Error determining indoor/outdoor: {str(e)}")
                decision = "indoor"  # Default to indoor on error
                st.session_state.activity_type = decision
                st.session_state.errors.append(f"Error choosing activity type: {str(e)}")

            # Indoor flow
            if decision == "indoor":
                try:
                          # Enhance the prompt with history to avoid repetition
                    base_prompt = build_llm_prompt_indoor(user, top_interest, st.session_state.user_feedback)
                    enhanced_prompt = get_llm_prompt_with_history(base_prompt, "indoor")
                    
                    # Try up to 3 times to get a non-duplicate suggestion
                    max_attempts = 3
                    activity_description = None
                    for attempt in range(max_attempts):
                        response = model.generate_content(enhanced_prompt)
                        activity_description = response.text.strip()
                        
                        # Check if it's a duplicate
                        if not is_duplicate_suggestion(activity_description, "indoor"):
                            # Not a duplicate, we can use this
                            break
                        
                        # If it's a duplicate and not the last attempt, try again with stronger instruction
                        if attempt < max_attempts - 1:
                            logger.warning(f"Duplicate indoor suggestion detected, trying again (attempt {attempt+1})")
                            enhanced_prompt += "\n\n❗ IMPORTANT: Your previous suggestion was too similar to one you've made before. Please suggest something COMPLETELY DIFFERENT."
                    
                    # Record the suggestion in history
                    if activity_description:
                        add_to_suggestion_history(activity_description, "indoor")
                        st.session_state.last_short_response = activity_description
                    
                    response = model.generate_content(build_llm_prompt_indoor(user, top_interest, st.session_state.user_feedback))
                    activity_description = response.text.strip()
                    st.session_state.last_short_response = activity_description
            
                    # Extract keywords and fetch related image
                    image_url = None
                    main_keyword = None
                    
                    try:
                        # First, try to extract keywords using the advanced method
                        try:
                            keywords = extract_keywords_from_prompt(activity_description)
                            
                            # Try each keyword until we find an image
                            for keyword in keywords:
                                if not keyword or len(keyword.strip()) < 3:
                                    continue
                                    
                                logging.info(f"Trying to fetch image for keyword: {keyword}")
                                try:
                                    # Try with all available APIs
                                    img_url = fetch_image_for_keyword(
                                        keyword, 
                                        st.session_state.GOOGLE_MAPS_API_KEY,
                                        st.session_state.get('GOOGLE_CSE_ID'),
                                        st.session_state.get('GOOGLE_CSE_API_KEY')
                                    )
                                    if img_url:
                                        image_url = img_url
                                        main_keyword = keyword
                                        logging.info(f"Found image for keyword: {keyword}")
                                        break
                                except Exception as img_err:
                                    logging.error(f"Error fetching image for '{keyword}': {str(img_err)}")
                                    continue
                        except Exception as kw_err:
                            logging.error(f"Error extracting keywords: {str(kw_err)}")
                            
                        # If still no image, try with the backup method
                        if not image_url:
                            # Fall back to simple extraction
                            main_keyword = extract_main_keywords(activity_description)
                            if main_keyword and len(main_keyword) >= 3:
                                logging.info(f"Trying fallback keyword: {main_keyword}")
                                image_url = fetch_image_for_keyword(
                                    main_keyword, 
                                    st.session_state.GOOGLE_MAPS_API_KEY,
                                    st.session_state.get('GOOGLE_CSE_ID'),
                                    st.session_state.get('GOOGLE_CSE_API_KEY')
                                )
                            
                        # Final direct fallback to Unsplash with core keyword extraction
                        if not image_url and main_keyword:
                            core_keyword = extract_core_keyword(main_keyword)
                            logging.info(f"Trying core keyword: {core_keyword}")
                            image_url = fetch_unsplash_image(core_keyword)
                            main_keyword = core_keyword
                            
                        # Last resort - try with the interest type
                        if not image_url:
                            logging.info(f"Using interest type as keyword: {top_interest}")
                            image_url = fetch_unsplash_image(top_interest)
                            main_keyword = top_interest
                            
                    except Exception as e:
                        logging.error(f"All image fetching methods failed: {str(e)}")
                        image_url = None
                        main_keyword = top_interest
            
                    # Debug logging
                    if image_url:
                        logging.info(f"Successfully found image URL: {image_url} for keyword: {main_keyword}")
                    else:
                        logging.error("Failed to find any image URL")
            
                    st.session_state.recommendation_data = {
                        "type": "indoor",
                        "name": f"Indoor {top_interest} Activity",
                        "description": activity_description,
                        "image_url": image_url,
                        "activity_type": top_interest,
                        "keyword": main_keyword if main_keyword else top_interest
                    }
                except Exception as e:
                    logging.error(f"Error in indoor flow: {str(e)}")
                    traceback.print_exc()
                    st.session_state.recommendation_data = {
                        "type": "indoor",
                        "name": "Indoor Activity Suggestion",
                        "description": "Try a fun indoor activity related to your interests!",
                        "image_url": None,
                        "activity_type": top_interest
                    }
                    st.session_state.errors.append(f"Error creating indoor suggestion: {str(e)}")
                    
           # Outdoor flow
            else:        
                try:
                    # Fetch places from Google Maps
                    places = fetch_places(user, top_interest, st.session_state.GOOGLE_MAPS_API_KEY)
            
                    # Choose one - pass user feedback to the LLM
                    selected_place, description = choose_place(user, places, model, st.session_state.user_feedback)
                    if selected_place:
                        # Check if this place has been suggested before
                        place_id = selected_place.get("place_id")
                        if place_id and is_duplicate_suggestion(place_id, "outdoor"):
                            # It's a duplicate, try again with a different place
                            logger.warning("Duplicate outdoor place detected, trying to choose a different one")
                            # Filter out this place_id
                            filtered_places = [p for p in places if p.get("place_id") != place_id]
                            if filtered_places:
                                selected_place, description = choose_place(user, filtered_places, model, st.session_state.user_feedback)
                        
                        # Proceed with the selected place
                        if selected_place:
                            place_id = selected_place.get("place_id")
                            # Add to history
                            if place_id:
                                add_to_suggestion_history(description, "outdoor", place_id)
                                
                            try:
                                image_url = fetch_place_image(selected_place, st.session_state.GOOGLE_MAPS_API_KEY)
                                
                                # Make sure place name is mentioned in the description for clarity
                                place_name = selected_place.get("name", "Unknown place")
                                if place_name not in description:
                                    description = f"Check out {place_name}! {description}"
                                    
                            except Exception as e:
                                logging.error(f"Error fetching place image: {str(e)}")
                                image_url = None
                    
                            st.session_state.recommendation_data = {
                                "type": "outdoor",
                                "place": selected_place,
                                "name": selected_place.get("name", "Unknown place"),
                                "description": description,
                                "image_url": image_url,
                                "activity_type": top_interest
                            }
                            st.session_state.last_short_response = description
                    else:
                        # Fallback to indoor if no outdoor places found
                        logging.warning("No outdoor places found, falling back to indoor")
                        response = model.generate_content(build_llm_prompt_indoor(user, top_interest, st.session_state.user_feedback))
                        activity_description = response.text.strip()
            
                        # Extract keywords and fetch related image
                        main_keyword = extract_main_keywords(activity_description)
                        image_url = None
            
                        try:
                            if main_keyword:
                                image_url = fetch_image_for_keyword(main_keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                        except Exception as e:
                            logging.error(f"Error fetching indoor fallback image: {str(e)}")
            
                        st.session_state.last_short_response = activity_description
                        st.session_state.recommendation_data = {
                            "type": "indoor",
                            "name": f"Indoor {top_interest} Activity",
                            "description": activity_description,
                            "image_url": image_url,
                            "activity_type": top_interest,
                            "keyword": main_keyword
                        }
                except Exception as e:
                    logging.error(f"Error in outdoor flow: {str(e)}")
                    traceback.print_exc()
                    # Emergency fallback
                    st.session_state.recommendation_data = {
                        "type": "indoor",
                        "name": "Activity Suggestion",
                        "description": "We recommend trying something fun related to your interests!",
                        "image_url": None,
                        "activity_type": "activity"
                    }
                    st.session_state.errors.append(f"Error creating outdoor suggestion: {str(e)}")

            # Reset user feedback after using it
            if st.session_state.user_feedback:
                st.session_state.previous_feedback = st.session_state.user_feedback
                st.session_state.user_feedback = None

            st.session_state.recommendation_shown = True

        except Exception as e:
            logging.error(f"Unexpected error in recommendation process: {str(e)}")
            traceback.print_exc()
            st.session_state.errors.append(f"Unexpected error: {str(e)}")
            # Set up a basic fallback recommendation
            st.session_state.recommendation_data = {
                "type": "indoor",
                "name": "Activity Suggestion",
                "description": "Try something relaxing or fun based on your interests!",
                "image_url": None,
                "activity_type": "activity"
            }
            st.session_state.recommendation_shown = True

# Display the recommendation
if "recommendation_data" in st.session_state:
    data = st.session_state.recommendation_data

    # Display image if available (for both indoor and outdoor activities)
    if data.get("image_url"):
        st.image(data["image_url"], use_container_width=True)

    st.subheader("🔍 Suggested Activity")
    st.write(data["description"])

    # Show if this was based on previous feedback
    if "previous_feedback" in st.session_state and st.session_state.previous_feedback:
        st.info("This is a new suggestion based on your feedback.")
        st.session_state.previous_feedback = None

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 I like it!"):
            # Update user preferences with like
            item_data = {
                "name": data.get("name", "Unknown"),
                "type": data.get("activity_type", "Unknown")
            }
            update_preferences_from_feedback("like", item_data)
            st.balloons()
            st.success("Great! I'll remember you liked this for future recommendations!")

    with col2:
        if st.button("👎 Show me something else"):
        # Update user preferences with dislike
            item_data = {
                "name": data.get("name", "Unknown"),
                "type": data.get("activity_type", "Unknown")
            }
            update_preferences_from_feedback("dislike", item_data)
        
            # Add to disliked places list if it was an outdoor place
            if data.get("type") == "outdoor" and "place" in data and "place_id" in data["place"]:
                if "disliked_places_ids" not in st.session_state:
                    st.session_state.disliked_places_ids = []
                st.session_state.disliked_places_ids.append(data["place"]["place_id"])
                
            # Store feedback to use in next recommendation
            st.session_state.user_feedback = "The user did not like the previous suggestion. Please provide a completely different recommendation."
            
            # Update the top interest after preference change
            st.session_state.top_interest = top_activity_interest_llm(user)
            
            # Reset recommendation to get new one
            st.session_state.recommendation_shown = False
            
            # Clear any activity type decision to allow reconsideration
            if "activity_type" in st.session_state:
                del st.session_state.activity_type
                
            st.rerun()

    # Know More button
    if st.button("🔎 Tell me more"):
        # Update preferences when user views details
        item_data = {
            "name": data.get("name", "Unknown"),
            "type": data.get("activity_type", "Unknown")
        }
        update_preferences_from_feedback("view_details", item_data)
    
        # Get detailed suggestion
        detailed, maps_html = get_detailed_suggestion(
            user,
            model,
            st.session_state.last_short_response,
            st.session_state.top_interest,
            st.session_state.recommendation_data  # Pass the recommendation data
        )
        st.markdown(f"### 📖 More details:\n\n{detailed}")
        
        # Display maps link if available
        if maps_html:
            st.markdown(maps_html, unsafe_allow_html=True)

# Display errors if any occurred
if "errors" in st.session_state and st.session_state.errors:
    with st.expander("Troubleshooting Information", expanded=False):
        st.warning("Some issues occurred while generating your recommendations. We've provided alternatives instead.")
        for error in st.session_state.errors[-3:]:  # Show only the most recent errors
            st.error(error)
        if st.button("Clear Errors"):
            st.session_state.errors = []
            st.rerun()

# Display personalization summary in sidebar
with st.sidebar.expander("📊 Your Preference Profile"):
    prefs = get_user_preferences_db()

    # Show category preferences
    st.sidebar.subheader("Category Preferences")
    if prefs["category_preferences"]:
        for category, score in sorted(prefs["category_preferences"].items(), key=lambda x: x[1], reverse=True):
            st.sidebar.write(f"- {category}: {score:.1f}")
    else:
        st.sidebar.write("No preferences recorded yet.")

    # Show recent likes
    st.sidebar.subheader("Recent Likes")
    if prefs["liked_places"]:
        for item in prefs["liked_places"][-3:]:
            st.sidebar.write(f"- {item['name']} ({item['type']})")
    else:
        st.sidebar.write("No likes recorded yet .")

    # Show recent dislikes
    st.sidebar.subheader("Recent Dislikes")
    if prefs["disliked_places"]:
        for item in prefs["disliked_places"][-3:]:
            st.sidebar.write(f"- {item['name']} ({item['type']})")
    else:
        st.sidebar.write("No dislikes recorded yet.")

# Reset buttons
with st.sidebar.expander("🔄 Reset Options"):
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Reset Suggestion"):
            # Reset only current suggestion
            if "recommendation_shown" in st.session_state:
                del st.session_state.recommendation_shown
            if "recommendation_data" in st.session_state:
                del st.session_state.recommendation_data
            st.rerun()

    with col2:
        if st.button("Reset All"):
            # Reset everything including preferences
            for key in list(st.session_state.keys()):
                if key != "initialized" and key not in ["GOOGLE_MAPS_API_KEY", "model", "ors_client", "gmaps_client"]:
                    del st.session_state[key]
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Activity Planner App • v1.0")
