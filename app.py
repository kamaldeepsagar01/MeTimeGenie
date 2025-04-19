import streamlit as st
import os
from datetime import datetime
import google.generativeai as genai
import logging
import traceback

# Import from utils.py
from utils import (
    get_synthetic_user,
    get_synthetic_weekend_slots,
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
    fetch_and_store_events,
    has_more_events,
    get_next_event_for_display,
    format_event,
    get_upcoming_weekend,
    fetch_ticketmaster_events,
    fetch_eventbrite_events,
    fetch_predicthq_events,
    scrape_google_events,
    #Booking functions
    show_booking_options,
    generate_booking_urls,
    generate_airbnb_url,
    generate_booking_com_url,
    generate_agoda_url,
    generate_expedia_url,
    generate_hotels_com_url,
    open_booking_platform,
    AppError, APIError, LLMError, ImageError
)
from utils import astra_manager


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
        TICKETMASTER_API_KEY = st.secrets["TICKETMASTER_API_KEY"]
        EVENTBRITE_API_KEY = st.secrets["EVENTBRITE_API_KEY"]
        PREDICTHQ_API_KEY = st.secrets["PREDICTHQ_API_KEY"]

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

# Session State Initialization for Weekend Planner


# App Header
st.title("What should I do now?")

# Session State Initialization for Weekend Planner
if "weekend_initialized" not in st.session_state:
   # Get basic user data
   if "user" not in st.session_state:
       user = get_synthetic_user()
       st.session_state.user = user
   else:
       user = st.session_state.user
   
   # Get weekend slots
   weekend_slots = get_synthetic_weekend_slots()
    
   # Initialize weekend planning session state variables
   st.session_state.weekend_slots = weekend_slots
   st.session_state.booked_slots = {}
   st.session_state.current_view = "main"
   st.session_state.slot_recommendations = {}
   st.session_state.weekend_initialized = True





def render_main_view():
   """
   Renders the main view of the weekend planner with the primary recommendation
   and buttons for time slots and quick glance.
   """
   # Get user data
   user = st.session_state.user
   
   # Check if primary recommendation exists, if not generate one
   if "primary_recommendation" not in st.session_state:
       with st.spinner("Finding the perfect activity for you..."):
           # Get top interest
           top_interest = top_activity_interest_llm(user)
           st.session_state.top_interest = top_interest
           
           # Decide indoor or outdoor
           decision_prompt = build_llm_decision_prompt(user, top_interest)
           decision_response = st.session_state.model.generate_content(decision_prompt)
           decision = decision_response.text.strip().lower()
           
           # Generate recommendation based on decision
           if decision == "indoor":
               # Generate indoor activity
               prompt = build_llm_prompt_indoor(user, top_interest)
               response = st.session_state.model.generate_content(prompt)
               activity_description = response.text.strip()
               
               # Get image for activity
               main_keyword = extract_main_keywords(activity_description)
               image_url = fetch_image_for_keyword(main_keyword, st.session_state.GOOGLE_MAPS_API_KEY)
               
               recommendation = {
                   "type": "indoor",
                   "name": f"Indoor {top_interest} Activity",
                   "description": activity_description,
                   "image_url": image_url,
                   "activity_type": top_interest
               }
           elif decision == "outdoor":
               # Check if the user's top interest aligns with event categories
               event_related_interests = ["music", "sports", "entertainment", "theatre", "concerts", "festivals", "events", "arts"]
               
               # See if the top interest is event-related
               is_event_related = top_interest.lower() in [interest.lower() for interest in event_related_interests]
               
               # If interest is event-related, attempt to fetch events first
               if is_event_related:
                   try:
                       # Get user location info
                       city = user.get("location", {}).get("city", "")
                       country_code = user.get("location", {}).get("country_code", "US")  # Default to US
                       
                       # Get upcoming weekend dates
                       today = datetime.now()
                       saturday, sunday = get_upcoming_weekend(today)
                       
                       # Format dates for API
                       start_date = saturday.strftime("%Y-%m-%d")
                       end_date = sunday.strftime("%Y-%m-%d")
                       
                       # Try to fetch events
                       events_found = fetch_and_store_events(
                           interest=top_interest,
                           city=city,
                           country_code=country_code,
                           start_date=start_date,
                           end_date=end_date
                       )
                       
                       # If events were found, process and display them
                       if events_found and has_more_events():
                           # Get first event
                           event = get_next_event_for_display()
                           
                           if event:
                               # Format event description
                               event_description = f"Check out this event: **{event['title']}**\n\n"
                               event_description += f"📅 **Date:** {event['date']}\n"
                               event_description += f"📍 **Location:** {event['location']}\n"
                               if event.get('venue'):
                                   event_description += f"🏢 **Venue:** {event['venue']}\n"
                               
                               # Get image for event
                               image_url = None
                               try:
                                   keywords = extract_keywords_from_prompt(event['title'])
                                   for keyword in keywords:
                                       if keyword and len(keyword.strip()) >= 3:
                                           img_url = fetch_image_for_keyword(keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                                           if img_url:
                                               image_url = img_url
                                               break
                               except Exception as e:
                                   logging.error(f"Error getting event image: {str(e)}")
                               
                               recommendation = {
                                   "type": "event",
                                   "name": event['title'],
                                   "description": event_description,
                                   "image_url": image_url,
                                   "activity_type": top_interest,
                                   "event_data": event
                               }
                           else:
                               # Fall back to regular outdoor options if no events available
                               places = fetch_places(user, top_interest, st.session_state.GOOGLE_MAPS_API_KEY)
                               selected_place, description = choose_place(user, places, st.session_state.model)
                               
                               if selected_place:
                                   image_url = fetch_place_image(selected_place, st.session_state.GOOGLE_MAPS_API_KEY)
                                   recommendation = {
                                       "type": "outdoor",
                                       "place": selected_place,
                                       "name": selected_place.get("name", "Unknown place"),
                                       "description": description,
                                       "image_url": image_url,
                                       "activity_type": top_interest
                                   }
                               else:
                                   # Fallback to indoor if no places found
                                   prompt = build_llm_prompt_indoor(user, top_interest)
                                   response = st.session_state.model.generate_content(prompt)
                                   activity_description = response.text.strip()
                                   main_keyword = extract_main_keywords(activity_description)
                                   image_url = fetch_image_for_keyword(main_keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                                   
                                   recommendation = {
                                       "type": "indoor",
                                       "name": f"Indoor {top_interest} Activity",
                                       "description": activity_description,
                                       "image_url": image_url,
                                       "activity_type": top_interest
                                   }
                       else:
                           # No events found, use regular outdoor recommendations
                           places = fetch_places(user, top_interest, st.session_state.GOOGLE_MAPS_API_KEY)
                           selected_place, description = choose_place(user, places, st.session_state.model)
                           
                           if selected_place:
                               image_url = fetch_place_image(selected_place, st.session_state.GOOGLE_MAPS_API_KEY)
                               recommendation = {
                                   "type": "outdoor",
                                   "place": selected_place,
                                   "name": selected_place.get("name", "Unknown place"),
                                   "description": description,
                                   "image_url": image_url,
                                   "activity_type": top_interest
                               }
                           else:
                               # Fallback to indoor if no places found
                               prompt = build_llm_prompt_indoor(user, top_interest)
                               response = st.session_state.model.generate_content(prompt)
                               activity_description = response.text.strip()
                               main_keyword = extract_main_keywords(activity_description)
                               image_url = fetch_image_for_keyword(main_keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                               
                               recommendation = {
                                   "type": "indoor",
                                   "name": f"Indoor {top_interest} Activity",
                                   "description": activity_description,
                                   "image_url": image_url,
                                   "activity_type": top_interest
                               }
                   except Exception as e:
                       logging.error(f"Error in event flow: {str(e)}")
                       # Fall back to regular outdoor recommendations
                       places = fetch_places(user, top_interest, st.session_state.GOOGLE_MAPS_API_KEY)
                       selected_place, description = choose_place(user, places, st.session_state.model)
                       
                       if selected_place:
                           image_url = fetch_place_image(selected_place, st.session_state.GOOGLE_MAPS_API_KEY)
                           recommendation = {
                               "type": "outdoor",
                               "place": selected_place,
                               "name": selected_place.get("name", "Unknown place"),
                               "description": description,
                               "image_url": image_url,
                               "activity_type": top_interest
                           }
                       else:
                           # Fallback to indoor if no places found
                           prompt = build_llm_prompt_indoor(user, top_interest)
                           response = st.session_state.model.generate_content(prompt)
                           activity_description = response.text.strip()
                           main_keyword = extract_main_keywords(activity_description)
                           image_url = fetch_image_for_keyword(main_keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                           
                           recommendation = {
                               "type": "indoor",
                               "name": f"Indoor {top_interest} Activity",
                               "description": activity_description,
                               "image_url": image_url,
                               "activity_type": top_interest
                           }
               else:
                   # Not event-related, use regular outdoor recommendations
                   places = fetch_places(user, top_interest, st.session_state.GOOGLE_MAPS_API_KEY)
                   selected_place, description = choose_place(user, places, st.session_state.model)
                   
                   if selected_place:
                       image_url = fetch_place_image(selected_place, st.session_state.GOOGLE_MAPS_API_KEY)
                       recommendation = {
                           "type": "outdoor",
                           "place": selected_place,
                           "name": selected_place.get("name", "Unknown place"),
                           "description": description,
                           "image_url": image_url,
                           "activity_type": top_interest
                       }
                   else:
                       # Fallback to indoor if no places found
                       prompt = build_llm_prompt_indoor(user, top_interest)
                       response = st.session_state.model.generate_content(prompt)
                       activity_description = response.text.strip()
                       main_keyword = extract_main_keywords(activity_description)
                       image_url = fetch_image_for_keyword(main_keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                       
                       recommendation = {
                           "type": "indoor",
                           "name": f"Indoor {top_interest} Activity",
                           "description": activity_description,
                           "image_url": image_url,
                           "activity_type": top_interest
                       }
           
           st.session_state.primary_recommendation = recommendation
           st.session_state.last_short_response = recommendation["description"]
   
   # Display primary recommendation
   recommendation = st.session_state.primary_recommendation
   
   # Display image if available
   if recommendation.get("image_url"):
       st.image(recommendation["image_url"], use_container_width=True)
   
   # Display recommendation details
   st.subheader("🔍 Suggested Activity")
   st.write(recommendation["description"])
   
   # Add event-specific UI elements
   if recommendation.get("type") == "event" and recommendation.get("event_data"):
       event = recommendation["event_data"]
       if event.get("source"):
           st.markdown(f"**Source:** {event['source']}")
       
       # Check if there are more events
       if has_more_events():
           if st.button("Show Next Event"):
               next_event = get_next_event_for_display()
               if next_event:
                   # Format event description
                   event_description = f"Check out this event: **{next_event['title']}**\n\n"
                   event_description += f"📅 **Date:** {next_event['date']}\n"
                   event_description += f"📍 **Location:** {next_event['location']}\n"
                   if next_event.get('venue'):
                       event_description += f"🏢 **Venue:** {next_event['venue']}\n"
                   
                   # Try to get image for the event
                   image_url = None
                   try:
                       # Try with event title
                       keywords = extract_keywords_from_prompt(next_event['title'])
                       for keyword in keywords:
                           if keyword and len(keyword.strip()) >= 3:
                               img_url = fetch_image_for_keyword(keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                               if img_url:
                                   image_url = img_url
                                   break
                       
                       # If no image found, try with venue
                       if not image_url and next_event.get('venue'):
                           image_url = fetch_image_for_keyword(next_event['venue'], st.session_state.GOOGLE_MAPS_API_KEY)
                       
                       # Last resort - try with interest type
                       if not image_url:
                           image_url = fetch_unsplash_image(st.session_state.top_interest)
                   except Exception:
                       pass
                   
                   # Update recommendation data
                   st.session_state.primary_recommendation = {
                       "type": "event",
                       "name": next_event['title'],
                       "description": event_description,
                       "image_url": image_url,
                       "activity_type": st.session_state.top_interest,
                       "event_data": next_event
                   }
                   st.session_state.last_short_response = event_description
                   st.rerun()
   
   # Record interaction in database
   success = astra_manager.record_interaction({
       "user_id": user.get("user_id", "unknown"),
       "interaction_type": recommendation.get("type"),  # "indoor", "outdoor", or "event"
       "suggested_activity": recommendation.get("description"),
       "recommendation_data": recommendation,  # full recommendation payload
       "user_action": "",  # placeholder for Like / Dislike / etc.
       "session_id": st.session_state.get("session_id"),
       "user_interests": user.get("interests", {}),
       "location": user.get("location", {}),
       "weather": user.get("weather", ""),
       "time": user.get("current_time", ""),
       "calendar": user.get("calendar", [])
   })

   if not success:
       st.warning("⚠️ Failed to save interaction to database.")
   
   # Action buttons for primary recommendation
   col1, col2 = st.columns(2)
   
   with col1:
       if st.button("👍 I like it!"):
           # Update user preferences with like
           item_data = {
               "name": recommendation.get("name", "Unknown"),
               "type": recommendation.get("activity_type", "Unknown")
           }
           update_preferences_from_feedback("like", item_data)
           st.balloons()
           st.success("Great! I'll remember you liked this!")
   
   with col2:
       if st.button("👎 Show me something else"):
           # Update user preferences with dislike
           item_data = {
               "name": recommendation.get("name", "Unknown"),
               "type": recommendation.get("activity_type", "Unknown")
           }
           update_preferences_from_feedback("dislike", item_data)
           
           # Clear primary recommendation to generate a new one
           if "primary_recommendation" in st.session_state:
               del st.session_state.primary_recommendation
           
           st.rerun()
   
   # Know More button
   if st.button("🔎 Tell me more"):
       # Update preferences when user views details
       item_data = {
           "name": recommendation.get("name", "Unknown"),
           "type": recommendation.get("activity_type", "Unknown")
       }
       update_preferences_from_feedback("view_details", item_data)
       
       # Get detailed suggestion
       detailed, maps_html = get_detailed_suggestion(
           user,
           st.session_state.model,
           st.session_state.last_short_response,
           st.session_state.top_interest,
           st.session_state.primary_recommendation
       )
       st.markdown(f"### 📖 More details:\n\n{detailed}")
       
       # Display ticket link if it's an event and has a URL
       if recommendation.get("type") == "event" and recommendation.get("event_data", {}).get("event_url"):
           event_url = recommendation["event_data"]["event_url"]
           ticket_html = f"""
           <div style="margin-top: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
               <strong>🎫 Get Tickets:</strong> <a href="{event_url}" target="_blank">Click here to view tickets/details</a>
           </div>
           """
           st.markdown(ticket_html, unsafe_allow_html=True)
       # Display maps link if available
       elif maps_html:
           st.markdown(maps_html, unsafe_allow_html=True)
           # Show booking options for relevant recommendation types
           show_booking_options(recommendation)
   
   # Book this slot button
   if st.button("📌 Book this slot"):
       # Find nearest slot in weekend slots (for demo, just use first slot)
       if st.session_state.weekend_slots and len(st.session_state.weekend_slots) > 0:
           nearest_slot = st.session_state.weekend_slots[0]
           slot_id = nearest_slot["id"]
           
           # Book the activity for this slot
           st.session_state.booked_slots[slot_id] = recommendation
           
           # Record interaction in database
           success = astra_manager.record_interaction({
               "user_id": user.get("user_id", "unknown"),
               "interaction_type": "booking",
               "slot_id": slot_id,
               "slot_day": nearest_slot["day"],
               "slot_time": f"{nearest_slot['start_time']} - {nearest_slot['end_time']}",
               "activity": recommendation,
               "timestamp": datetime.now().isoformat()
           })
           
           st.success(f"Activity booked for {nearest_slot['day']} {nearest_slot['start_time']} - {nearest_slot['end_time']}!")
   
   # Display weekend planning section
   st.markdown("---")
   st.subheader("Hey, I've planned exciting stuff for your weekend! Here's a quick glance!")
   
   # Display slot buttons
   st.markdown("### Choose a time slot:")
   
   # Create buttons for each time slot
   slot_cols = st.columns(min(len(st.session_state.weekend_slots), 4))
   
   for i, slot in enumerate(st.session_state.weekend_slots):
       slot_id = slot["id"]
       slot_text = f"{slot['day']} {slot['start_time']}-{slot['end_time']}"
       
       # Check if slot is booked
       is_booked = slot_id in st.session_state.booked_slots
       
       # Display slot button with appropriate styling
       with slot_cols[i % len(slot_cols)]:
           button_label = f"{slot_text}"
           if is_booked:
               button_label += " ✓"
           
           if st.button(button_label, key=f"slot_btn_{slot_id}"):
               st.session_state.selected_slot_id = slot_id
               st.session_state.current_view = "slot"
               st.rerun()
   
   # Quick Glance button
   if st.button("🔍 Quick Glance", key="quick_glance_btn"):
       st.session_state.current_view = "quick_glance"
       st.rerun()




def render_slot_recommendation(slot_id):
   """
   Renders the recommendation view for a specific time slot.
   
   Args:
       slot_id: ID of the selected slot (e.g., "S1", "S2")
   """
   # Get user data
   user = st.session_state.user
   
   # Get slot data
   slot = None
   for s in st.session_state.weekend_slots:
       if s["id"] == slot_id:
           slot = s
           break
   
   if not slot:
       st.error(f"Slot {slot_id} not found!")
       st.session_state.current_view = "main"
       st.rerun()
       return
   
   # Display slot information
   st.header(f"Activity for {slot['day']} {slot['start_time']}-{slot['end_time']}")
   
   # Check if this slot is already booked
   is_booked = slot_id in st.session_state.booked_slots
   
   if is_booked:
       # Display the booked activity
       recommendation = st.session_state.booked_slots[slot_id]
       st.success("✅ This slot is booked!")
   else:
       # Generate or retrieve recommendation for this slot
       if slot_id not in st.session_state.slot_recommendations:
           with st.spinner("Finding the perfect activity for this time slot..."):
               # Get top interest
               top_interest = top_activity_interest_llm(user)
               
               # Decide indoor or outdoor
               decision_prompt = build_llm_decision_prompt(user, top_interest)
               decision_response = st.session_state.model.generate_content(decision_prompt)
               decision = decision_response.text.strip().lower()
               
               # Generate recommendation based on decision
               if decision == "indoor":
                   # Build prompt with slot-specific context
                   slot_context = f"You have {slot['duration_hours']} hours available on {slot['day']} from {slot['start_time']} to {slot['end_time']}."
                   prompt = build_llm_prompt_indoor(user, top_interest)
                   # Insert slot context into prompt
                   prompt = prompt.replace("My context:", f"My context:\n- {slot_context}\n-")
                   
                   response = st.session_state.model.generate_content(prompt)
                   activity_description = response.text.strip()
                   
                   # Get image for activity
                   main_keyword = extract_main_keywords(activity_description)
                   image_url = fetch_image_for_keyword(main_keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                   
                   recommendation = {
                       "type": "indoor",
                       "name": f"Indoor {top_interest} Activity",
                       "description": activity_description,
                       "image_url": image_url,
                       "activity_type": top_interest
                   }
               elif decision == "outdoor":
                   # Check if the user's top interest aligns with event categories
                   event_related_interests = ["music", "sports", "entertainment", "theatre", "concerts", "festivals", "events", "arts"]
                   
                   # See if the top interest is event-related
                   is_event_related = top_interest.lower() in [interest.lower() for interest in event_related_interests]
                   
                   # If interest is event-related, attempt to fetch events first
                   if is_event_related:
                       try:
                           # Get user location info
                           city = user.get("location", {}).get("city", "")
                           country_code = user.get("location", {}).get("country_code", "US")  # Default to US
                           
                           # Use the slot's date to determine which weekend day to use
                           slot_date = None
                           if "saturday" in slot["day"].lower():
                               saturday, _ = get_upcoming_weekend(datetime.now())
                               slot_date = saturday
                           elif "sunday" in slot["day"].lower():
                               _, sunday = get_upcoming_weekend(datetime.now())
                               slot_date = sunday
                               
                           if slot_date:
                               date_str = slot_date.strftime("%Y-%m-%d")
                               
                               # Try to fetch events for this specific date
                               events_found = fetch_and_store_events(
                                   interest=top_interest,
                                   city=city,
                                   country_code=country_code,
                                   start_date=date_str,
                                   end_date=date_str
                               )
                               
                               if events_found and has_more_events():
                                   event = get_next_event_for_display()
                                   if event:
                                       # Format event description
                                       event_description = f"Check out this event: **{event['title']}**\n\n"
                                       event_description += f"📅 **Date:** {event['date']}\n"
                                       event_description += f"📍 **Location:** {event['location']}\n"
                                       if event.get('venue'):
                                           event_description += f"🏢 **Venue:** {event['venue']}\n"
                                       
                                       # Get image for event
                                       image_url = None
                                       try:
                                           keywords = extract_keywords_from_prompt(event['title'])
                                           for keyword in keywords:
                                               if keyword and len(keyword.strip()) >= 3:
                                                   img_url = fetch_image_for_keyword(keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                                                   if img_url:
                                                       image_url = img_url
                                                       break
                                           
                                           # If no image found, try with venue
                                           if not image_url and event.get('venue'):
                                               image_url = fetch_image_for_keyword(event['venue'], st.session_state.GOOGLE_MAPS_API_KEY)
                                               
                                           # Last resort - try with interest type
                                           if not image_url:
                                               image_url = fetch_unsplash_image(top_interest)
                                       except Exception as e:
                                           logging.error(f"Error getting event image: {str(e)}")
                                       
                                       recommendation = {
                                           "type": "event",
                                           "name": event['title'],
                                           "description": event_description,
                                           "image_url": image_url,
                                           "activity_type": top_interest,
                                           "event_data": event
                                       }
                                       
                                       # Store recommendation and continue
                                       st.session_state.slot_recommendations[slot_id] = recommendation
                                       st.session_state.last_short_response = event_description
                                       
                                       # Skip the rest of the function
                                       recommendation = st.session_state.slot_recommendations[slot_id]
                
                       except Exception as e:
                           logging.error(f"Error finding events for slot: {str(e)}")
                   
                   # If we're still here, either not event-related, no events found, or error occurred
                   # Proceed with regular outdoor recommendation
                   if slot_id not in st.session_state.slot_recommendations:
                       # Generate outdoor activity
                       places = fetch_places(user, top_interest, st.session_state.GOOGLE_MAPS_API_KEY)
                       
                       # Add slot context for outdoor selection
                       slot_context = f"The user has {slot['duration_hours']} hours available on {slot['day']} from {slot['start_time']} to {slot['end_time']}."
                       selected_place, description = choose_place(user, places, st.session_state.model, user_feedback=slot_context)
                       
                       if selected_place:
                           image_url = fetch_place_image(selected_place, st.session_state.GOOGLE_MAPS_API_KEY)
                           recommendation = {
                               "type": "outdoor",
                               "place": selected_place,
                               "name": selected_place.get("name", "Unknown place"),
                               "description": description,
                               "image_url": image_url,
                               "activity_type": top_interest
                           }
                       else:
                           # Fallback to indoor if no outdoor places found
                           prompt = build_llm_prompt_indoor(user, top_interest)
                           # Insert slot context into prompt
                           prompt = prompt.replace("My context:", f"My context:\n- {slot_context}\n-")
                           
                           response = st.session_state.model.generate_content(prompt)
                           activity_description = response.text.strip()
                           main_keyword = extract_main_keywords(activity_description)
                           image_url = fetch_image_for_keyword(main_keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                           
                           recommendation = {
                               "type": "indoor",
                               "name": f"Indoor {top_interest} Activity",
                               "description": activity_description,
                               "image_url": image_url,
                               "activity_type": top_interest
                           }
                       
                       # Store recommendation in session state
                       st.session_state.slot_recommendations[slot_id] = recommendation
                       st.session_state.last_short_response = recommendation.get("description", "")
               else:
                   # Shouldn't reach here, but just in case we get an unexpected decision value
                   logging.warning(f"Unexpected decision value: {decision}")
                   # Fallback to indoor
                   slot_context = f"You have {slot['duration_hours']} hours available on {slot['day']} from {slot['start_time']} to {slot['end_time']}."
                   prompt = build_llm_prompt_indoor(user, top_interest)
                   prompt = prompt.replace("My context:", f"My context:\n- {slot_context}\n-")
                   
                   response = st.session_state.model.generate_content(prompt)
                   activity_description = response.text.strip()
                   main_keyword = extract_main_keywords(activity_description)
                   image_url = fetch_image_for_keyword(main_keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                   
                   recommendation = {
                       "type": "indoor",
                       "name": f"Indoor {top_interest} Activity",
                       "description": activity_description,
                       "image_url": image_url,
                       "activity_type": top_interest
                   }
                   
                   st.session_state.slot_recommendations[slot_id] = recommendation
                   st.session_state.last_short_response = activity_description
       
       # Get the recommendation from session state
       recommendation = st.session_state.slot_recommendations[slot_id]
   
   # Display recommendation
   if recommendation.get("image_url"):
       st.image(recommendation["image_url"], use_container_width=True)
   
   st.subheader("🔍 Suggested Activity")
   st.write(recommendation["description"])
   
   # Add event-specific UI elements
   if recommendation.get("type") == "event" and "event_data" in recommendation:
       event = recommendation["event_data"]
       st.info("🎟️ Event")
       
       if event.get("source"):
           st.markdown(f"**Source:** {event['source']}")
       
       # Check if there are more events
       if has_more_events():
           if st.button("Show Next Event", key=f"next_event_{slot_id}"):
               next_event = get_next_event_for_display()
               if next_event:
                   # Format event description
                   event_description = f"Check out this event: **{next_event['title']}**\n\n"
                   event_description += f"📅 **Date:** {next_event['date']}\n"
                   event_description += f"📍 **Location:** {next_event['location']}\n"
                   if next_event.get('venue'):
                       event_description += f"🏢 **Venue:** {next_event['venue']}\n"
                   
                   # Try to get image for the event
                   image_url = None
                   try:
                       keywords = extract_keywords_from_prompt(next_event['title'])
                       for keyword in keywords:
                           if keyword and len(keyword.strip()) >= 3:
                               img_url = fetch_image_for_keyword(keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                               if img_url:
                                   image_url = img_url
                                   break
                       
                       # If no image found, try with venue
                       if not image_url and next_event.get('venue'):
                           image_url = fetch_image_for_keyword(next_event['venue'], st.session_state.GOOGLE_MAPS_API_KEY)
                       
                       # Last resort - try with interest type
                       if not image_url:
                           image_url = fetch_unsplash_image(recommendation["activity_type"])
                   except Exception as e:
                       logging.error(f"Error getting next event image: {str(e)}")
                   
                   # Update recommendation for this slot
                   updated_recommendation = {
                       "type": "event",
                       "name": next_event['title'],
                       "description": event_description,
                       "image_url": image_url,
                       "activity_type": recommendation["activity_type"],
                       "event_data": next_event
                   }
                   
                   st.session_state.slot_recommendations[slot_id] = updated_recommendation
                   st.rerun()
   
   # Record interaction in database
   success = astra_manager.record_interaction({
       "user_id": user.get("user_id", "unknown"),
       "interaction_type": recommendation.get("type"),  # "indoor", "outdoor" or "event"
       "suggested_activity": recommendation.get("description"),
       "recommendation_data": recommendation,  # full recommendation payload
       "user_action": "",  # placeholder for Like / Dislike / etc.
       "session_id": st.session_state.get("session_id"),
       "user_interests": user.get("interests", {}),
       "location": user.get("location", {}),
       "weather": user.get("weather", ""),
       "time": user.get("current_time", ""),
       "calendar": user.get("calendar", []),
       "slot_id": slot_id,
       "slot_info": slot
   })
   
   if not success:
       st.warning("⚠️ Failed to save interaction to database.")
   
   # Action buttons
   if not is_booked:
       # Show like/dislike buttons only if not booked
       col1, col2 = st.columns(2)
       
       with col1:
           if st.button("👍 I like it!", key=f"like_slot_{slot_id}"):
               # Update user preferences with like
               item_data = {
                   "name": recommendation.get("name", "Unknown"),
                   "type": recommendation.get("activity_type", "Unknown")
               }
               update_preferences_from_feedback("like", item_data)
               st.balloons()
               st.success("Great! I'll remember you liked this!")
       
       with col2:
           if st.button("👎 Show me something else", key=f"dislike_slot_{slot_id}"):
               # Update user preferences with dislike
               item_data = {
                   "name": recommendation.get("name", "Unknown"),
                   "type": recommendation.get("activity_type", "Unknown")
               }
               update_preferences_from_feedback("dislike", item_data)
               
               # Remove this recommendation to generate a new one
               if slot_id in st.session_state.slot_recommendations:
                   del st.session_state.slot_recommendations[slot_id]
               
               st.rerun()
       
       # Know More button
       if st.button("🔎 Tell me more", key=f"more_slot_{slot_id}"):
           # Update preferences when user views details
           item_data = {
               "name": recommendation.get("name", "Unknown"),
               "type": recommendation.get("activity_type", "Unknown")
           }
           update_preferences_from_feedback("view_details", item_data)
           
           # Get detailed suggestion
           detailed, maps_html = get_detailed_suggestion(
               user,
               st.session_state.model,
               recommendation["description"],
               recommendation.get("activity_type", ""),
               recommendation
           )
           st.markdown(f"### 📖 More details:\n\n{detailed}")
           
           # Display ticket link if it's an event and has a URL
           if recommendation.get("type") == "event" and recommendation.get("event_data", {}).get("event_url"):
               event_url = recommendation["event_data"]["event_url"]
               ticket_html = f"""
               <div style="margin-top: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
                   <strong>🎫 Get Tickets:</strong> <a href="{event_url}" target="_blank">Click here to view tickets/details</a>
               </div>
               """
               st.markdown(ticket_html, unsafe_allow_html=True)
           # Display maps link if available
           elif maps_html:
               st.markdown(maps_html, unsafe_allow_html=True)
               # Show booking options for relevant recommendation types
               show_booking_options(recommendation)
       
       # Book this slot button
       if st.button("📌 Book this slot", key=f"book_slot_{slot_id}"):
           # Book the activity for this slot
           st.session_state.booked_slots[slot_id] = recommendation
           
           # Record interaction in database
           success = astra_manager.record_interaction({
               "user_id": user.get("user_id", "unknown"),
               "interaction_type": "booking",
               "slot_id": slot_id,
               "slot_day": slot["day"],
               "slot_time": f"{slot['start_time']} - {slot['end_time']}",
               "activity": recommendation,
               "timestamp": datetime.now().isoformat()
           })
           
           st.success(f"Activity booked for {slot['day']} {slot['start_time']} - {slot['end_time']}!")
           st.rerun()
   
   # Back to main view button
   if st.button("← Back to main view", key=f"back_slot_{slot_id}"):
       st.session_state.current_view = "main"
       st.rerun()






def render_quick_glance_view():
   """
   Renders the quick glance view showing all weekend slots with their recommended activities.
   """
   # Get user data
   user = st.session_state.user
   
   st.header("Your Weekend Plan - Quick Glance")
   
   # Ensure all slots have recommendations
   for slot in st.session_state.weekend_slots:
       slot_id = slot["id"]
       if slot_id not in st.session_state.slot_recommendations and slot_id not in st.session_state.booked_slots:
           with st.spinner(f"Finding an activity for {slot['day']} {slot['start_time']}-{slot['end_time']}..."):
               # Get top interest
               top_interest = top_activity_interest_llm(user)
               
               # Decide indoor/outdoor
               decision_prompt = build_llm_decision_prompt(user, top_interest)
               decision_response = st.session_state.model.generate_content(decision_prompt)
               decision = decision_response.text.strip().lower()
               
               # Follow the full recommendation flow
               if decision == "indoor":
                   # Generate indoor activity
                   prompt = build_llm_prompt_indoor(user, top_interest)
                   slot_context = f"You have {slot['duration_hours']} hours available on {slot['day']} from {slot['start_time']} to {slot['end_time']}."
                   prompt = prompt.replace("My context:", f"My context:\n- {slot_context}\n-")
                   
                   response = st.session_state.model.generate_content(prompt)
                   activity_description = response.text.strip()
                   
                   # Get image
                   main_keyword = extract_main_keywords(activity_description)
                   image_url = fetch_image_for_keyword(main_keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                   
                   recommendation = {
                       "type": "indoor",
                       "name": f"Indoor {top_interest} Activity",
                       "description": activity_description,
                       "image_url": image_url,
                       "activity_type": top_interest
                   }
               elif decision == "outdoor":
                   # Check if event-related interest
                   event_related_interests = ["music", "sports", "entertainment", "theatre", "concerts", "festivals", "events", "arts"]
                   is_event_related = top_interest.lower() in [interest.lower() for interest in event_related_interests]
                   
                   if is_event_related:
                       try:
                           # Get location & date info
                           city = user.get("location", {}).get("city", "")
                           country_code = user.get("location", {}).get("country_code", "US")
                           
                           # Use the slot's date instead of calculating weekend
                           slot_date = None
                           if "saturday" in slot["day"].lower():
                               saturday, _ = get_upcoming_weekend(datetime.now())
                               slot_date = saturday
                           elif "sunday" in slot["day"].lower():
                               _, sunday = get_upcoming_weekend(datetime.now())
                               slot_date = sunday
                               
                           if slot_date:
                               date_str = slot_date.strftime("%Y-%m-%d")
                               
                               # Try to fetch events for this specific date
                               events_found = fetch_and_store_events(
                                   interest=top_interest,
                                   city=city,
                                   country_code=country_code,
                                   start_date=date_str,
                                   end_date=date_str
                               )
                               
                               if events_found and has_more_events():
                                   event = get_next_event_for_display()
                                   if event:
                                       # Format event for this slot
                                       event_description = f"Check out this event: **{event['title']}**\n\n"
                                       event_description += f"📅 **Date:** {event['date']}\n"
                                       event_description += f"📍 **Location:** {event['location']}\n"
                                       
                                       # Get image for event
                                       image_url = None
                                       try:
                                           keywords = extract_keywords_from_prompt(event['title'])
                                           for keyword in keywords:
                                               if keyword and len(keyword.strip()) >= 3:
                                                   img_url = fetch_image_for_keyword(keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                                                   if img_url:
                                                       image_url = img_url
                                                       break
                                       except Exception as e:
                                           logging.error(f"Error getting event image: {str(e)}")
                                       
                                       recommendation = {
                                           "type": "event",
                                           "name": event['title'],
                                           "description": event_description,
                                           "image_url": image_url,
                                           "activity_type": top_interest,
                                           "event_data": event
                                       }
                                       # Successfully created event recommendation
                                       st.session_state.slot_recommendations[slot_id] = recommendation
                                       
                       except Exception as e:
                           logging.error(f"Error fetching events for slot: {str(e)}")
                   
                   # If we reached here, either not event-related or no events found
                   # Fall back to places
                   try:
                       # Add slot context for outdoor selection
                       slot_context = f"The user has {slot['duration_hours']} hours available on {slot['day']} from {slot['start_time']} to {slot['end_time']}."
                       places = fetch_places(user, top_interest, st.session_state.GOOGLE_MAPS_API_KEY)
                       selected_place, description = choose_place(user, places, st.session_state.model, user_feedback=slot_context)
                       
                       if selected_place:
                           image_url = fetch_place_image(selected_place, st.session_state.GOOGLE_MAPS_API_KEY)
                           recommendation = {
                               "type": "outdoor",
                               "place": selected_place,
                               "name": selected_place.get("name", "Unknown place"),
                               "description": description,
                               "image_url": image_url,
                               "activity_type": top_interest
                           }
                       else:
                           # Final fallback to indoor if no places found
                           prompt = build_llm_prompt_indoor(user, top_interest)
                           slot_context = f"You have {slot['duration_hours']} hours available on {slot['day']} from {slot['start_time']} to {slot['end_time']}."
                           prompt = prompt.replace("My context:", f"My context:\n- {slot_context}\n-")
                           
                           response = st.session_state.model.generate_content(prompt)
                           activity_description = response.text.strip()
                           main_keyword = extract_main_keywords(activity_description)
                           image_url = fetch_image_for_keyword(main_keyword, st.session_state.GOOGLE_MAPS_API_KEY)
                           
                           recommendation = {
                               "type": "indoor",
                               "name": f"Indoor {top_interest} Activity",
                               "description": activity_description,
                               "image_url": image_url,
                               "activity_type": top_interest
                           }
                   except Exception as e:
                       logging.error(f"Error processing outdoor for slot: {str(e)}")
                       # Emergency indoor fallback
                       recommendation = {
                           "type": "indoor",
                           "name": "Activity Suggestion",
                           "description": "Try something fun related to your interests!",
                           "image_url": None,
                           "activity_type": top_interest
                       }
               
               # Store recommendation in session state
               st.session_state.slot_recommendations[slot_id] = recommendation
   
   # Display all slots in a grid
   num_cols = 2  # Display 2 slots per row
   
   # Group slots by day
   saturday_slots = [slot for slot in st.session_state.weekend_slots if slot["day"] == "Saturday"]
   sunday_slots = [slot for slot in st.session_state.weekend_slots if slot["day"] == "Sunday"]
   
   # Display Saturday slots
   if saturday_slots:
       st.subheader("Saturday")
       rows = (len(saturday_slots) + num_cols - 1) // num_cols  # Ceiling division
       
       for row in range(rows):
           cols = st.columns(num_cols)
           for col_idx in range(num_cols):
               slot_idx = row * num_cols + col_idx
               if slot_idx < len(saturday_slots):
                   slot = saturday_slots[slot_idx]
                   slot_id = slot["id"]
                   
                   with cols[col_idx]:
                       # Get the recommendation for this slot
                       if slot_id in st.session_state.booked_slots:
                           recommendation = st.session_state.booked_slots[slot_id]
                           is_booked = True
                       elif slot_id in st.session_state.slot_recommendations:
                           recommendation = st.session_state.slot_recommendations[slot_id]
                           is_booked = False
                       else:
                           continue  # Skip if no recommendation (shouldn't happen)
                       
                       # Create a card-like UI
                       st.markdown(f"### {slot['start_time']}-{slot['end_time']}")
                       if is_booked:
                           st.success("✅ Booked")
                       
                       # Show event tag if it's an event
                       if recommendation.get("type") == "event":
                           st.info("🎟️ Event")
                       
                       if recommendation.get("image_url"):
                           st.image(recommendation["image_url"], width=200)
                       
                       # Truncate description if too long
                       description = recommendation["description"]
                       if len(description) > 100:
                           description = description[:97] + "..."
                       st.write(description)
                       
                       # Make card clickable
                       if st.button("View Details", key=f"quickview_{slot_id}"):
                           st.session_state.selected_slot_id = slot_id
                           st.session_state.current_view = "slot"
                           st.rerun()
   
   # Display Sunday slots
   if sunday_slots:
       st.subheader("Sunday")
       rows = (len(sunday_slots) + num_cols - 1) // num_cols  # Ceiling division
       
       for row in range(rows):
           cols = st.columns(num_cols)
           for col_idx in range(num_cols):
               slot_idx = row * num_cols + col_idx
               if slot_idx < len(sunday_slots):
                   slot = sunday_slots[slot_idx]
                   slot_id = slot["id"]
                   
                   with cols[col_idx]:
                       # Get the recommendation for this slot
                       if slot_id in st.session_state.booked_slots:
                           recommendation = st.session_state.booked_slots[slot_id]
                           is_booked = True
                       elif slot_id in st.session_state.slot_recommendations:
                           recommendation = st.session_state.slot_recommendations[slot_id]
                           is_booked = False
                       else:
                           continue  # Skip if no recommendation (shouldn't happen)
                       
                       # Create a card-like UI
                       st.markdown(f"### {slot['start_time']}-{slot['end_time']}")
                       if is_booked:
                           st.success("✅ Booked")
                           
                       # Show event tag if it's an event
                       if recommendation.get("type") == "event":
                           st.info("🎟️ Event")
                       
                       if recommendation.get("image_url"):
                           st.image(recommendation["image_url"], width=200)
                       
                       # Truncate description if too long
                       description = recommendation["description"]
                       if len(description) > 100:
                           description = description[:97] + "..."
                       st.write(description)
                       
                       # Make card clickable
                       if st.button("View Details", key=f"quickview_{slot_id}"):
                           st.session_state.selected_slot_id = slot_id
                           st.session_state.current_view = "slot"
                           st.rerun()
   
   # Back to main view button
   if st.button("← Back to main view", key="back_from_quickglance"):
       st.session_state.current_view = "main"
       st.rerun()

# View Management
if "current_view" not in st.session_state:
   st.session_state.current_view = "main"

# Determine which view to display based on current_view value
if st.session_state.current_view == "main":
   render_main_view()
elif st.session_state.current_view == "slot" and "selected_slot_id" in st.session_state:
   render_slot_recommendation(st.session_state.selected_slot_id)
elif st.session_state.current_view == "quick_glance":
   render_quick_glance_view()
else:
   # Fallback to main view if something is wrong
   st.session_state.current_view = "main"
   render_main_view()



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
