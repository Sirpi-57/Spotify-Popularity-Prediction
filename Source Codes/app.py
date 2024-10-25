import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import joblib
import tensorflow as tf

# Streamlit app configuration
icon = "Spotify_Primary_Logo_RGB_Green.png"
st.set_page_config(page_title="Spotify Popularity Prediction", page_icon=icon)

# Initialize the session state for page management
if 'page' not in st.session_state:
    st.session_state.page = "welcome"  # Start on the welcome page

# Function to show the main page
def main_page():

    st.title("Spotify Popularity Predictor")
    st.subheader("Lets Predict If Your Song will be Popular!")

    # Custom CSS for labels below the sliders
    st.markdown(
        """
        <style>
        .slider-container {
            position: relative;
            height: 50px;
        }
        .slider-label {
            position: absolute;
            top: 0.03px;
            font-size: 12px;
        }
        .slider-label-low {
            left: 0;
        }
        .slider-label-mid {
            left: 50%;
            transform: translateX(-50%);
        }
        .slider-label-high {
            right: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Function to create a labeled slider
    def labeled_slider(label, min_value, max_value, default_value, low_label, mid_label, high_label):
        st.header(label)
        value = st.slider(label, min_value, max_value, default_value)

        # Custom label positions below the slider
        st.markdown(
            f"""
            <div class="slider-container">
                <span class="slider-label slider-label-low">{low_label}</span>
                <span class="slider-label slider-label-mid">{mid_label}</span>
                <span class="slider-label slider-label-high">{high_label}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        return value

    # Input song duration using sliders for minutes and seconds
    st.header("Enter your song's duration")
    minutes = st.slider("Minutes:", 0, 10, 0)
    seconds = st.slider("Seconds:", 0, 59, 0)

    # Sliders for existing features
    danceability = labeled_slider(
        "Rate the danceability of your song:",
        0.0, 10.0, 5.0,
        "Least Danceable", "Medium Danceable", "Most Danceable"
    )

    energy = labeled_slider(
        "Rate the energy of your song:",
        0.0, 10.0, 5.0,
        "Least Energetic", "Medium Energetic", "Most Energetic"
    )

    speechiness = labeled_slider(
        "Rate the speechiness of your song:",
        0.0, 10.0, 5.0,
        "Non-speech/Music", "Mix/Rap", "AudioBook/Podcast"
    )

    valence = labeled_slider(
        "Rate the emotional tone of your track:",
        0.0, 10.0, 5.0,
        "Sad", "Neutral", "Happy"
    )

    # New input for Tempo
    tempo = st.header("Overall Estimated Tempo (BPM)")
    tempo_value = st.slider("Tempo (0-250 BPM)", 0, 250, 120)

    # Input for Is Explicit
    explicit = st.radio("Is your song Explicit?", ("Yes", "No"))
    explicit_value = 1 if explicit == "Yes" else 0

    # Input for Is Mode
    mode = st.radio("What is your song Mode?", ("Major", "Minor"))
    mode_value = 1 if mode == "Major" else 0

    # Input for Is Acoustic
    is_acoustic = st.radio("Is your song acoustic?", ("Yes", "No"))
    is_acoustic_value = 1 if is_acoustic == "Yes" else 0

    # Input for Is Instrumental
    is_instrumental = st.radio("Is your song instrumental?", ("Yes", "No"))
    is_instrumental_value = 1 if is_instrumental == "Yes" else 0

    # Input for Is Live Performance
    is_live = st.radio("Is it a live performance?", ("Yes", "No"))
    is_live_value = 1 if is_live == "Yes" else 0

    # Input for Key
    st.header("Select the key of your song (0-11)")
    key = st.slider("Key", 0, 11, 0)

    # Encoding the key
    key_encoded = {f'key_{i}': 0 for i in range(12)}
    key_encoded[f'key_{key}'] = 1

    # Input for time_signature

    st.header("Select the time_signature of your song (0-5")
    time_signature = st.slider("Key", 0, 5, 0)

    # Encoding the time_signature
    time_signature_encoded = {f'time_signature_{i}': 0 for i in range(6)}
    time_signature_encoded[f'time_signature_{time_signature}'] = 1

    # Display scaled values
    # Convert the input duration to milliseconds
    duration_ms = (minutes * 60 + seconds) * 1000
    st.write(f"The song's duration in milliseconds is: {duration_ms} ms")
    st.write(f"Danceability (0-1 scale): {danceability / 10:.1f}")
    st.write(f"Energy (0-1 scale): {energy / 10:.1f}")
    st.write(f"Speechiness (0-1 scale): {speechiness / 10:.1f}")
    st.write(f"Valence (0-1 scale): {valence / 10:.1f}")
    st.write(f"Tempo: {tempo_value} BPM")
    st.write(f"Explicit: {explicit_value} (1 for Yes, 0 for No)")
    st.write(f"Mode: {mode_value} (1 for Major, 0 for Minor)")
    st.write(f"Is Acoustic: {is_acoustic_value} (1 for Yes, 0 for No)")
    st.write(f"Is Instrumental: {is_instrumental_value} (1 for Yes, 0 for No)")
    st.write(f"Is Live Performance: {is_live_value} (1 for Yes, 0 for No)")
    st.write(f"Encoded Key: {key_encoded}")
    st.write(f"Encoded time_signature: {time_signature_encoded}")

    # Genre category options
    genre_categories = {
        "Rock and Metal": [
            "track_genre_alt-rock", "track_genre_black-metal", "track_genre_death-metal",
            "track_genre_grindcore", "track_genre_hard-rock", "track_genre_heavy-metal",
            "track_genre_metal", "track_genre_metalcore", "track_genre_punk",
            "track_genre_punk-rock", "track_genre_rock", "track_genre_rock-n-roll",
            "track_genre_grunge", "track_genre_goth", "track_genre_hardcore",
            "track_genre_emo", "track_genre_psych-rock", "track_genre_alternative",
            "track_genre_rockabilly", "track_genre_j-rock"
        ],
        "Electronic and Dance": [
            "track_genre_ambient", "track_genre_chicago-house", "track_genre_club",
            "track_genre_dance", "track_genre_dancehall", "track_genre_deep-house",
            "track_genre_detroit-techno", "track_genre_disco", "track_genre_drum-and-bass",
            "track_genre_dub", "track_genre_dubstep", "track_genre_edm",
            "track_genre_electro", "track_genre_electronic", "track_genre_garage",
            "track_genre_house", "track_genre_idm", "track_genre_minimal-techno",
            "track_genre_progressive-house", "track_genre_techno", "track_genre_trance",
            "track_genre_hardstyle"
        ],
        "Pop and Mainstream": [
            "track_genre_afrobeat", "track_genre_anime", "track_genre_british",
            "track_genre_cantopop", "track_genre_children", "track_genre_comedy",
            "track_genre_country", "track_genre_indie", "track_genre_indie-pop",
            "track_genre_j-pop", "track_genre_k-pop", "track_genre_latin",
            "track_genre_latino", "track_genre_mandopop", "track_genre_pop",
            "track_genre_pop-film", "track_genre_power-pop", "track_genre_romance",
            "track_genre_sad", "track_genre_salsa", "track_genre_samba"
        ],
        "Classical and Traditional": [
            "track_genre_classical", "track_genre_new-age", "track_genre_opera",
            "track_genre_piano"
        ],
        "Hip-Hop and R&B": [
            "track_genre_hip-hop", "track_genre_r-n-b", "track_genre_reggaeton"
        ],
        "World and Cultural": [
            "track_genre_brazil", "track_genre_french", "track_genre_german",
            "track_genre_indian", "track_genre_iranian", "track_genre_j-dance",
            "track_genre_j-idol", "track_genre_kids", "track_genre_malay",
            "track_genre_pagode", "track_genre_swedish", "track_genre_tango",
            "track_genre_turkish", "track_genre_world-music", "track_genre_mpb", 
            "track_genre_sertanejo", "track_genre_spanish" 
        ],
        "Folk, Blues, and Jazz": [
            "track_genre_bluegrass", "track_genre_blues", "track_genre_folk",
            "track_genre_forro", "track_genre_gospel", "track_genre_jazz", "track_genre_funk", 
            "track_genre_reggae", "track_genre_soul"
        ],
        "Experimental and Niche Genres": [
            "track_genre_breakbeat", "track_genre_groove", "track_genre_industrial",
            "track_genre_synth-pop", "track_genre_trip-hop", "track_genre_honky-tonk",
            "track_genre_show-tunes", "track_genre_ska", "track_genre_singer-songwriter"
        ],
        "Other": [
            "track_genre_disney", "track_genre_sleep", "track_genre_study",
            "track_genre_party", "track_genre_guitar", "track_genre_chill",
            "track_genre_happy"
        ]
    }

    # Step 1: Genre category selection
    st.header("Select the genre category")
    selected_category = st.selectbox("Genre Category", options=list(genre_categories.keys()))

    # Step 2: Genre selection based on the selected category
    if selected_category:
        st.header(f"Select the genre from {selected_category}")
        selected_genre = st.selectbox("Genre", options=genre_categories[selected_category])

        # Encoding the selected genre
        genre_encoded = {genre: 0 for genre in [genre for sublist in genre_categories.values() for genre in sublist]}
        genre_encoded[selected_genre] = 1

        # Display the encoded genre
        #st.write("Encoded Genre:")
        #st.write(genre_encoded)

    @st.cache_resource
    def load_models():
        model_file_path = "model_files/"  # Define the file path here

        try:
            # Load the scaler
            scaler = joblib.load(model_file_path + 'scaler.joblib')
            
            # Load the neural network model
            nn_model = tf.keras.models.load_model(model_file_path + 'nn_model.h5')
            
            # Load the random forest model
            rf_model = joblib.load(model_file_path + 'rf_model.joblib')
            
            # Load the feature names
            with open(model_file_path + 'feature_names.json', 'r') as f:
                feature_names = json.load(f)
            
            return scaler, nn_model, rf_model, feature_names
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None, None, None, None


    # Load models
    scaler, nn_model, rf_model, feature_names = load_models()

    def prepare_input_data(duration_ms, danceability, energy, speechiness, valence,
                        tempo_value, explicit_value, mode_value, is_acoustic_value, 
                        is_instrumental_value, is_live_value, key_encoded, 
                        time_signature_encoded, genre_encoded):
        features = {
            'duration_ms': duration_ms,
            'danceability': danceability / 10,
            'energy': energy / 10,
            'speechiness': speechiness / 10,
            'valence': valence / 10,
            'tempo': tempo_value,
            'explicit': explicit_value,
            'mode': mode_value,
            'is_acoustic': is_acoustic_value,
            'is_instrumental': is_instrumental_value,
            'is_live': is_live_value,
            **key_encoded,
            **time_signature_encoded,
            **genre_encoded
        }
        input_df = pd.DataFrame([features], columns=feature_names)
        return input_df

    def make_prediction(input_df, scaler, nn_model, rf_model):
        try:
            # Print shapes and types for debugging
            #st.write("Input DataFrame Shape:", input_df.shape)
            
            # Scale the features
            scaled_features = scaler.transform(input_df)
            #st.write("Scaled Features Shape:", scaled_features.shape)
            
            # Get NN predictions
            nn_pred_proba = nn_model.predict(scaled_features)
            #st.write("Neural Network Output Shape:", nn_pred_proba.shape)
            
            # Reshape nn_pred_proba if needed
            if len(nn_pred_proba.shape) > 1 and nn_pred_proba.shape[1] > 1:
                nn_pred_proba = nn_pred_proba[:, 0]  # Take first column if multiple outputs
            #nn_pred_proba = nn_pred_proba.reshape(-1, 1)
            
            # Stack features
            stacked_features = np.column_stack((scaled_features, nn_pred_proba))
            #st.write("Stacked Features Shape:", stacked_features.shape)
            
            # Get final predictions
            final_prediction = rf_model.predict(stacked_features)
            final_proba = rf_model.predict_proba(stacked_features)
            
            #st.write("Final Prediction Shape:", final_prediction.shape)
            #st.write("Final Probability Shape:", final_proba.shape)
            
            return final_prediction, final_proba
            
        except Exception as e:
            st.error(f"Error in make_prediction: {str(e)}")
            st.error(f"Error type: {type(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None, None

    # When making predictions:
    if st.button('Predict Popularity'):
        if scaler is None or nn_model is None or rf_model is None:
            st.error("Models failed to load. Please check the model files.")
        else:
            try:
                # Prepare input data
                input_df = prepare_input_data(
                    duration_ms=duration_ms,
                    danceability=danceability,
                    energy=energy,
                    speechiness=speechiness,
                    valence=valence,
                    tempo_value=tempo_value,
                    explicit_value=explicit_value,
                    mode_value=mode_value,
                    is_acoustic_value=is_acoustic_value,
                    is_instrumental_value=is_instrumental_value,
                    is_live_value=is_live_value,
                    key_encoded=key_encoded,
                    time_signature_encoded=time_signature_encoded,
                    genre_encoded=genre_encoded
                )
                
                # Get predictions
                predictions = make_prediction(input_df, scaler, nn_model, rf_model)
                
                # Check if predictions is None
                if predictions is None:
                    st.error("Failed to make predictions")
                    
                final_prediction, final_prediction_proba = predictions
                
                if final_prediction is not None:
                    st.subheader('Prediction Results')
                    
                    # Get probabilities for both classes
                    prob_unpopular = final_prediction_proba[0][0]  # Probability of being unpopular
                    prob_popular = final_prediction_proba[0][1]    # Probability of being popular
                    
                    # Display the prediction and probabilities
                    st.write("### Probability Breakdown:")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Probability of being Popular", f"{prob_popular:.1%}")
                    with col2:
                        st.metric("Probability of being Unpopular", f"{prob_unpopular:.1%}")
                    
                    # Show the final prediction
                    st.write("### Final Prediction:")
                    if final_prediction[0] == 1:
                        st.success(f'Your song is predicted to be POPULAR')
                    else:
                        st.warning(f'Your song is predicted to be UNPOPULAR')
                    
                    # Add interpretation
                    st.write("### Understanding the Prediction:")
                    st.write("""
                    - The model gives probabilities for both classes (Popular and Unpopular)
                    - These probabilities always sum to 100%
                    - If the probability of being Popular is > 50%, the song is classified as Popular
                    - If the probability of being Popular is ≤ 50%, the song is classified as Unpopular
                    - The further the probability is from 50%, the more confident the model is in its prediction
                    """)
                    
            except Exception as e:
                st.error(f"An error occurred in prediction display: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")


def show_welcome_page():
    st.title("Welcome to Spotify Prediction!")
    st.image("1645596871156.png")
    if st.button("Let's Go"):
        st.session_state.page = 'main_page'
if st.session_state.page == 'welcome':
    st.markdown(
    """
    <div style="text-align: center;">
    <h1>Welcome To Spotify Popularity Prediction App!</h1>
    <h3>"Check If Your Track is a Hit!"</h3>
    </div>
    """,
    unsafe_allow_html=True
    )
    # Center the button using Streamlit
    col1, col2, col3 = st.columns([10, 12, 4])
    with col2:
        if st.button("Let's Go"):
            st.session_state.page = 'prediction'
    # Center an image below the button
    col1, col2, col3 = st.columns([0.5, 2000, 0.5])
    with col2:
        st.image("1645596871156.png")
else:
    main_page()

