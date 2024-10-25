# Spotify-Popularity-Prediction
**Objective**

This Project develops a deep learning model to predict Spotify track popularity using audio features and metadata, enabling data-driven decision making with 81% accuracy through a stacked ensemble approach combining neural networks and random forest classifiers.

**Dataset:**

The data used in this project was collected and cleaned from Spotify's Web API using Python. The dataset includes the following features:
**track_id:** Unique Spotify identifier for each track.
**artists:** Names of artists, separated by ; for multiple artists.
**album_name:** Album title.
**track_name:** Track title.
**popularity:** Score from 0 to 100, indicating the track's popularity based on play count and recency.
**duration_ms:** Track duration in milliseconds.
**explicit:** Indicates explicit content (true/false).
**danceability:** Measures how suitable the track is for dancing (0.0 to 1.0).
**energy:** Tracks intensity and activity (0.0 to 1.0).
**key:** Musical key (integer representation).
**loudness:** Average loudness in dB.
**mode:** Modality (1 for major, 0 for minor).
**speechiness:** Detects the presence of spoken words.
**acousticness:** Likelihood of the track being acoustic (0.0 to 1.0).
**instrumentalness:** Probability of no vocals (0.0 to 1.0).
**liveness:** Indicates a live performance probability.
**valence:** Emotional tone (0.0 to 1.0).
**tempo:** Track tempo in beats per minute (BPM).
**time_signature:** Time signature of the track.
**track_genre:** Genre classification of the track.

**Detailed Workflow**

**A. Data Cleaning and Preprocessing**

**Missing Value Treatment**: Cleaned any missing values to ensure data consistency.
**Data Transformation**: Converted data into numerical formats and handled categorical data using encoding techniques.

**B. Exploratory Data Analysis (EDA)**

**Statistical Summaries:** Explored each feature to understand distributions and key trends.
**Correlation Analysis:** Analyzed the correlation matrix to identify relationships among variables.
**Visualization:** Created visualizations to better understand feature distributions and relationships with track popularity.

**C. Regression Approach**

**Model Selection:** Applied several regression models, including Linear Regression, Random Forest, and XGBoost.
**Hyperparameter Tuning:** Optimized parameters for each model.
**Feature Engineering:** Created additional features based on domain insights; however, results showed limited improvement.
**Conclusion:** Due to unsatisfactory results and data imbalance, the regression approach was discontinued.

**D. Classification Approach**

**Class Definition:** Tracks were categorized as Popular (popularity > 71) and Unpopular (popularity ≤ 71).

**E.Imbalance Handling**

Implemented multiple techniques due to high class imbalance.

**Oversampling:** **SMOTE and ADASYN** techniques were applied.

**Undersampling:** Reduced samples from the majority class to balance the dataset.

**Class Weights:** Used built-in class weighting options in XGBoost.

**F.Model Testing:** 

For each setup (raw data, capped outliers, and outlier removal), the following models were applied along with different Imbalance Handling:
  **Logistic Regression 
  Random Forest
  XGBoost**
Optimal Result: Without removing outliers, **XGBoost with oversampling** provided the most balanced and consistent results.

**G. Deep Learning Approach**

**Neural Network:** Created a neural network architecture, iteratively increasing layers and adjusting activation functions, dropouts, and regularization.
**Stacked Model:** Combined deep learning with ensemble techniques by stacking a neural network output with a Random Forest model. This approach yielded the best predictive performance.
**Model Saving:** The optimized stacked model was saved as a joblib file for easy deployment.

**H. Deployment**

A Streamlit application was developed to allow users to interactively predict the popularity of a track. This interface serves as a practical application of the model for real-time decision-making.

**Results:**

**1.Models Trained Without Removing Outliers:**

![Screenshot (179)](https://github.com/user-attachments/assets/4285ae30-75be-481c-8da4-46a5da68b43d)

**2.Models Trained by Capping Outliers:**

![Screenshot (180)](https://github.com/user-attachments/assets/76cf5123-3072-4a1d-aeca-c85ab914488b)

**3.Models Trained by Removing Outliers:**

![Screenshot (181)](https://github.com/user-attachments/assets/f81c8b04-5572-4725-a291-35716289eaca)

**4.Deep Learning Models:**

![Screenshot (182)](https://github.com/user-attachments/assets/c483c462-014d-4cbe-9099-4d52c4c25bbf)

**5.Best Model: Stacked Ensemble Model (Deep Learning NN + Random Forest)**

![Screenshot (177)](https://github.com/user-attachments/assets/eab5b83b-6ef3-49b4-b634-dac5ebd01aff)

![Screenshot (178)](https://github.com/user-attachments/assets/50b8e7f9-25bf-483e-ac9c-876347bff528)

**STREAMLIT UI:**

**Welcome Page:**

![Screenshot (193)](https://github.com/user-attachments/assets/285abe63-ab34-44fd-b654-cab4c060ac70)

**Prediction Page:**

![Screenshot (191)](https://github.com/user-attachments/assets/6f5fd475-0177-4400-9878-7953a37fc5fd)

![Screenshot (192)](https://github.com/user-attachments/assets/e1cdfb4e-480c-4232-a3bb-90ab1f9bf52d)











