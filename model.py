import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv('students.csv')

# Preprocess the data to include the new attributes
def preprocess_data(df):
    def create_attribute_string(row):
        weighted_attributes = []
        
        # Assign weights based on boolean attributes
        weighted_attributes.append('music_high' if row['likes_music'] else 'music_low')
        weighted_attributes.append('study_night_high' if row['studies_at_night'] else 'study_night_low')
        weighted_attributes.append('smokes_high' if row['smokes'] else 'smokes_low')
        weighted_attributes.append('health_issues_high' if row['health_issues'] else 'health_issues_low')
        weighted_attributes.append('likes_reading_high' if row['likes_reading'] else 'likes_reading_low')
        weighted_attributes.append('drinks_coffee_high' if row['drinks_coffee'] else 'drinks_coffee_low')
        weighted_attributes.append('exercises_regularly_high' if row['exercises_regularly'] else 'exercises_regularly_low')
        weighted_attributes.append('group_study_high' if row['prefers_group_study'] else 'group_study_low')
        
        return ' '.join(weighted_attributes)
    
    df['attributes'] = df.apply(create_attribute_string, axis=1)
    return df

# Preprocess the dataset
df = preprocess_data(df)

# Create a TF-IDF matrix
tfidf = TfidfVectorizer(
    token_pattern=r'\b\w+\b',
    ngram_range=(1, 2), 
    max_df=0.8,  
    min_df=1     
)
tfidf_matrix = tfidf.fit_transform(df['attributes'])

# Prediction function
def predict_matches(user_attributes, top_n=5, min_score=0.1):
    def create_input_attribute_string(attributes):
        weighted_attributes = []
        
        # Assign weights based on user input
        weighted_attributes.append('music_high' if attributes['likes_music'] else 'music_low')
        weighted_attributes.append('study_night_high' if attributes['studies_at_night'] else 'study_night_low')
        weighted_attributes.append('smokes_high' if attributes['smokes'] else 'smokes_low')
        weighted_attributes.append('health_issues_high' if attributes['health_issues'] else 'health_issues_low')
        weighted_attributes.append('likes_reading_high' if attributes['likes_reading'] else 'likes_reading_low')
        weighted_attributes.append('drinks_coffee_high' if attributes['drinks_coffee'] else 'drinks_coffee_low')
        weighted_attributes.append('exercises_regularly_high' if attributes['exercises_regularly'] else 'exercises_regularly_low')
        weighted_attributes.append('group_study_high' if attributes['prefers_group_study'] else 'group_study_low')
        
        return ' '.join(weighted_attributes)
    
    input_attribute_string = create_input_attribute_string(user_attributes)
    
    # Transform input to TF-IDF vector
    input_vector = tfidf.transform([input_attribute_string])
    
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)[0]
    
    # Apply noise and clip scores for variability
    similarity_scores = similarity_scores * 0.9 + np.random.normal(0, 0.05, len(similarity_scores))
    similarity_scores = np.clip(similarity_scores, 0, 1)
    
    # Prepare results
    results = list(zip(df['name'], similarity_scores, df['student_id']))
    results.sort(key=lambda x: x[1], reverse=True)
    filtered_results = [
        match for match in results 
        if match[1] > min_score
    ][:top_n]

    return filtered_results

# Example input
user_attributes = {
    'likes_music': True,
    'studies_at_night': True,
    'smokes': False,
    'health_issues': False,
    'likes_reading': True,
    'drinks_coffee': False,
    'exercises_regularly': True,
    'prefers_group_study': False
}

# Run predictions
for _ in range(3):
    matches = predict_matches(user_attributes)
    print("Matches:")
    for name, score, student_id in matches:
        print(f"Name: {name}, Similarity Score: {score:.4f}, Student ID: {student_id}")
    print("---")
