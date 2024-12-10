import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random


# ce code pour generer la dataset sample, on a utilise la function random pour 'randomization' de les noms et puis ecrier les donnes dans un fichier csv

########################################################################

# arabic_names = ["Ahmad", "Mohammed", "Ali", "Omar", "Youssef", "Hassan", "Ibrahim", "Khaled", "Abdullah", "Saeed",
#     "Hussein", "Adel", "Kareem", "Tariq", "Jamal", "Rami", "Sami", "Nabil", "Majed", "Waleed",
#     "Ziad", "Faisal", "Hisham", "Bassam", "Mahmoud", "Mustafa", "Osama", "Reda", "Tamer", "Wael",
#     "Amjad", "Bilal", "Fahad", "Hani", "Imad", "Kamal", "Maher", "Nasser", "Rashid", "Zaher"]

# def generate_student_data(num_students=100):
#     names = arabic_names.copy()
#     random.shuffle(names)


#     if num_students > len(names):
#         extended_names = []
#         for i in range(num_students):
#             if i < len(names):
#                 extended_names.append(names[i])
#             else:
#                 base_name = names[i % len(names)]
#                 extended_names.append(f"{base_name} {i//len(names) + 1}")
#         names = extended_names

#     data = {
#         'student_id': range(1000, 1000 + num_students),
#         'name': names[:num_students],
#         'likes_music': np.random.choice([True, False], num_students),
#         'studies_at_night': np.random.choice([True, False], num_students),
#         'smokes': np.random.choice([True, False], num_students, p=[0.2, 0.8]),
#         'health_issues': np.random.choice([True, False], num_students, p=[0.1, 0.9]),
#         'likes_reading': np.random.choice([True, False], num_students),
#         'drinks_coffee': np.random.choice([True, False], num_students),
#         'exercises_regularly': np.random.choice([True, False], num_students),
#         'prefers_group_study': np.random.choice([True, False], num_students)
#     }
    
#     return pd.DataFrame(data)

# ########################################################################

# df = generate_student_data(100)


# df.to_csv('students.csv', index=False)


# print("Sample of generated data:")
# print(df.head())


# print("\nDataset Statistics:")
# print(f"Total number of students: {len(df)}")
# for column in df.columns:
#     if column not in ['student_id', 'name']:
#         true_percentage = (df[column].sum() / len(df)) * 100
#         print(f"{column}: {true_percentage:.1f}% True")

########################################################################## 

df = pd.read_csv('students.csv')

def preprocess_data(df):
    def create_attribute_string(row):
        weighted_attributes = []
        
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

df = preprocess_data(df)

tfidf = TfidfVectorizer(
    token_pattern=r'\b\w+\b',
    ngram_range=(1, 2), 
    max_df=0.8,  
    min_df=1     
)
tfidf_matrix = tfidf.fit_transform(df['attributes'])

def predict_matches(user_attributes, top_n=5, min_score=0.1):
    def create_input_attribute_string(attributes):
        weighted_attributes = []
        
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
    
    input_vector = tfidf.transform([input_attribute_string])
    
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)[0]
    
    similarity_scores = similarity_scores * 0.9 + np.random.normal(0, 0.05, len(similarity_scores))
    similarity_scores = np.clip(similarity_scores, 0, 1)
    
    results = list(zip(df['name'], similarity_scores, df['student_id']))
    results.sort(key=lambda x: x[1], reverse=True)
    filtered_results = [
        match for match in results 
        if match[1] > min_score
    ][:top_n]

    return filtered_results

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

for _ in range(3):
    matches = predict_matches(user_attributes)
    print("Matches:")
    for name, score, student_id in matches:
        print(f"Name: {name}, Similarity Score: {score:.4f}, Student ID: {student_id}")
    print("---")
