import pandas as pd
import numpy as np
import random

# Arabic names array (transliterated male names)
arabic_names = [
    "Ahmad", "Mohammed", "Ali", "Omar", "Youssef", "Hassan", "Ibrahim", "Khaled", "Abdullah", "Saeed",
    "Hussein", "Adel", "Kareem", "Tariq", "Jamal", "Rami", "Sami", "Nabil", "Majed", "Waleed",
    "Ziad", "Faisal", "Hisham", "Bassam", "Mahmoud", "Mustafa", "Osama", "Reda", "Tamer", "Wael",
    "Amjad", "Bilal", "Fahad", "Hani", "Imad", "Kamal", "Maher", "Nasser", "Rashid", "Zaher"
]

# Generate random student data
def generate_student_data(num_students=100):
    # Shuffle the names array to randomly assign names
    names = arabic_names.copy()
    random.shuffle(names)
    
    # If we need more names than we have, we'll add numbers to existing names
    if num_students > len(names):
        extended_names = []
        for i in range(num_students):
            if i < len(names):
                extended_names.append(names[i])
            else:
                base_name = names[i % len(names)]
                extended_names.append(f"{base_name} {i//len(names) + 1}")
        names = extended_names

    data = {
        'student_id': range(1000, 1000 + num_students),
        'name': names[:num_students],
        'likes_music': np.random.choice([True, False], num_students),
        'studies_at_night': np.random.choice([True, False], num_students),
        'smokes': np.random.choice([True, False], num_students, p=[0.2, 0.8]),  # Less likely to smoke
        'health_issues': np.random.choice([True, False], num_students, p=[0.1, 0.9]),  # Less likely to have health issues
        'likes_reading': np.random.choice([True, False], num_students),
        'drinks_coffee': np.random.choice([True, False], num_students),
        'exercises_regularly': np.random.choice([True, False], num_students),
        'prefers_group_study': np.random.choice([True, False], num_students)
    }
    
    return pd.DataFrame(data)

# Generate the dataset
df = generate_student_data(100)

# Export to CSV
df.to_csv('students.csv', index=False)

# Display first few rows of the generated dataset
print("Sample of generated data:")
print(df.head())

# Display some basic statistics
print("\nDataset Statistics:")
print(f"Total number of students: {len(df)}")
for column in df.columns:
    if column not in ['student_id', 'name']:
        true_percentage = (df[column].sum() / len(df)) * 100
        print(f"{column}: {true_percentage:.1f}% True") 