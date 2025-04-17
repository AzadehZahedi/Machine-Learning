# Machine-Learning

Project: Gender Prediction Using NLP Features from User Profile Text (Persian Web)

This mini-project aims to predict the gender of Iranian social networks users (Twitter and Instagram) based on their profile information (e.g., full name, username, and biography) by leveraging basic natural language processing (NLP) techniques and machine learning models. The data collected from Datak Website (Persian Website).

1. Text Cleaning
We define a function clean_text() to normalize and clean raw text data:
- Remove digits ( re.sub(r'\d+', '', text) )
- Replace HTML entities like &amp; with proper words ( re.sub(r'&amp;', 'and', text) )
- Simplify symbols like @ to letters ( re.sub(r'@', 'a', text) )
- Keep only alphabetic characters and whitespace ( re.sub(r'[^a-zA-Z\s.]', '', text) )

2. Missing Value Handling

3. Feature Engineering from Text
a. Biography Word Count
Added a new numeric feature that counts the number of words in the user's biography (Our assumption is that women are generally more likely to provide longer biographies).
b. Emoji Count
Counted the number of emojis in the biography using the emoji Python library, adding another numeric feature that could hint at user behavior or tone (Our assumption is that women may use emojis more frequently, especially in social contexts).

4. Text Vectorization
Text, which is unstructured data, must be transformed into a structured format. We used TfidfVectorizer (TF-IDF) to convert text data (combining fullname, username, and biography) into numerical vectors for machine learning (working only with numeric input).

5. Label Encoding
Encoded the target column gender using LabelEncoder.

6. Text-based Feature Prediction
Built a text-based classifier using XGBoost:
- Input: TF-IDF features from combined text columns
- Target: Gender
- Output: New feature ''gender_by_txt'', a gender prediction solely from text features

7. Final Model Training and Evaluation
After dropping the original text columns, we trained a final XGBClassifier using the engineered features (e.g., emoji count, word count, gender_by_txt), then evaluated the model using F1-score.

8. Evaluation
Generated predictions on the test set





  
