# Machine-Learning

Project 1: Gender Prediction Using NLP Features from User Profile Text (Persian Web)

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Project 2-1: User Search Behavior Analysis (MrBilit Dataset-A traveling website)
This project analyzes user search behavior from the MrBilit website, an Iranian travel platform. 
Each entry in the dataset corresponds to a user's single search action, including the type of service requested and location-based text fields.
In parallel, a dataset of Iranian cities with demographic information from Wikipedia is used to enrich the search data and compare behavioral trends with population data.

1. Service Popularity Analysis
We analyzed the popularity of different service types (e.g., bus, airplane, taxi, hotel)

2. Preprocessing AcceptString
Many values in ''AcceptString'' column contain additional suffixes like - ''پایانه'' (indicating terminal name). We extracted the clean city name using a helper function.

3. Focus on Transport Services (Excluding Hotels)
To analyze transportation behavior, we filtered out hotel-related entries. Then, we identified the top 20 most frequently searched cities for transport services and visualized them.

4. Merging with City Demographics
We enriched the transport search data by merging it with the city info (iran_cities.csv) using the Farsi city name. This allowed us to explore which provinces users are searching for most.

5. Exploring Gaps Between Population and Search Behavior
To see whether highly populated cities were not being searched as frequently as expected, we identified large cities (population > 500,000) that did not appear in the top 20 search destinations. This can reveal potential gaps in demand vs. population, helping with product/service planning or marketing focus.

Project 2-2: Smart Search Suggester for Iranian Cities
This system enhances city name suggestions using fuzzy matching, typo correction (keyboard layout issues), and usage frequency data to suggest relevant destinations when a user begins typing in a travel search bar.

1. Preprocess City Names & Search Strings
- Strips out terminal tags (''- پایانه'')
- Removes non-printable Unicode characters
- Applies to city names, AcceptString, and test set inputs

2. Filter to Transportation Services
Focuses only on land transport to simplify and target the scope (''data_zamini'').

3. Fix Keyboard Typos (EN --> FA)
Corrects input strings typed with an English keyboard layout instead of Persian.

4. Fuzzy Matching for City Suggestions
- Uses Levenshtein distance to compare user input with official city names
- Returns top N suggestions ranked by similarity

5. Fallback: Frequent Cities Suggestion
If fuzzy matches return fewer than 5 cities, fills the remaining slots with the most commonly searched cities.

6. Combining All Suggestions
Final suggestion list = fuzzy matches + (optional) frequent cities

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Project 3: Persian News Text Categorization with SVM
To build a machine learning model that classifies Persian news articles into relevant topic categories (e.g., politics, sports, music) using the article's title as input text.
Dataset Source: Provided by Yektanet, a Persian content platform.

1. Data Preprocessing & Balancing
- Dropped Irrelevant Columns:
Only the title column was used for modeling. Other fields like description, text_content, url, etc., were dropped to reduce noise and dimensionality.
- Label Encoding:
Category (target variable) was encoded into numeric format using LabelEncoder.
- Class Balancing:
Oversampling with RandomOverSampler from imbalanced-learn was used to balance class distributions in the training data.

2. Text Preprocessing
- Custom preprocessor function:
Removed numbers, Latin characters, punctuations
Removed Persian stopwords (using hazm)
- Tokenization:
Used hazm.word_tokenize for proper Persian language token splitting.

3. Feature Extraction & Modeling
Pipeline (using sklearn.Pipeline):
- CountVectorizer: with bigrams (ngram_range=(1, 2)) and stopword-filtered tokens
- TfidfTransformer: to weight token importance
- LinearSVC: SVM classifier with hinge loss

4. Model Performance
- Training F1 Score: 0.978 (weighted average)
This high score indicates very strong performance on the training set, though cross-validation or a separate validation split would be needed to confirm generalization.

5. Predictions
- Predictions on the test set were made using the trained model and converted back to the original category labels with inverse_transform.
- Output was saved in a DataFrame for submission



  
