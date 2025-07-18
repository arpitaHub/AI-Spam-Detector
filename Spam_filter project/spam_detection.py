# Import libraries
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



#  Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]  # Keep only needed columns
df.columns = ['label', 'message']  # Rename columns



#  Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


#  Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)   # keep letters/numbers only
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



# ✅ Apply cleaning
df['message'] = df['message'].apply(clean_text)




# ✅ TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['message']).toarray()
y = df['label']




#  Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



#  Train model
model = LogisticRegression()
model.fit(X_train, y_train)




#  Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))





#  User input check
print("\n----- SPAM CHECKER -----")
user_input = input("Enter message to check: ")
cleaned = clean_text(user_input)
vector = tfidf.transform([cleaned])
result = model.predict(vector)




#  Output result
# if result[0] == 1:
#     print("🔴 Alert!! This message is SPAM")
# else:
#     print("🟢 Relax!! This message is NOT spam")





# save model
import pickle
from sklearn.linear_model import LogisticRegression
pickle.dump(model,open('spam_model.pkl','wb'))
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
