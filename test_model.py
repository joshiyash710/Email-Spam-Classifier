import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- Sample emails dataset ---
emails = [
    "Congratulations! You won a lottery. Claim your prize now!",  # Spam
    "Get cheap meds online, click here!",                           # Spam
    "Hi, can we meet tomorrow for lunch?",                          # Ham
    "Don't forget the meeting at 10 AM.",                           # Ham
]

labels = [1, 1, 0, 0]  # 1=Spam, 0=Ham

# --- Vectorize emails ---
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# --- Train model ---
model = MultinomialNB()
model.fit(X, labels)

# --- Save model and vectorizer ---
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and vectorizer saved!")

# --- Test examples ---
test_emails = [
    "Congratulations! You won a lottery. Claim your prize now!",  # Spam
    "Hi, can we meet tomorrow for lunch?",                        # Ham
]

X_test = vectorizer.transform(test_emails)
preds = model.predict(X_test)

for email, pred in zip(test_emails, preds):
    print(email)
    print("Predicted:", "Spam" if pred == 1 else "Ham")
    print("---")
