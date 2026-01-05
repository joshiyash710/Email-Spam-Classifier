import streamlit as st
import pickle
from sklearn.exceptions import NotFittedError

# --- Load vectorizer and trained model ---
try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model or vectorizer not found. Make sure 'model.pkl' and 'vectorizer.pkl' exist.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model/vectorizer: {e}")
    st.stop()

# --- Streamlit app layout ---
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")
st.title("📧 Email Spam Classifier")
st.write("Enter an email message below to check whether it is Spam or Not Spam.")

# Text input
email_text = st.text_area("Email content", height=200)

# Predict button
if st.button("Predict"):
    if not email_text.strip():
        st.warning("⚠️ Please enter some email content to predict.")
    else:
        try:
            # Transform the input
            input_vector = vectorizer.transform([email_text])
            # Predict
            prediction = model.predict(input_vector)[0]

            # Show result
            if prediction == 1:
                st.error("⚠️ This email is predicted as: Spam")
            else:
                st.success("✅ This email is predicted as: Not Spam (Ham)")

        except NotFittedError:
            st.error("❌ The model is not fitted. Please retrain your model first.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

# Optional: show example emails
st.markdown("---")
st.subheader("Example Emails to Test")
st.write("**Spam:** Congratulations! You won a lottery. Claim your prize now!")  
st.write("**Ham:** Hi, can we meet tomorrow for lunch?")
