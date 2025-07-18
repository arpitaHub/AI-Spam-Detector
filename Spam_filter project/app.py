from flask import Flask , render_template , request
import pickle 
import re

app=Flask(__name__)


# load trained model nd vectorizer
model=pickle.load(open("spam_model.pkl","rb"))
vectorizer=pickle.load(open("vectorizer.pkl","rb"))

# clean text that the user input
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)  # ✅ Keep only letters and numbers
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    message=request.form['message']
    cleaned=clean_text(message)
    vector=vectorizer.transform([cleaned])
    result=model.predict(vector)
    if result[0]==1:
        output="🔴 This message is SPAM!!" 
    else:
        output="🟢 Relax! This message is NOT spam"

    return render_template('index.html',prediction=output,message=message)

if __name__ == '__main__':
    app.run(debug=True)
