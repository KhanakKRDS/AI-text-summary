import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize #used for analyzing words and sentences.
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer # TF_IDF for better word importance
from flask import Flask, request, render_template #helpful for website
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM# pre-trained model

##nltk.download("punkt") #used to split the text into sentences
##nltk.download("stopwords") # used to remove words like the, and, is


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained ("../My-tune-model-og/My-tune-model-og") #BART model
model = AutoModelForSeq2SeqLM.from_pretrained("../My-tune-model-og/My-tune-model-og")
print("Model loaded.")

app = Flask(__name__, template_folder="templates")

def understand_text(text): #Tokenizes the input text, convert to lowercase amd removes stopwords
    sentences = sent_tokenize(text) #splitt text into sentences
    stop_words = set(stopwords.words("english")) #define common stopwords
    words = [
        [word for word in word_tokenize(sentence.lower()) if word.isalnum() and word not in stop_words]
        for sentence in sentences #filter out non-alphanumeric tokens and stopwords
    ]#isalnum(alphanumeric)in above code the non-alphanumeric tokens and stopwords are filtered out then printed

    return sentences, words #returns processed sentences and words


def important_words(words): #uses TF_IDF for better identification of important words rather than word frequency
    corpus = [" ".join(sentence) for sentence in words] #convert sentences into a text corpus
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    word_freq = dict(zip(feature_names, tfidf_matrix.sum(axis=0).A1)) #get word importance based on TF_IDF
    
    return Counter(word_freq) #return a frequency count based on importance

def get_sentence_scores(sentences, word_freq):#scores each sentence based on the importance of their words,
    #ensuring key sentences are selected
    sentence_scores = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())#tokenize sentence
        score = sum(word_freq.get(word, 0) for word in words if word.isalnum())#score based on word importance 
        sentence_scores[sentence] = score/len(words) if len(words) > 0 else 0 #normalizing score
    return sentence_scores #returns a dictionary of sentence scores

def get_summary(text): #generates a summary of the text using a pre-trained model, combined with extractive summarization.
    if text.strip() == "":
        return "" #if input text is empty, return an empty summary
    sentences, words =  understand_text(text) #process text
    word_freq = important_words(words) #identify important words
    sentence_scores = get_sentence_scores(sentences, word_freq) #sentence score based on word significance
    
    #Extractive summarization (select top 5 important sentences)
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:5] #select top-ranked sentences
    important_text = " ".join(top_sentences) #prepare input for abstractive summarization

    #Abstractive summarization (using BART model)
    inputs = tokenizer(important_text, return_tensors="pt", max_length=1024, truncation=True) #tokenize input
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=130,
        min_length=50,
        num_beams = 5, #improves summary quality using beam search
        repetition_penalty = 1.2, #reduces redundant phrases
        length_penalty = 0.7, #prevents excessive trunction
        do_sample=False
        )#generate summary
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True) #Decode model output

    return summary #return final summarized text

@app.route("/", methods=["GET"])
def index(): #on a post request, it retrieves user input, process the text to extract
    #sentencs and words, calculates their frequencies, and generated a summary. Renders the result on an HTML template
    print(">>> Home page accessed")
    return render_template("AI-text-summary.html", og = "", summary = "")

@app.route("/process", methods=["POST"])
def process(): #handles user input, process the text, and displays the summary
    print(">>> Process route hit")
    try:
        original_text = request.form.get("og", "").strip() #get input from the webpage
        print(f"Original text length: {len(original_text)}")
        summary = get_summary(original_text) if original_text else "Invalid input" #generate summary or return an error
        print("Summary generated")
    except Exception as e:
        summary = f"Error: {str(e)}" #handle unexpected errors
        print(f"Exception occurred: {e}")
    return render_template("AI-text-summary.html", og = original_text, summary = summary)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True) #start web server

