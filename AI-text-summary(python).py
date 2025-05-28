import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize #used for analyzing words and sentences.
from collections import Counter
from flask import Flask, request, render_template #helpful for website
from transformers import pipeline # pre-trained model

##nltk.download("punkt") #used to split the text into sentences
##nltk.download("stopwords") # used to remove words like the, and, is

app = Flask(__name__, template_folder = "./")
summarizer = pipeline("summarization", model = "facebook/bart-large-cnn")#BART model

def understand_text(text): #Tokenizes the input text, convert to lowercase amd removes stopwords
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))

    words = [word_tokenize(sentence.lower()) for sentence in sentences] # each sentence is converted to lowercase
    words = [[word for word in sentence if word.isalnum() and word not in stop_words] for sentence in words]
    #isalnum(alphanumeric)in above code the non-alphanumeric tokens and stopwords are filtered out then printed

    return sentences, words

##text =("")
##sentences, words = understand_text(text)

##print("sentences:", sentences)
##print("changes words:", words)


def important_words(words): # counts the frequency of each word in processed text, identify most significant terms
    word_counts = Counter([word for sentence in words for word in sentence])# count how many times the word occurred
    return word_counts

#word_freq = important_words(words)
##print("Word Frequency:", word_freq)

def get_sentence_scores(sentences, word_freq): #scores each sentence based on the frequency of its words, providing a
    #metric for dertermining the importance of each 
    sentence_scores = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        sentence_scores[sentence] = sum(word_freq.get(word, 0) for word in words if word in word_freq) /len(words)
    return sentence_scores

#sentence_scores = get_sentence_scores(sentences, word_freq)
##print("sentence scores:", sentence_scores)

def get_summary(text): #generates a summary of the text using a pre-trained model.
    if not text or text.strip() == "":
        return ""
    summary_list = summarizer(text, max_length=130, do_sample = False)
    return summary_list[0]["summary_text"]

@app.route("/", methods=["GET", "POST"]) # handles user input and displays the summary and frequency on a web page

def index(): #on a post request, it retrieves user input, process the text to extract
    #sentencs and words, calculates their frequencies, and generated a summary. Renders the result on an HTML template
    return render_template("AI-text-summary.html")

@app.route("/process", methods=["GET", "POST"])
def process():
    summary = ""
    text = ""
    if request.method == "POST":
        text = request.form.get("new") #Retrieves user input. "form" word is used for post

    else:
        text = request.args.get("og") # args word is used of get request

    if text:
        summary = get_summary(sentence_scores)
    return render_template("AI-text-summary.html", summary = summary, og=text)

##print("Summary:", summary)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=3000)





