import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize #used for analyzing words and sentences.
from collections import Counter
from flask import Flask, request, render_template #helpful for website

##nltk.download("punkt") #used to split the text into sentences
##nltk.download("stopwords") # used to remove words like the, and, is

app = Flask(__name__, template_folder = "C:\\Khanak\\AI-text-summary")

def understand_text(text): #Tokenizes the input text into sentences and words, removing stopwords
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))

    words = [word_tokenize(sentence.lower()) for sentence in sentences] # each sentence is converted to lowercase
    words = [[word for word in sentence if word.isalnum() and word not in stop_words] for sentence in words]
    #isalnum(alphanumeric)in above code the non-alphanumeric tokens and stopwords are filtered out then printed

    return sentences, words

text =("")
sentences, words = understand_text(text)

##print("sentences:", sentences)
##print("changes words:", words)


def important_words(words): # counts the frequency of each word in processed text
    word_counts = Counter([word for sentence in words for word in sentence])# count how many times the word occurred
    
    return word_counts

word_freq = important_words(words)
##print("Word Frequency:", word_freq)


def get_sentence_scores(sentences, word_freq): #scores each sentence based on the frequency of its words
    sentence_scores = {}

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        sentence_scores[sentence] = sum(word_freq.get(word, 0) for word in words if word in word_freq) /len(words)

    return sentence_scores

sentence_scores = get_sentence_scores(sentences, word_freq)
##print("sentence scores:", sentence_scores)


def get_summary(sentence_scores, top_n=2): # Sorts the sentence based on their scores in descending order
    #and selects the top N sentences to form the summary
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse = True)
    summary_sentences = [sentence for sentence, _ in sorted_sentences[:top_n]]
    return"". join(summary_sentences)


@app.route("/", methods=["GET"]) # handles user input and displays the summary and frequency on a web page

def index(): #on a post request, it retrieves user input, process the text to extract
    #sentencs and words, calculates their frequencies, and generated a summary. Renders the result on an HTML template
    summary = ""
    word_feq = {}
    return render_template("AI-text-summary.html")

@app.route("/process", methods=["GET", "POST"])
def process():
    summary = ""
    if request.method == "POST":
        text = request.form.get("new") #Retrieves user input. "form" word is used for post

    else:
        text = request.args.get("og") # args word is used of get request
        sentences, words = understand_text(text)
        word_freq = important_words(words)
        sentence_scores = get_sentence_scores(sentences, word_freq)
        summary = get_summary(sentence_scores)
    return render_template("AI-text-summary.html", summary = summary)

summary = get_summary(sentence_scores)
##print("Summary:", summary)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=3000)





