import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize #used for analyzing words and sentences.
from collections import Counter
from sklearn.feature_extraction.text import TfidVectorize # TF_IDF for better word importance
from flask import Flask, request, render_template #helpful for website
from transformers import BartTokenizer, BartForConditionalGeneration# pre-trained model


##nltk.download("punkt") #used to split the text into sentences
##nltk.download("stopwords") # used to remove words like the, and, is

app = Flask(__name__, template_folder = "./")
tokenizer = BartTokenizer.from_pretrained ("fine-tune(BART)") #BART model
model = BartForConditionalGeneration.from_pretrained("fine-tune(BART)")

def understand_text(text): #Tokenizes the input text, convert to lowercase amd removes stopwords
    sentences = sent_tokenize(text) #splitt text into sentences
    stop_words = set(stopwords.words("english")) #define common stopwords
    words = [
        [word for word in word_tokenize(sentence.lower()) if word.isalnum and word not in stop_words]
        for sentence in sentences #filter out non-alphanumeric tokens and stopwords
    ]#isalnum(alphanumeric)in above code the non-alphanumeric tokens and stopwords are filtered out then printed

    return sentences, words #returns processed sentences and words


def important_words(words): #uses TF_IDF for better identification of important words rather than word frequency
    corpus = ["". join(sentence) for sentence in words] #convert sentences into a text corpus
    vectorizer = TfidfVectorizer()
    tfidf_matrix = TfidfVectorizer.fit_transform(corpus) 
    return Counter([word for sentence in words for word in sentence])# count how many times the word occurred

def get_sentence_scores(sentences, word_freq): #scores each sentence based on the frequency of its words, providing a
    #metric for dertermining the importance of each 
    sentence_scores = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())#tokenizing sentence
        score = sum(word_freq.get(word, 0) for word in words if word.isalnum())#summing word frequencies
        sentence_scores[sentence] = score/len(words) if len(words) > 0 else 0 #normalizing score
    return sentence_scores #returns a dictionary of sentence scores

def get_summary(text): #generates a summary of the text using a pre-trained model.
    if not text or text.strip() == "":
        return "" #if input text is empty, return an empty summary
    sentences, words =  understand_text(text) #process text
    word_freq = important_words(words) #count word frequency
    sentence_scores = get_sentence_scores(sentences, word_freq) #score sentences

    #Extractive summarization (select top 5 important sentences)
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:5]
    important_text = "".join(top_sentences) # select 5 important words and give it to the model to summarize

    #Abstractive summarization (using BART model)
    inputs = tokenizer(important_text, return_tensors="pt", max_length=1024, truncation=True) #tokenize
    summary_ids = model.generate(inputs["input_ids"], max_length=130, min_length=50, do_sample=False) #generate summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True) #Decode input

    return summary #return final summarized text

@app.route("/", methods=["GET"]) # handles user input and displays the summary a web page

def index(): #on a post request, it retrieves user input, process the text to extract
    #sentencs and words, calculates their frequencies, and generated a summary. Renders the result on an HTML template
    return render_template("AI-text-summary.html", og = "", summary = "")

@app.route("/process", methods=["POST"])
def process(): #handles user input, process the text, and displays the summary
    original_text = request.form.get("og", "") #get input from the webpage
    summary = get_summary(original_text) if original_text else "" #generate summary
    return render_template("AI-text-summary.html", og = original_text, summary = summary)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=3000, debug=True) #start web server

