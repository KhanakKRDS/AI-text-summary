# Import required libraries
# For making HTTP requests, so we can download files from the web
import requests
# For extracting text from PDF files
import pdfplumber
# A library by Hugging Face that provides pre-trained models and tokenizers for natural language processing tasks
from transformers import BartTokenizer, BartForConditionalGeneration

# Define the URL of the PDF to access -- replace this with your own URL to a PDF you want to summarize
url = 'https://arxiv.org/pdf/2102.04342'

# Download the PDF
# Make a GET request to the URL
# Open a file in write-binary mode
# And write the content of the PDF to the file
response = requests.get(url)
pdf_path = 'document.pdf'
with open(pdf_path, 'wb') as f:
    f.write(response.content)

# Extract text from the PDF
# Create an empty string to store the extracted text
text = ''
# Open the PDF and extract the text from the document page by page into the text string
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

# Load the pre-trained summarization model and tokenizer
# Load the tokenizer for BART, which will convert text into tokens that the model can understand
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
# Load the pre-trained BART model specifically fine-tuned for text summarization
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Function to summarize text
# Convert the input text into token IDs that the model can process
# Specify that the token IDs should be returned as PyTorch tensors
# Generate a summary of the text based on the token IDs, using beam search to optimize the output
# Convert the generated token IDs back into human-readable text
def summarize_with_transformers(text, max_length=150, min_length=50):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Generate and print summary
# Call the summarization function on the extracted text, specifying the desired length of the summary
summary = summarize_with_transformers(text, max_length=150, min_length=50)

# Finally, print your summary
print(summary)
        
