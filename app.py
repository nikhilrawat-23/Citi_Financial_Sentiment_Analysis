# Importing Libraries
import streamlit as st
import base64
import pandas as pd
import pdfplumber
import re
from fpdf import FPDF
import os
import openai
import matplotlib.pyplot as plt
import nltk
from nltk.util import ngrams
from collections import Counter
from wordcloud import WordCloud
import requests
from bs4 import BeautifulSoup
import time  # Import the time module

# Ensure you have the NLTK data downloaded
nltk.download('punkt')
nltk.download('punkt_tab')

# Streamlit app configuration
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="✔",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This is a Financial Sentiment Analyzer app created with Streamlit"
    }
)

# Function to set up OpenAI API key
def setup_openai_api_key():
    # Provide a text input for users to input their API key
    user_api_key = st.text_input("Enter your OpenAI API Key", type="password")

    if user_api_key:
        openai.api_key = user_api_key
        st.success("API key set successfully!")
    else:
        st.warning("Please enter your OpenAI API Key.")

# Call the function to set up API key
setup_openai_api_key()

# Define weights for specific phrases
phrase_weights = {
    "net income": 1.5,
    "gross margin": 1.2,
    "operating expenses": 1.0,
    "free cash flow": 1.3,
    "earnings per share": 1.4,
    "capital expenditure": 1.1,
    "revenue growth": 1.2,
    "debt equity ratio": 1.0,
    "return on investment": 1.5,
    "profit margin": 1.3,
    "cost of goods sold": 1.1,
    "working capital": 1.2,
    "current ratio": 1.1,
    "quick ratio": 1.1,
    "interest coverage ratio": 1.2,
    "dividend yield": 1.3,
    "price to earnings ratio": 1.4,
    "asset turnover": 1.2,
    "inventory turnover": 1.1,
    "debt service coverage": 1.3,
    "return on equity": 1.5,
    "capital structure": 1.1,
    "liquidity ratio": 1.0,
    "cash flow from operations": 1.4,
    "net profit margin": 1.3,
    "total shareholder return": 1.2,
    "earnings before interest and taxes": 1.3,
    "restructuring charges": -2.0,
    "decline": -1.5,
    "decrease": -1.5,
    "loss": -2.0,
    "negative impact": -1.5,
    "downturn": -2.0
}

# Load lexicon CSV file
lexicon_path = 'Loughran-McDonald_MasterDictionary_1993-2023.csv'
lexicon_df = pd.read_csv(lexicon_path)
lexicon_words = set(lexicon_df['Word'].str.lower())

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to clean text with selective n-grams and filtering
def clean_text_with_priority(text, lexicon_words, specific_phrases, phrase_weights, ngram_range=(1, 2)):
    # Convert to lowercase
    text = text.lower()

    # Remove unnecessary punctuation but keep financial symbols
    text = re.sub(r'[^\w\s$%]', '', text)

    # Tokenize text into words
    words = nltk.word_tokenize(text)

    # Generate unigrams and bigrams 
    all_ngrams = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        ngrams_list = list(ngrams(words, n))
        all_ngrams.extend(ngrams_list)
    
    # Convert n-grams to strings
    ngram_strings = [' '.join(ngram) for ngram in all_ngrams]

    # Filter and prioritize relevant n-grams, apply weights
    cleaned_ngrams = []
    weighted_phrases = []
    for ngram in ngram_strings:
        if ngram in lexicon_words or ngram in specific_phrases:
            if ngram in phrase_weights:
                # Apply the weight to the phrase by repeating it
                weighted_ngram = ' '.join([ngram] * int(phrase_weights[ngram] * 10))
                weighted_phrases.append(weighted_ngram)
            cleaned_ngrams.append(ngram)
        elif ngram in words:  # Preserve unigrams
            cleaned_ngrams.append(ngram)

    # Join the cleaned n-grams back into a cleaned text
    cleaned_text = ' '.join(cleaned_ngrams + weighted_phrases)

    return cleaned_text

# Function to get sentiment analysis for text input using OpenAI
def get_text_sentiment_analysis(text, temperature=0.05):
    if not openai.api_key:
        st.error("Please enter your OpenAI API Key to proceed.")
        return

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert Financial Analyst conducting sentiment analysis on textual input to evaluate potential market impacts."},
            {"role": "user", "content": f"Analyze the text for sentiment related to market impact. Consider both positive and negative aspects, and assess how this information could influence market sentiment. Provide an overall sentiment (positive, negative, neutral) and justify your conclusion in 2-3 sentences.\n\n\n\n\n\n{text}"}
        ],
        temperature=temperature
    )
    sentiment_text = response['choices'][0]['message']['content'].strip()
    
    # Extract a continuous sentiment score from the sentiment analysis
    score = sentiment_text.count('positive') * 1 - sentiment_text.count('negative') * 1
    return sentiment_text, score

# Function to get sentiment analysis using OpenAI for PDFs
def get_sentiment_analysis(text, temperature=0.05):
    if not openai.api_key:
        st.error("Please enter your OpenAI API Key to proceed.")
        return

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert Financial Analyst conducting trading sentiment analysis with a focus on negative financial impacts."},
            {"role": "user", "content": f"Analyze the text for immediate trading sentiment. Emphasize negative financial outcomes and heavily weigh terms indicating financial downturns or losses. Determine the accurate immediate trading sentiment (positive, negative, neutral) based on weighted assessment. Justify the conclusion in 2 sentences.\n\n\n\n\n\n{text}"}
        ],
        temperature=temperature
    )
    sentiment_text = response['choices'][0]['message']['content'].strip()
    
    # Extract a continuous sentiment score from the sentiment analysis
    score = sentiment_text.count('positive') * 1 - sentiment_text.count('negative') * 1
    return sentiment_text, score

# Function to get sentiment analysis for news articles using OpenAI
def get_news_sentiment_analysis(text, temperature=0.05):
    if not openai.api_key:
        st.error("Please enter your OpenAI API Key to proceed.")
        return

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert Financial Analyst analyzing news articles to determine their impact on the market sentiment."},
            {"role": "user", "content": f"Analyze the text for immediate trading sentiment. Mention positive, negative and neutral statements with necessary details. Evaluate and outweigh each point whose significance for short-term and immediate impacts is high. Determine the accurate immediate trading sentiment (positive, negative, neutral) based on weighted assessment. Justify the conclusion. Show the conclusion first. Display the conclusion first for each file. Keep the output format consistent for every file.\n\n\n\n\n\n{text}"}
        ],
        temperature=temperature
    )
    sentiment_text = response['choices'][0]['message']['content'].strip()
    
    # Extract a continuous sentiment score from the sentiment analysis
    score = sentiment_text.count('positive') * 1 - sentiment_text.count('negative') * 1
    return sentiment_text, score

# Function to generate a word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Function to plot sentiment scores line chart
def plot_sentiment_scores(df):
    fig, ax = plt.subplots()
    ax.plot(df['PDF Name'], df['Sentiment Score'], marker='o')
    ax.set_xlabel('PDF Name')
    ax.set_ylabel('Sentiment Score')
    ax.set_title('Sentiment Scores for PDFs')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# Function to fetch articles from URLs
def fetch_articles(urls, headers):
    data = []
    for url in urls:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('h1').get_text() if soup.find('h1') else 'No title found'
            
            # Extract article text
            paragraphs = soup.find_all('p')
            article_text = "\n".join([paragraph.get_text() for paragraph in paragraphs])
            
            # Append to data list
            data.append({
                "title": title,
                "text": article_text
            })
        else:
            st.error(f"Failed to fetch the article from {url}, status code: {response.status_code}")
    return data

# Function to encode an image to base64
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Encode your background images
img_main = get_img_as_base64("Background3.png")

# Custom CSS for background images and styling
def add_custom_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_main}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .css-1d391kg {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 10px;
            padding: 20px;
        }}
        .stButton > button {{
            background-color: #1f77b4;
            color: white;
            border-radius: 10px;
            height: 50px;
            width: 100%;
            font-size: 18px;
            border: none;
        }}
        .stButton > button:hover {{
            background-color: #0056a1;
        }}
        .stTextInput > div > div > input {{
            font-size: 18px;
            padding: 10px;
            background-color: #ffffff;
            border-radius: 10px;
            border: 2px solid #000000;
        }}
        .stFileUploader > div > div > div > button {{
            font-size: 18px;
            border-radius: 10px;
            background-color: #1f77b4;
            color: white;
            border: none;
        }}
        .stFileUploader > div > div > div > button:hover {{
            background-color: #0056a1;
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: #000000;
        }}
        .css-1aumxhk, .css-1avcm0n, .css-1kyxreq, .css-1d391kg, .css-1offfwp, .css-pkbazv {{
            background-color: rgba(255, 255, 255, 0.85) !important;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

# Define the PDF class here so it's available throughout the script
class PDF(FPDF):
    def body(self, body):
        self.set_font('Arial', '', 14)
        # Handling utf-8 encoded strings
        body = body.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 10, body)
        self.ln()

# Function to create and download outcome PDF safely
def create_and_download_pdf(filename, sentiment_analysis, score):
    try:
        outcome_pdf = PDF()
        outcome_pdf.add_page()
        outcome_pdf.set_font('Arial', 'B', 16)
        outcome_pdf.multi_cell(0, 10, f"Sentiment Analysis Outcome for {filename}\n")
        outcome_pdf.set_font('Arial', '', 14)
        outcome_pdf.multi_cell(0, 10, f"{sentiment_analysis}\n")
        outcome_pdf.multi_cell(0, 10, f"Sentiment Score: {score}\n")
        outcome_pdf_path = f'outcome_output_{filename}.pdf'
        outcome_pdf.output(outcome_pdf_path)

        with open(outcome_pdf_path, "rb") as file:
            st.download_button(label=f"Download Sentiment Analysis Outcome", data=file, file_name=outcome_pdf_path)
    
    except UnicodeEncodeError:
        st.warning(f"Unable to generate PDF for {filename} due to encoding issues. However, you can view the sentiment analysis above.")

# Function to create and download cleaned text PDF safely
def create_and_download_cleaned_pdf(filename, cleaned_text):
    try:
        cleaned_pdf = PDF()
        cleaned_pdf.add_page()
        cleaned_pdf.body(cleaned_text)
        output_pdf_path = f'cleaned_output_{filename}.pdf'
        cleaned_pdf.output(output_pdf_path)

        with open(output_pdf_path, "rb") as file:
            st.download_button(label=f"Download Cleaned PDF for {filename}", data=file, file_name=output_pdf_path)
    
    except UnicodeEncodeError:
        st.warning(f"Unable to generate cleaned PDF for {filename} due to encoding issues.")

# Streamlit app main function
def main():
    st.title("Financial Sentiment Analyzer ✔")

    st.markdown("## Analyze PDFs, Text, and News Articles for Financial Data")

    # Add a refresh button
    if st.button("Refresh"):
        st.session_state.clear()

    # Dropdown menu to choose analysis mode
    analysis_mode = st.selectbox(
        "Choose Analysis Mode",
        ["Analyze Text", "Analyze PDF", "Analyze News Article"]
    )

    if analysis_mode == "Analyze Text":
        st.header("Input Your Text")
        user_input = st.text_area("Enter your text here:", height=300)

        # Define the specific phrases within the scope of text analysis
        specific_phrases = st.multiselect(
            "Select specific financial phrases to prioritize:",
            options=list(phrase_weights.keys()),
            default=None
        )

        if st.button("Analyze"):
            if user_input:
                st.write("Processing entered text...")
                
                start_time = time.time()  # Start timer
                
                cleaned_text = clean_text_with_priority(user_input, lexicon_words, specific_phrases, phrase_weights)

                with st.spinner("Analyzing sentiment..."):
                    sentiment_analysis, score = get_text_sentiment_analysis(cleaned_text, temperature=0.05)
                    end_time = time.time()  # End timer

                    if sentiment_analysis:
                        st.subheader("Sentiment Analysis")
                        st.write(sentiment_analysis)

                        st.subheader("Word Cloud")
                        generate_word_cloud(cleaned_text)

                        elapsed_time = end_time - start_time
                        st.write(f"Time taken for analysis: {elapsed_time:.2f} seconds")  # Display elapsed time

                        # Attempt to create and download the outcome PDF
                        create_and_download_pdf("Input Text", sentiment_analysis, score)

    elif analysis_mode == "Analyze PDF":
        st.header("Upload PDF Files")
        uploaded_files = st.file_uploader("Or upload PDF files", type="pdf", accept_multiple_files=True)

        all_sentiments = []
        pdf_names = []

        # Select phrases to prioritize
        specific_phrases = st.multiselect(
            "Select specific financial phrases to prioritize:",
            options=list(phrase_weights.keys()),
            default=None
        )

        # Adjust the temperature parameter for the sentiment analysis
        temperature = st.slider("Select temperature for sentiment analysis (lower is more deterministic):", 0.0, 1.0, 0.05)

        st.markdown("---")  # Adds a horizontal divider

        if st.button("Analyze"):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    st.write(f"Processing {uploaded_file.name}...")
                    pdf_names.append(uploaded_file.name)
                    
                    start_time = time.time()  # Start timer

                    # Extract text from the uploaded PDF
                    with st.spinner("Extracting text from PDF..."):
                        extracted_text = extract_text_from_pdf(uploaded_file)

                    # Clean the extracted text
                    with st.spinner("Cleaning text..."):
                        cleaned_text = clean_text_with_priority(extracted_text, lexicon_words, specific_phrases, phrase_weights)

                    # Perform sentiment analysis on the cleaned text
                    with st.spinner("Analyzing sentiment..."):
                        sentiment_analysis, score = get_sentiment_analysis(cleaned_text, temperature)
                        end_time = time.time()  # End timer

                        if sentiment_analysis:
                            all_sentiments.append(score)

                            # Display the conclusion for the current PDF
                            st.subheader(f"Conclusion for {uploaded_file.name}")
                            st.write(sentiment_analysis)

                            st.subheader("Word Cloud")
                            generate_word_cloud(cleaned_text)

                            elapsed_time = end_time - start_time
                            st.write(f"Time taken for analysis: {elapsed_time:.2f} seconds")  # Display elapsed time

                            # Attempt to create and download the cleaned PDF
                            create_and_download_cleaned_pdf(uploaded_file.name, cleaned_text)

                            # Attempt to create and download the outcome PDF
                            create_and_download_pdf(uploaded_file.name, sentiment_analysis, score)

                # Plot the sentiment scores
                sentiment_df = pd.DataFrame({
                    'PDF Name': pdf_names,
                    'Sentiment Score': all_sentiments
                })
                plot_sentiment_scores(sentiment_df)

    elif analysis_mode == "Analyze News Article":
        st.header("Input News Article URL(s)")
        urls_input = st.text_area("Enter the URL(s) of the news articles (separate by comma if multiple):")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

        if st.button("Fetch and Analyze"):
            if urls_input:
                urls = [url.strip() for url in urls_input.split(",")]
                st.write("Fetching articles...")
                
                with st.spinner("Fetching and analyzing articles..."):
                    start_time = time.time()  # Start timer

                    articles = fetch_articles(urls, headers)
                    
                    for article in articles:
                        st.subheader(article['title'])
                        sentiment_analysis, score = get_news_sentiment_analysis(article['text'], temperature=0.05)
                        end_time = time.time()  # End timer

                        if sentiment_analysis:
                            st.write(sentiment_analysis)

                            st.subheader("Word Cloud")
                            generate_word_cloud(article['text'])

                            elapsed_time = end_time - start_time
                            st.write(f"Time taken for analysis: {elapsed_time:.2f} seconds")  # Display elapsed time

                            # Attempt to create and download the outcome PDF
                            create_and_download_pdf(article['title'], sentiment_analysis, score)

if __name__ == "__main__":
    main()
