# Citi_Financial_Sentiment_Analysis
MarketPulse Pro: A Comprehensive Financial Sentiment Analysis Tool

MarketPulse Pro is an innovative tool designed to automate the sentiment analysis of financial texts, including earnings reports, PDFs, and news articles. Leveraging Natural Language Processing (NLP) and advanced Large Language Models (LLMs) like OpenAI’s GPT-3.5, it enables financial professionals to extract actionable insights efficiently and accurately.

This project was developed as part of an MSc Business Analytics dissertation in collaboration with Citibank and Bayes Business School.


Key Features
 - Automated Text Analysis: Extracts and analyzes sentiment from financial texts and PDFs.
- News Article Sentiment: Processes and evaluates financial news for market insights.
- Advanced NLP Techniques: Utilizes a lexicon-based approach and GPT-3.5 for nuanced sentiment classification.
- Data Visualization: Generates word clouds and sentiment trend charts for easy interpretation.
- User-Friendly Interface: Built using Streamlit for accessible deployment and interaction.


Future Enhancements
- Real-time sentiment analysis for breaking financial news.
- Enhanced financial lexicon for domain-specific terminology.
- Integration with automated trading systems for live trading insights.


Technologies Used
- Programming Language: Python
- Frameworks: Streamlit (Web App)

Libraries: 
- NLP: nltk, pdfplumber, wordcloud
- Visualization: matplotlib
- API Integration: openai, BeautifulSoup
- Models: GPT-3.5 by OpenAI


Project Structure
- app.py: The main Streamlit application script.
- Comparison.py: Script to compare the performance of various NLP models.
- Future_work_sentiment_analysis.ipynb: Jupyter notebook exploring real-time sentiment analysis concepts.
- requirements.txt: Python dependencies for running the project.
- Loughran-McDonald_MasterDictionary_1993-2023.csv: Domain-specific lexicon for financial text analysis.


How to Use

Clone the repository: git clone https://github.com/your-username/MarketPulse_Pro.git

Install dependencies: pip install -r requirements.txt

Run the Streamlit app: streamlit run app.py


Acknowledgments
Developed by: Nikhil Rawat
Academic Supervisor: Dr. Ahmad Abu-Khazneh
Industry Partner: Citibank
Institution: Bayes Business School (City, University of London)

