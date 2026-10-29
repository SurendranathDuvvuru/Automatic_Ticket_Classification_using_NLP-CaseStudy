# Automatic Ticket Classification using Natural Language Processing (NLP)
## Project Overview
This NLP case study groups customer complaints by topic and aims to classify new tickets into the relevant service category.
## Objective
Automate complaint routing to help customer-support teams direct issues to the appropriate department.
## Technology Stack
- Python, Pandas, NumPy
- NLTK, spaCy, TextBlob
- Matplotlib, Seaborn, Plotly, WordCloud
- Scikit-learn and Gensim
## Workflow
1. Load complaint records from a JSON file into a Pandas DataFrame.
2. Remove records with blank complaint text and prepare the data.
3. Clean text by lowercasing, removing brackets, punctuation, and words containing numbers.
4. Lemmatize words and remove selected parts of speech.
5. Explore complaint lengths, word clouds, and frequent unigrams, bigrams, and trigrams.
6. Convert complaint text into TF-IDF features.
7. Use Non-negative Matrix Factorization (NMF) for unsupervised topic modeling.
8. Explore topic counts using coherence scores and inspect the top words per topic.
9. Map topics to five categories: Bank Account Services, Credit Card/ Prepaid Card, Theft/ Dispute Reporting, Mortgage/ Loan, and Others.
10. Plan a supervised classifier to predict categories for new complaints.
## Data
Input file `complaints-2021-05-14_08_16.json` is used from Google Drive folder during code execution.
## Run
Install the required libraries, configure the JSON file path, and execute the notebook cells in order in Jupyter or Google Colab.
## Outcome
A complaint-routing workflow that assigns customer messages to service categories and supports faster ticket handling.
