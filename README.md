# News Classification Website

## Overview

This project is an AI-powered News Classification Website that automatically collects news articles from the Hiru News API, classifies them using a trained BERT model, and displays the results through a web interface.

The system fetches real-time news data, processes the article content, predicts the news category using a fine-tuned BERT model, and serves the classified news through a Flask REST API.

## Features

* Real-time news collection from Hiru News API
* Automated data gathering
* News classification using BERT
* K-Means clustering support for news grouping
* Flask REST API backend
* JSON news responses
* Web-based frontend
* MySQL database support
* Real-time news updates

## Technologies Used

### Backend

* Python
* Flask
* Flask-CORS
* Requests
* Pandas
* NumPy

### Machine Learning

* BERT
* Transformers
* PyTorch
* Scikit-Learn

### Frontend

* HTML
* CSS
* JavaScript

### Database

* MySQL

## Project Structure

```text
news_classification_website/
│
├── newbackend.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── bert_news_model_clustering/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── model.safetensors
│
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## Installation

Clone the repository:

```bash
git clone https://github.com/sithum8363/news_classification_website.git

cd news_classification_website
```

Create a virtual environment:

```bash
python -m venv env
```

Activate the environment:

Windows:

```bash
env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Start the Flask backend:

```bash
python newbackend.py
```

Server:

```text
http://127.0.0.1:5000
```

## API Endpoints

### Home

```http
GET /
```

Response:

```json
{
  "message": "News API Running"
}
```

### Get Classified News

```http
POST /news
```

Returns classified news articles.

### Get News Article

```http
GET /news/<id>
```

Returns a specific article.

## Machine Learning Model

The project uses a fine-tuned BERT model stored in: https://huggingface.co/sithum8363/new_classifiaction_bert

```text
bert_news_model_clustering/
```

The model predicts news categories from article titles and content.

## Data Source

News data is collected from:

* Hiru News API
* Real-time API requests

Collected fields:

* News Title
* News Story
* Predicted Category
* Source

## Future Improvements

* Multi-language classification
* Sentiment analysis
* Fake news detection
* User authentication
* News recommendation system
* Dashboard analytics

## Author

Sithum Senath Marasinghe

Computer Science and Artificial Intelligence Student

Sri Lanka

## License

This project is intended for educational and research purposes.
