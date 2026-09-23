# Sentiment Dashboard for Product Reviews

An AI-powered web app that analyzes product reviews and classifies them as Positive, Negative, or Neutral using DistilBERT — with live Plotly charts and a summary insight.

**Live Demo:** _(add your Streamlit Cloud link after deployment)_

---

## Features

- Paste reviews directly or upload a CSV file
- Per-review sentiment label with confidence score
- Bar chart, donut chart, and per-review confidence chart
- Auto-generated summary insight based on results
- Download labeled results as CSV

## Tech Stack

- Python, Streamlit
- HuggingFace Transformers (DistilBERT fine-tuned on SST-2)
- Plotly, pandas

## Run Locally

```bash
git clone https://github.com/MehXarii/sentiment-dashboard.git
cd sentiment-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
sentiment-dashboard/
├── app.py                  # Main Streamlit app
├── utils/
│   ├── analyzer.py         # Model loading and inference
│   └── charts.py           # Plotly chart generation
├── sample_reviews.csv      # Test data
├── requirements.txt
└── README.md
```

## About

Built by **Mehak Ansari** — BS Computer Science, University of Central Punjab.  
GitHub: [MehXarii](https://github.com/MehXarii)  
LinkedIn: [mehak-ansari-80144a246](https://linkedin.com/in/mehak-ansari-80144a246)
