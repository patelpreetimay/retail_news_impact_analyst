# RNIA — Retail News Impact Analyst

RNIA analyzes financial news articles and identifies important financial events, stance, and impact scores.

## Pipeline Flow

News Scraping → Preprocessing → Annotation → Event Taxonomy → Model Training → Impact Scoring → Final Output

## Modules

- **Scrapers** — collect financial news
- **Preprocessing** — clean and normalize text
- **Annotation** — label financial events
- **Taxonomy** — classify event categories
- **Models** — train machine learning models
- **Scoring** — compute impact scores
- **Pipeline** — execute the entire workflow

## How to Run

Step 1: Install dependencies

```
pip install -r requirements.txt
```

Step 2: Run main pipeline

```
python main.py
```

## Notes

The system integrates multiple modules including data scraping, preprocessing, event classification, and financial impact scoring. The pipeline ensures modular execution and easy experimentation with different models.
