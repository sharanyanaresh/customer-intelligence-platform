# Data Sources

Raw data is downloaded locally and is not committed to Git.

- UCI Bank Marketing dataset: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
- CFPB Consumer Complaint Database: https://www.consumerfinance.gov/data-research/consumer-complaints/

Run `python src/data_pipeline/ingest.py --sample 5000` to create local samples and SHA-256 hash sidecar files.
