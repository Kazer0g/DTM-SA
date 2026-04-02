# Distributed Text Mining and Sentiment Analysis

## Overview

This project studies a distributed MapReduce-based pipeline for text mining and sentiment analysis. It is designed to process large text corpora and produce aggregated insights, such as word frequencies and sentiment-aware term rankings.

## Goals

- Extract word frequency distributions from input text.
- Apply sentiment analysis to identify positive and negative terms.
- Group results by document metadata (e.g., source, category, author) and rank top terms.
- Validate correctness and performance with reproducible checks.

## Input Data

Acceptable inputs include:

- Tweets
- Product reviews
- News headlines or articles
- Web-scraped text

## Output

Expected outputs include:

- Term frequency statistics per corpus or group
- Positive/negative term rankings per group
- Performance metrics for scalability tests

## Validation Checklist

- [ ] Verify term frequencies against sample texts
- [ ] Compare sentiment classification accuracy to expected labels
- [ ] Demonstrate scalability with document count and chunk size

## Notes

Keep this README as a reference for implementation and testing. No additional features are added in this change; the structure and clarity are improved only.