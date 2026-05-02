# fp-NewsSentimentTracker
Scraping the internet (news outlets, reddit, etc.) to get a general outline for what the news of the day look like and summarize them into categories, while also generating sentiment scores about these category topics.

The program scrapes the top 50 headlines from three news sites (BBC, CNN, and Al Jazeera) and uses Jaccard Similarity to group similar headlines together. This produces a summary of the most prominent stories being covered, and a sentiment score is generated for each story.

## Setup

Install the required libraries with `pip install newspaper3k lxml_html_clean stop-words langdetect vaderSentiment`, then run the script with `python3 webScrapper.py`. The script writes three output files to the working directory: `clusters.csv`, `headlines.csv`, and `news_chart.png`.

## How it works

The program runs as a five-step pipeline. First, the `newspaper` library visits each site's homepage and extracts headlines, skipping any that fail to download. Each headline is then passed through `langdetect` to identify its language, with a secondary script check that verifies headlines flagged as English actually use Latin characters — this catches cases where the detector mislabels short non-Latin headlines. Next, each headline is lowercased, stripped of punctuation, and split into words, with stop words removed using the `stop-words` package, which provides word lists for many languages.

Headlines are then grouped using a clustering algorithm based on Jaccard Similarity, defined as the size of the intersection of two word sets divided by the size of their union. If a headline's similarity to any existing cluster exceeds the threshold (default 0.25), it joins that cluster; otherwise it starts a new one. Finally, each cluster's English headlines are scored with VADER, a sentiment analysis tool specialized for short text, which returns a compound score from -1 (very negative) to +1 (very positive). Non-English headlines are skipped and counted separately, since VADER is English-only and would silently return 0 for unrecognized text.

## Output files

`clusters.csv` contains one row per cluster, sorted by size (biggest stories on top), with columns for cluster ID, size, top keywords, sentiment label, sentiment score, and the number of headlines scored versus skipped for being non-English. `headlines.csv` contains one row per headline, tagged with its cluster ID along with the source, detected language, headline text, and URL. The two files are linked by `cluster_id` so we can sort the summary file and look up supporting headlines in the detail file. `news_chart.png` is a horizontal bar chart of the top 15 clusters by size, where bar length shows how many outlets cover the story and bar color shows sentiment (purple for positive, orange for negative, magenta-pink for neutral, and gray when no English headlines were available to score).

## Design decisions and limitations

Greedy clustering is order-dependent, meaning the sequence in which headlines are scraped can affect groupings, but for a daily news snapshot, I believe this is acceptable. The 0.25 similarity threshold worked well in testing. Lower values can cause unrelated headlines to merge, while higher values can fragment related stories. VADER is English-only, so non-English headlines have no sentiment score. VADER is also calibrated for casual social media text, so its scores tend to be muted on formal news prose, which means a score of -0.4 on a news cluster reflects genuinely negative coverage.

## Concepts and tools used

This project uses web scraping with `newspaper`, language detection with `langdetect` plus an additional fallback, multi-language stop-word filtering with the `stop-words` package, Jaccard Similarity, greedy single-pass clustering, sentiment analysis with VADER, data export with the `csv` standard library module, and data visualization with `matplotlib`.

## External help
- Similarity Clustering: Google + documentation + https://stackoverflow.com/questions/48323926/python-package-function-for-jaccard-similarity-between-sets
- I was having issue with news scraping, so I used Generative AI tools to find me the suitable packages, leading me to newspaper3k and lxml_html_clean
- I also used Generative AI tools to edit the print_summary() function since I had all the information, but was not sure how efficiently display them in clean way without error messaging.
- Bar chart code: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barh.html
- Sentiment analysis: https://www.geeksforgeeks.org/python/python-sentiment-analysis-using-vader/
- Suggested stop words issue solution: https://github.com/Alir3z4/stop-words; https://pypi.org/project/stop-words/
- 
