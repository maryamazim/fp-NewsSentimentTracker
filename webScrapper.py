# Scrape 50 headlines from BBC news, CNN, and Al Jazeera

import newspaper

# News sources
news_sites = {
    "BBC": "https://www.bbc.com",
    "CNN": "https://www.cnn.com",
    "Al Jazeera": "https://www.aljazeera.com"
}

all_data = []

for name, url in news_sites.items():
    print(f"\nScraping {name}...")

    # Build URL to find article and store it in varibale 'paper'
    try:
        paper = newspaper.build(url, memoize_articles=False)
        print(f"Found {len(paper.articles)} article links")
    except Exception as e:
        print(f"Could not build paper for {name}: {e}")
        continue

    # Include a counter variable since we will skip articles that fail the process,
    # meaning that we may need to parse through more than 50 times
    count = 0

    # Go through each newspaper
    for article in paper.articles:
        if count >= 50:
            break

        try:
            article.download()
            article.parse()

            if article.title and article.title.strip():
                all_data.append({
                    "source": name,
                    "headline": article.title.strip()
                })
                count += 1
                
        # Skip if unabel to access
        except Exception as e:
            print(f"Skipping one article from {name}: {e}")

print("\nCollected headlines:")
for item in all_data:
    print(item)
