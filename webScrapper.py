# News headline scraper, clusterer, and sentiment analyzer.
# Pipeline: scrape, detect language, tokenize, cluster (Jaccard), score sentiment (VADER), generate csv report and grpahic.

import string
import html
from datetime import datetime
import csv
import matplotlib.pyplot as plt

import newspaper
from stop_words import get_stop_words, AVAILABLE_LANGUAGES
from langdetect import detect, DetectorFactory, LangDetectException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# langdetect is randomized by default, so set a seed for replicability
DetectorFactory.seed = 0


news_sites = {
    "BBC": "https://www.bbc.com",
    "CNN": "https://www.cnn.com",
    "Al Jazeera": "https://www.aljazeera.com"
}

num_headlines = 50
cluster_threshold = 0.25
output_html = "news_report.html"

# Source-branding noise that shows up across many headlines from the same
noise_wrds = {"bbc", "cnn", "aljazeera", "al", "jazeera", "news"}

# stop-word sets per language so we don't rebuild them on every call
stop_word = {}

# One shared sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()


### Step 1: scrape headlines ###

def scrape_headlines(news_sites, per_site=50):
    """Visit each site's homepage, collect up to `per_site` headlines."""
    headlines = []

    for name, url in news_sites.items():
        print(f"\nScraping {name}...")

        try:
            paper = newspaper.build(url, memoize_articles=False)
            print(f"  Found {len(paper.articles)} article links")
        except Exception as e:
            print(f"  Could not build paper for {name}: {e}")
            continue

        count = 0
        for article in paper.articles:
            if count >= per_site:
                break
            try:
                article.download()
                article.parse()

                if article.title and article.title.strip():
                    headlines.append({
                        "source": name,
                        "headline": article.title.strip(),
                        "url": article.url
                    })
                    count += 1
            except Exception as e:
                print(f"  Skipping one article from {name}: {e}")

        print(f"  Collected {count} headlines from {name}")

    return headlines


### Step 2: language detection and tokenization ###

def looks_like_english_script(text):
    """
    Returns True if the text is predominantly Latin alphabet. This works on top of langdetect,
    which can mislabel short headlines. Requires that at least 80% of letters be ASCII.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False

    ascii_letters = [ch for ch in letters if ch.isascii()]
    return len(ascii_letters) / len(letters) >= 0.8


def detect_language(text):
    """
    Return a 2-letter language code for the text. Falls back to 'en' on
    detector failure, but ONLY if the text actually looks like Latin
    script - otherwise we mark it 'unknown' so it gets skipped by VADER.
    """
    try:
        code = detect(text)
    except LangDetectException:
        code = "en" if looks_like_english_script(text) else "unknown"

    if code == "en" and not looks_like_english_script(text):
        return "unknown"

    if code in AVAILABLE_LANGUAGES:
        return code

    # Detected a language we don't have stop words for
    return code if not looks_like_english_script(text) else "en"

def get_stop_word_set(lang_code):
    """
    Return the stop-word set for a language. source-noise words are added on top
    so 'BBC' / 'CNN' style branding gets stripped regardless of headline language.
    """
    if lang_code not in stop_word:
        try:
            words = set(get_stop_words(lang_code))
        except Exception:
            words = set(get_stop_words("en"))
        stop_word[lang_code] = words | noise_wrds

    return stop_word[lang_code]


def tokenize(headline, lang_code):
    """
    Lowercase, strip punctuation, split, and remove stop words for the
    detected language. Returns set of meaningful words.
    """
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    cleaned = headline.lower().translate(translator)

    stop_words = get_stop_word_set(lang_code)
    words = cleaned.split()
    return {w for w in words if w not in stop_words and len(w) > 2}


### Step 3: Jaccard similarity and clustering ###

def jaccard_similarity(set_a, set_b):
    """|A ∩ B| / |A ∪ B|, with both-empty handled."""
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def cluster_headlines(headlines, threshold=0.25):
    """
    Greedy single-pass clustering on tokenized headlines.
    Each item gets language-detected and tokenized once; clusters track
    the union of words across their members for comparison.
    """
    # Pre-process: attach language and tokens to each headline so we don't redo the work on every cluster comparison.
    for item in headlines:
        item["language"] = detect_language(item["headline"])
        item["tokens"] = tokenize(item["headline"], item["language"])

    clusters = []

    for item in headlines:
        words = item["tokens"]
        if not words:
            continue

        best_cluster = None
        best_score = 0.0
        for cluster in clusters:
            score = jaccard_similarity(words, cluster["word_set"])
            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster is not None and best_score >= threshold:
            best_cluster["headlines"].append(item)
            best_cluster["word_set"] |= words
        else:
            clusters.append({
                "headlines": [item],
                "word_set": set(words)
            })

    return clusters


### Step 4: sentiment scoring ###

def score_cluster_sentiment(cluster):
    """
    Run VADER only on headlines that are both flagged as English AND
    pass the Latin-script check
    """
    scores = []
    skipped = 0

    for item in cluster["headlines"]:
        is_english = (
            item["language"] == "en"
            and looks_like_english_script(item["headline"])
        )
        if is_english:
            score = sentiment_analyzer.polarity_scores(item["headline"])["compound"]
            scores.append(score)
        else:
            skipped += 1

    if not scores:
        return {"average": None, "scored": 0, "skipped": skipped}

    return {
        "average": sum(scores) / len(scores),
        "scored": len(scores),
        "skipped": skipped
    }


def sentiment_label(score):
    """Map a VADER compound score to a human-readable mood label."""
    if score is None:
        return "N/A"
    if score >= 0.5:
        return "Very positive"
    if score >= 0.1:
        return "Positive"
    if score > -0.1:
        return "Neutral"
    if score > -0.5:
        return "Negative"
    return "Very negative"


### Step 5: cluster labeling ###

def label_cluster(cluster, max_words=4):
    """
    Pick keywords for a cluster based on word frequency across its members. Words appearing in more headlines
    move to top, which gives a much better label than just sampling from the union set.
    """
    counts = {}
    for item in cluster["headlines"]:
        # Use a set per headline so a word repeated in one headline doesn't
        # outrank a word that appears once in many headlines
        for word in set(item["tokens"]):
            counts[word] = counts.get(word, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[:max_words]]


### Step 6: CSV report ###

cluster_csv = "clusters.csv"
headlines_csv = "headlines.csv"


def generate_csv_reports(clusters, clusters_path, headlines_path):
    """
    Write two CSV files:
      - clusters.csv:  one row per story cluster (summary view)
      - headlines.csv: one row per headline, tagged with its cluster_id
    The cluster_id column links the two files so the user can sort the
    summary file and then look up supporting headlines in the detail file.
    """
    sorted_clusters = sorted(clusters, key=lambda c: len(c["headlines"]), reverse=True)

    # clusters.csv (summary)
    with open(clusters_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster_id",
            "size",
            "keywords",
            "sentiment_label",
            "sentiment_score",
            "headlines_scored",
            "headlines_skipped_non_english"
        ])

        for i, cluster in enumerate(sorted_clusters, start=1):
            sentiment = score_cluster_sentiment(cluster)
            avg = sentiment["average"]
            score_str = f"{avg:.3f}" if avg is not None else ""
            keywords = ", ".join(label_cluster(cluster))

            writer.writerow([
                i,
                len(cluster["headlines"]),
                keywords,
                sentiment_label(avg),
                score_str,
                sentiment["scored"],
                sentiment["skipped"]
            ])

    # headlines.csv (detail)
    with open(headlines_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "cluster_id",
                "source",
                "language",
                "headline",
                "url"
            ])

            for i, cluster in enumerate(sorted_clusters, start=1):
                for h in cluster["headlines"]:
                    writer.writerow([
                        i,
                        h["source"],
                        h["language"],
                        h["headline"],
                        h.get("url", "")
                    ])
    print(f"\nWrote {clusters_path} and {headlines_path}")



chart_path = "news_chart.png"


def generate_chart(clusters, output_path, top_n=15):
    """
    Horizontal bar chart of the top N clusters by size.
    Bar length = number of headlines in the cluster.
    Bar color = sentiment (red = negative, gray = neutral, green = positive, tan = N/A because no English headlines were available).
    """
    sorted_clusters = sorted(clusters, key=lambda c: len(c["headlines"]), reverse=True)
    top = sorted_clusters[:top_n]

    labels = []
    sizes = []
    colors = []

    for cluster in top:
        keywords = label_cluster(cluster, max_words=3)
        label = ", ".join(keywords) if keywords else "(no keywords)"
        labels.append(label)
        sizes.append(len(cluster["headlines"]))

        avg = score_cluster_sentiment(cluster)["average"]
        if avg is None:
            colors.append("#BEBDB8")   # gray: no sentiment available
        elif avg >= 0.1:
            colors.append("#642F6A")   # purple: positive
        elif avg <= -0.1:
            colors.append("#D46F62")   # peach: negative
        else:
            colors.append("#A4446B")   # magenta: neutral

    # Reverse so the biggest cluster appears at the top of the chart
    labels.reverse()
    sizes.reverse()
    colors.reverse()

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(labels))))
    ax.barh(labels, sizes, color=colors)
    ax.set_xlabel("Number of headlines")
    ax.set_title(f"Top {len(top)} stories by coverage (color = sentiment)")

    # Legend explaining the color coding
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#642F6A", label="Positive"),
        Patch(facecolor="#A4446B", label="Neutral"),
        Patch(facecolor="#D46F62", label="Negative"),
        Patch(facecolor="#BEBDB8", label="No score (non-English)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Chart written to {output_path}")


### Extra Check: terminal summary ###

def print_summary(clusters):
    sorted_clusters = sorted(clusters, key=lambda c: len(c["headlines"]), reverse=True)
    print(f"\nFound {len(sorted_clusters)} story clusters")
    print("=" * 60)
    for i, cluster in enumerate(sorted_clusters, start=1):
        size = len(cluster["headlines"])
        keywords = ", ".join(label_cluster(cluster))
        sentiment = score_cluster_sentiment(cluster)
        avg = sentiment["average"]
        mood = sentiment_label(avg)
        score_str = f"{avg:+.2f}" if avg is not None else "N/A"

        print(f"\nCluster {i} ({size} articles) — {mood} ({score_str})")
        print(f"  keywords: {keywords}")
        for h in cluster["headlines"]:
            print(f"  [{h['source']}][{h['language']}] {h['headline']}")
    print("=" * 60)


### Main ###

if __name__ == "__main__":
    all_data = scrape_headlines(news_sites, per_site=num_headlines)
    print(f"\nTotal headlines collected: {len(all_data)}")

    clusters = cluster_headlines(all_data, threshold=cluster_threshold)
    print_summary(clusters)
    generate_csv_reports(clusters, cluster_csv, headlines_csv)
    generate_chart(clusters, chart_path)
