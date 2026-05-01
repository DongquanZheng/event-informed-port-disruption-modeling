# NLP Signal Construction

This document describes how GDELT news-event records are converted into weekly NLP-derived event signals.

The goal is not to claim that the NLP component perfectly understands each news article. The goal is to create a transparent, reproducible, and testable event-signal layer that can be compared against an operational port-activity baseline.

## 1. Raw GDELT Fields Used

The pipeline uses the GDELT Event Database daily files. The most important fields are:

- `SQLDATE`: event date in GDELT.
- `QuadClass`: broad event class, where higher values generally represent more conflictual material events.
- `GoldsteinScale`: event impact score, where negative values indicate conflictual or adverse events.
- `AvgTone`: average tone of the article or event context.
- `NumMentions`: number of event mentions.
- `NumSources`: number of sources.
- `NumArticles`: number of articles.
- `ActionGeo_CountryCode`: country code associated with the event action location.
- `ActionGeo_FullName`: full location name associated with the event action location.
- `SOURCEURL`: source article URL.

The project does not redistribute full news article text. Instead, it uses the URL path in `SOURCEURL` as a lightweight text proxy.

## 2. Maritime Event Filtering

The first filtering step keeps GDELT rows whose `SOURCEURL` contains broad maritime or logistics-related keywords.

Broad URL filter:

```text
shipping | maritime | cargo | container | vessel | tanker | terminal | freight |
suez | panama | red sea | houthi | houthis | maersk | port
```

After URL slug extraction, a second cleaner keyword filter is applied to the extracted slug text:

```text
port | ports | shipping | maritime | cargo | container | containers |
vessel | vessels | ship | ships | tanker | tankers | terminal | terminals |
freight | suez | panama | canal | strait | red sea | houthi | houthis | maersk
```

This two-step design is used because the raw GDELT daily files are large. The broad filter reduces the search space, while the cleaned slug filter improves topical relevance.

## 3. URL Slug Text Extraction

For each article URL, the pipeline extracts the URL path and converts it into a simple text representation.

Example:

```text
https://example.com/2024/01/maersk-halts-red-sea-transit-after-attack
```

becomes:

```text
maersk halts red sea transit after attack
```

Cleaning rules:

- keep the URL path only;
- replace hyphens and underscores with spaces;
- remove file extensions;
- remove non-alphabetical characters;
- lowercase all text;
- collapse repeated whitespace.

This produces the `url_slug_text` variable used by the NLP model.

## 4. Weak Label Construction

The NLP classifier is trained using weak labels rather than manually annotated labels.

An article is labeled as disruption-related if it satisfies both conditions:

### Condition A: severe or adverse event signal

At least one of the following is true:

```text
max_quadclass >= 3
min_goldstein <= -5
avg_tone <= -5
```

### Condition B: disruption-related text signal

The URL slug contains at least one disruption-related term:

```text
attack | attacks | strike | strikes | missile | war | conflict |
houthi | houthis | delay | delays | disruption | disrupted |
halt | halts | blocked | closure | closed | congestion |
diversion | reroute | protest | explosion | fire | sanction
```

Final weak target:

\(z_i=1\) if \(s_i=1\) and \(d_i=1\); otherwise, \(z_i=0\).

Here, \(z_i\) is the weak disruption label for article \(i\), \(s_i\) indicates whether the article satisfies the severe-event condition, and \(d_i\) indicates whether the article contains disruption-related terms.

This weak-labeling design intentionally combines structured GDELT event attributes with transparent text rules. It is not a ground-truth article annotation, but it provides a reproducible training signal for the NLP component.

## 5. NLP Model

The article-level NLP model is:

```text
TF-IDF vectorizer + Logistic Regression
```

TF-IDF settings:

```text
lowercase = True
min_df = 3
max_df = 0.8
ngram_range = (1, 2)
stop_words = "english"
```

Classifier settings:

```text
LogisticRegression(max_iter=1000, class_weight="balanced")
```

The model is trained on weakly labeled January 2024 maritime news. It is then applied to maritime article records from 2021 to 2025.

Output variable:

\(p_i^{NLP} = P(z_i = 1 \mid \text{URL slug text}_i)\)

High-risk article flag:

\(h_i=1\) if \(p_i^{NLP} \geq 0.5\); otherwise, \(h_i=0\).

## 6. Weekly Event Aggregation

Article-level NLP outputs are aggregated to weekly features using Monday-start weeks.

For each week, the following features are calculated:

- `article_count`: number of filtered maritime articles.
- `unique_url_count`: number of unique article URLs.
- `avg_nlp_risk_score`: average article-level disruption probability.
- `max_nlp_risk_score`: maximum article-level disruption probability.
- `high_risk_article_count`: number of high-risk articles.
- `high_risk_article_share`: share of high-risk articles.
- `total_mentions`: total GDELT mentions.
- `avg_tone`: average GDELT tone.
- `min_tone`: most negative tone observed in the week.

Weekly aggregation aligns the NLP signal with the weekly PortWatch prediction task and reduces same-day leakage concerns.

## 7. Spatial Signal Layers

The project constructs three event-signal layers.

### Global Layer

All filtered maritime news records are included.

Prefix:

```text
global_
```

### Europe Layer

Records are included if `main_country` belongs to the selected European country-code set used in the script.

Prefix:

```text
europe_
```

This layer tests whether regional exposure is more informative than undifferentiated global maritime news.

### Local Layer

Records are included if the action location is related to the Netherlands, Rotterdam, or nearby relevant port/geographic terms.

Local matching terms include:

```text
Netherlands
Rotterdam
Zuid-Holland
Amsterdam
Zeehaven
Eemshaven
North Sea
```

Prefix:

```text
local_
```

This layer is intentionally strict. It is expected to be sparse, but it helps test whether location-aligned event signals contain additional information.

## 8. Features Used in the Final Two-Stage Model

The first-stage model uses operational PortWatch features and outputs:

\(\hat{p}_{t+1}^{base}\)

The second-stage model uses:

```text
baseline_probability
global_high_risk_article_share
global_avg_tone
global_min_tone
europe_high_risk_article_share
europe_avg_tone
europe_min_tone
local_high_risk_article_share
local_avg_tone
local_min_tone
```

This design tests whether multiscale event information can correct the risk estimate produced by historical operational data.

## 9. Source of Truth

The executable implementation is:

```text
scripts/reproduce_pipeline.py
```

This document explains the logic. The script contains the exact implementation.
