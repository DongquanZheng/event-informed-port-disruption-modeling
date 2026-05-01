from __future__ import annotations

import argparse
import re
import time
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PORTWATCH_DAILY_TRADE_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Trade_Data/FeatureServer/0/query"
)

EVENT_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
    "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

OPERATIONAL_FEATURES = [
    "lag_activity_1w",
    "lag_activity_2w",
    "lag_activity_4w",
    "rolling_mean_4w",
    "rolling_mean_8w_feature",
    "rolling_std_4w",
    "rolling_std_8w_feature",
    "rolling_change_4w",
    "month",
    "quarter",
]

MULTISCALE_EVENT_FEATURES = [
    "global_high_risk_article_share",
    "global_avg_tone",
    "global_min_tone",
    "europe_high_risk_article_share",
    "europe_avg_tone",
    "europe_min_tone",
    "local_high_risk_article_share",
    "local_avg_tone",
    "local_min_tone",
]

ROUGH_MARITIME_PATTERN = re.compile(
    r"shipping|maritime|cargo|container|vessel|tanker|terminal|freight|"
    r"suez|panama|red[-_ ]sea|houthi|houthis|maersk|port",
    re.IGNORECASE,
)

CLEAN_MARITIME_PATTERN = re.compile(
    r"\b(?:port|ports|shipping|maritime|cargo|container|containers|vessel|vessels|"
    r"ship|ships|tanker|tankers|terminal|terminals|freight|suez|panama|"
    r"canal|strait|red sea|houthi|houthis|maersk)\b",
    re.IGNORECASE,
)

DISRUPTION_TERMS = re.compile(
    r"\b(?:attack|attacks|strike|strikes|missile|war|conflict|houthi|houthis|"
    r"delay|delays|disruption|disrupted|halt|halts|blocked|closure|closed|"
    r"congestion|diversion|reroute|protest|strike|explosion|fire|sanction)\b",
    re.IGNORECASE,
)

EUROPE_COUNTRIES = {
    "AL", "AU", "BE", "BG", "BK", "BO", "CY", "EZ", "DA", "EN", "FI", "FR",
    "GM", "GR", "HU", "IC", "EI", "IT", "LG", "LH", "LU", "MT", "MD", "MN",
    "NL", "NO", "PL", "PO", "RO", "RS", "SP", "SW", "SZ", "UK",
}


def ensure_directories() -> None:
    for directory in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def parse_portwatch_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def download_portwatch_rotterdam() -> pd.DataFrame:
    rows = []
    offset = 0

    while True:
        params = {
            "where": "portname = 'Rotterdam'",
            "outFields": "*",
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": 1000,
            "orderByFields": "date ASC",
        }
        response = requests.get(PORTWATCH_DAILY_TRADE_URL, params=params, timeout=60)
        response.raise_for_status()
        features = response.json().get("features", [])

        if not features:
            break

        rows.extend(feature["attributes"] for feature in features)
        offset += len(features)
        print(f"Downloaded PortWatch rows: {len(rows)}")

        if len(features) < 1000:
            break

    portwatch = pd.DataFrame(rows)
    portwatch["date"] = parse_portwatch_date(portwatch["date"])
    portwatch = portwatch.sort_values("date").reset_index(drop=True)
    output_path = RAW_DIR / "portwatch_rotterdam_full.csv"
    portwatch.to_csv(output_path, index=False)
    print(f"Saved PortWatch data: {output_path}")
    return portwatch


def build_weekly_port_features(portwatch: pd.DataFrame) -> pd.DataFrame:
    daily = portwatch.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week"] = daily["date"].dt.to_period("W-SUN").apply(lambda x: x.start_time)

    weekly = (
        daily.groupby("week", as_index=False)
        .agg(
            activity=("portcalls", "sum"),
            portcalls=("portcalls", "sum"),
            import_value=("import", "sum"),
            export_value=("export", "sum"),
            portcalls_container=("portcalls_container", "sum"),
            portcalls_tanker=("portcalls_tanker", "sum"),
            days_observed=("date", "nunique"),
        )
        .sort_values("week")
        .reset_index(drop=True)
    )

    weekly["next_week_activity"] = weekly["activity"].shift(-1)
    weekly["rolling_mean_8w"] = weekly["activity"].shift(1).rolling(8).mean()
    weekly["rolling_std_8w"] = weekly["activity"].shift(1).rolling(8).std()
    weekly["abnormal_activity_next_week"] = (
        weekly["next_week_activity"] < weekly["rolling_mean_8w"] - 1.5 * weekly["rolling_std_8w"]
    ).astype(int)

    weekly["lag_activity_1w"] = weekly["activity"].shift(1)
    weekly["lag_activity_2w"] = weekly["activity"].shift(2)
    weekly["lag_activity_4w"] = weekly["activity"].shift(4)
    weekly["rolling_mean_4w"] = weekly["activity"].shift(1).rolling(4).mean()
    weekly["rolling_mean_8w_feature"] = weekly["activity"].shift(1).rolling(8).mean()
    weekly["rolling_std_4w"] = weekly["activity"].shift(1).rolling(4).std()
    weekly["rolling_std_8w_feature"] = weekly["activity"].shift(1).rolling(8).std()
    weekly["rolling_change_4w"] = weekly["lag_activity_1w"] - weekly["lag_activity_4w"]
    weekly["month"] = weekly["week"].dt.month
    weekly["quarter"] = weekly["week"].dt.quarter
    weekly["year"] = weekly["week"].dt.year

    output_path = PROCESSED_DIR / "portwatch_weekly_operational_features.csv"
    weekly.to_csv(output_path, index=False)
    print(f"Saved weekly PortWatch features: {output_path}")
    return weekly


def extract_slug_text(url: str) -> str:
    path = urlparse(str(url)).path
    text = unquote(path)
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\.[A-Za-z0-9]+$", " ", text)
    text = re.sub(r"[^A-Za-z ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def download_gdelt_day(day: pd.Timestamp, retries: int = 2) -> pd.DataFrame | None:
    date_str = day.strftime("%Y%m%d")
    url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"

    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=60)

            if response.status_code == 404:
                print(f"Missing GDELT file, skipped: {date_str}")
                return None

            response.raise_for_status()
            with zipfile.ZipFile(BytesIO(response.content)) as archive:
                csv_name = archive.namelist()[0]
                with archive.open(csv_name) as file:
                    df = pd.read_csv(
                        file,
                        sep="\t",
                        header=None,
                        names=EVENT_COLUMNS,
                        dtype={"SOURCEURL": "string"},
                        low_memory=False,
                    )

            df["media_date"] = pd.to_datetime(date_str)
            return df

        except requests.RequestException:
            if attempt < retries:
                time.sleep(3)
            else:
                print(f"Failed GDELT download after retries, skipped: {date_str}")
                return None


def process_gdelt_range(start_date: str, end_date: str) -> pd.DataFrame:
    filtered_days = []
    days = pd.date_range(start_date, end_date, freq="D")

    for i, day in enumerate(days, start=1):
        df_day = download_gdelt_day(day)
        if df_day is None:
            continue

        df_day = df_day[df_day["SOURCEURL"].fillna("").str.contains(ROUGH_MARITIME_PATTERN)]
        if len(df_day) > 0:
            keep_columns = [
                "media_date", "SQLDATE", "QuadClass", "GoldsteinScale",
                "AvgTone", "NumMentions", "NumSources", "NumArticles",
                "ActionGeo_CountryCode", "ActionGeo_FullName", "SOURCEURL",
            ]
            filtered_days.append(df_day[keep_columns])

        if i % 20 == 0:
            print(f"Processed {i} GDELT days from {start_date} to {end_date}")

    events = pd.concat(filtered_days, ignore_index=True)
    events["url_slug_text"] = events["SOURCEURL"].apply(extract_slug_text)
    events = events[events["url_slug_text"].str.contains(CLEAN_MARITIME_PATTERN)].copy()
    return events


def events_to_articles(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(["media_date", "SOURCEURL", "url_slug_text"], as_index=False)
        .agg(
            event_rows=("SOURCEURL", "size"),
            max_quadclass=("QuadClass", "max"),
            min_goldstein=("GoldsteinScale", "min"),
            avg_goldstein=("GoldsteinScale", "mean"),
            avg_tone=("AvgTone", "mean"),
            min_tone=("AvgTone", "min"),
            total_mentions=("NumMentions", "sum"),
            total_sources=("NumSources", "sum"),
            total_articles=("NumArticles", "sum"),
            main_country=("ActionGeo_CountryCode", lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan),
            main_location=("ActionGeo_FullName", lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan),
        )
    )


def add_weak_labels(articles: pd.DataFrame) -> pd.DataFrame:
    df = articles.copy()
    severe_event = (df["max_quadclass"] >= 3) | (df["min_goldstein"] <= -5) | (df["avg_tone"] <= -5)
    disruption_text = df["url_slug_text"].fillna("").str.contains(DISRUPTION_TERMS)
    df["weak_target_label"] = (severe_event & disruption_text).astype(int)
    return df


def train_weak_nlp_model(training_articles: pd.DataFrame) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=3,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words="english",
    )
    x_train = vectorizer.fit_transform(training_articles["url_slug_text"].fillna(""))
    y_train = training_articles["weak_target_label"]
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, y_train)
    return vectorizer, model


def score_articles(articles: pd.DataFrame, vectorizer: TfidfVectorizer, model: LogisticRegression) -> pd.DataFrame:
    df = articles.copy()
    x_text = vectorizer.transform(df["url_slug_text"].fillna(""))
    df["nlp_disruption_probability"] = model.predict_proba(x_text)[:, 1]
    df["high_risk_article"] = (df["nlp_disruption_probability"] >= 0.5).astype(int)
    return df


def is_local_signal(row: pd.Series) -> bool:
    location = str(row.get("main_location", "")).lower()
    country = str(row.get("main_country", ""))
    return country == "NL" or any(
        token in location
        for token in ["netherlands", "rotterdam", "zuid-holland", "amsterdam", "zeehaven", "eemshaven", "north sea"]
    )


def aggregate_weekly_features(articles: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if articles.empty:
        return pd.DataFrame(columns=["week"])

    df = articles.copy()
    df["week"] = df["media_date"].dt.to_period("W-SUN").apply(lambda x: x.start_time)
    weekly = (
        df.groupby("week", as_index=False)
        .agg(
            article_count=("SOURCEURL", "count"),
            unique_url_count=("SOURCEURL", "nunique"),
            avg_nlp_risk_score=("nlp_disruption_probability", "mean"),
            max_nlp_risk_score=("nlp_disruption_probability", "max"),
            high_risk_article_count=("high_risk_article", "sum"),
            high_risk_article_share=("high_risk_article", "mean"),
            total_mentions=("total_mentions", "sum"),
            avg_tone=("avg_tone", "mean"),
            min_tone=("min_tone", "min"),
        )
    )
    rename = {col: f"{prefix}_{col}" for col in weekly.columns if col != "week"}
    return weekly.rename(columns=rename)


def build_multiscale_weekly_features(articles: pd.DataFrame) -> pd.DataFrame:
    global_weekly = aggregate_weekly_features(articles, "global")
    europe_weekly = aggregate_weekly_features(articles[articles["main_country"].isin(EUROPE_COUNTRIES)], "europe")
    local_weekly = aggregate_weekly_features(articles[articles.apply(is_local_signal, axis=1)], "local")

    weekly = global_weekly.merge(europe_weekly, on="week", how="outer").merge(local_weekly, on="week", how="outer")
    count_like = [col for col in weekly.columns if col != "week"]
    weekly[count_like] = weekly[count_like].fillna(0)
    return weekly.sort_values("week").reset_index(drop=True)


def evaluate_model(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "predicted_positive": int(predictions.sum()),
    }


def run_modeling(weekly_port: pd.DataFrame, weekly_events: pd.DataFrame) -> pd.DataFrame:
    dataset = weekly_port.merge(weekly_events, on="week", how="left")
    event_columns = [col for col in weekly_events.columns if col != "week"]
    dataset[event_columns] = dataset[event_columns].fillna(0)
    dataset = dataset.dropna(subset=OPERATIONAL_FEATURES + ["abnormal_activity_next_week"]).copy()

    train = dataset[dataset["year"].between(2021, 2024)].copy()
    test = dataset[dataset["year"] == 2025].copy()

    baseline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    baseline.fit(train[OPERATIONAL_FEATURES], train["abnormal_activity_next_week"])

    train["baseline_probability"] = baseline.predict_proba(train[OPERATIONAL_FEATURES])[:, 1]
    test["baseline_probability"] = baseline.predict_proba(test[OPERATIONAL_FEATURES])[:, 1]

    second_stage_features = ["baseline_probability"] + MULTISCALE_EVENT_FEATURES
    correction = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    correction.fit(train[second_stage_features], train["abnormal_activity_next_week"])
    corrected_probability = correction.predict_proba(test[second_stage_features])[:, 1]

    results = pd.DataFrame(
        [
            {
                "model": "operational_baseline",
                **evaluate_model(test["abnormal_activity_next_week"], test["baseline_probability"], threshold=0.5),
            },
            {
                "model": "two_stage_multiscale_nlp",
                **evaluate_model(test["abnormal_activity_next_week"], corrected_probability, threshold=0.5),
            },
        ]
    )

    dataset_path = PROCESSED_DIR / "reproduced_multiscale_model_dataset.csv"
    results_path = PROCESSED_DIR / "reproduced_multiscale_model_results.csv"
    dataset.to_csv(dataset_path, index=False)
    results.to_csv(results_path, index=False)
    print(f"Saved reproduced model dataset: {dataset_path}")
    print(f"Saved reproduced model results: {results_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the PortWatch-GDELT multiscale NLP modeling pipeline.")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--skip-portwatch-download", action="store_true")
    args = parser.parse_args()

    ensure_directories()

    if args.skip_portwatch_download and (RAW_DIR / "portwatch_rotterdam_full.csv").exists():
        portwatch = pd.read_csv(RAW_DIR / "portwatch_rotterdam_full.csv", parse_dates=["date"])
    else:
        portwatch = download_portwatch_rotterdam()

    weekly_port = build_weekly_port_features(portwatch)

    jan_events = process_gdelt_range("2024-01-01", "2024-01-31")
    jan_articles = add_weak_labels(events_to_articles(jan_events))
    vectorizer, nlp_model = train_weak_nlp_model(jan_articles)

    scored_articles = []
    for year in args.years:
        print(f"Processing GDELT year: {year}")
        events = process_gdelt_range(f"{year}-01-01", f"{year}-12-31")
        articles = score_articles(events_to_articles(events), vectorizer, nlp_model)
        articles["year"] = year
        articles_path = INTERIM_DIR / f"gdelt_maritime_articles_{year}_with_nlp.csv"
        articles.to_csv(articles_path, index=False)
        scored_articles.append(articles)
        print(f"Saved scored articles: {articles_path}")

    all_articles = pd.concat(scored_articles, ignore_index=True)
    weekly_events = build_multiscale_weekly_features(all_articles)
    weekly_events_path = INTERIM_DIR / "weekly_event_features_2021_2025_multiscale.csv"
    weekly_events.to_csv(weekly_events_path, index=False)
    print(f"Saved weekly event features: {weekly_events_path}")

    results = run_modeling(weekly_port, weekly_events)
    print(results)


if __name__ == "__main__":
    main()
