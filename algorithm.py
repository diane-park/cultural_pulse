# scoring algorithm for cultural pulse
'''
####### INPUTS ########
- news data (Event Registry / NewsAPI.ai category counts)
    * headline
    * section/category
    * number of articles in each category
- NYT article data (for headline-level zeitgeist ranking and pillar categorization)
    * headline
    * section
    * rank
    * number of articles in that section
- Wikipedia page view data (for category-level zeitgeist ranking and pillar categorization)

####### PROCESS #######
- Ingest the number of articles and their categories from Event Registry / NewsAPI.ai for the target date

'''
import math
import pandas as pd


# ======================================================
# CONFIG
# ======================================================

NYT_SUMMARY_CSV = "nyt_7_day_pillar_summary.csv"
NEWSAPI_SUMMARY_CSV = "newsapi_daily_category_shares.csv"
WIKI_SUMMARY_CSV = "wiki_7_day_pillar_summary.csv"

COMPOSITE_HISTORY_CSV = "composite_daily_history.csv"
FINAL_OUTPUT_CSV = "integrated_cultural_momentum.csv"

PILLARS = [
    "Politics & Civic Life",
    "Sports & Athletics",
    "Entertainment & Pop Culture",
    "Science & Technology",
    "Business & Finance",
    "Lifestyle & Wellness",
]

NEWSAPI_PILLAR_MAP = {
    "sports": "Sports & Athletics",
    "business_finance": "Business & Finance",
    "technology": "Science & Technology",
    "health_wellness": "Lifestyle & Wellness",
    "entertainment": "Entertainment & Pop Culture",
    "politics": "Politics & Civic Life",
}

SOURCE_WEIGHTS = {
    "newsapi": 0.60,
    "nyt": 0.25,
    "wiki": 0.15,
}


# ======================================================
# HELPERS
# ======================================================

def normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def build_complete_grid(dates: list[str]) -> pd.DataFrame:
    rows = []

    for date in dates:
        for pillar in PILLARS:
            rows.append(
                {
                    "date": date,
                    "pillar": pillar,
                }
            )

    return pd.DataFrame(rows)


def safe_std(series: pd.Series) -> float:
    std = series.std(ddof=1)

    if pd.isna(std) or std == 0:
        return 0.0

    return float(std)


# ======================================================
# LOAD SOURCE CSVs
# ======================================================

def load_nyt_summary(path: str = NYT_SUMMARY_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_date_column(df)

    required = {"date", "pillar", "nyt_share"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"NYT CSV missing columns: {missing}")

    return df[
        [
            "date",
            "pillar",
            "nyt_share",
        ]
    ]


def load_newsapi_summary(path: str = NEWSAPI_SUMMARY_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_date_column(df)

    required = {
        "date",
        "pillar",
        "article_count",
        "share_of_tracked_publishing",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"NewsAPI CSV missing columns: {missing}")

    df["pillar"] = df["pillar"].replace(NEWSAPI_PILLAR_MAP)

    df = df.rename(
        columns={
            "article_count": "newsapi_article_count",
            "share_of_tracked_publishing": "newsapi_share",
        }
    )

    return df[
        [
            "date",
            "pillar",
            "newsapi_article_count",
            "newsapi_share",
        ]
    ]


def load_wiki_summary(path: str = WIKI_SUMMARY_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_date_column(df)

    required = {
        "date",
        "pillar",
        "wiki_share",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Wikipedia CSV missing columns: {missing}")

    keep_cols = [
        "date",
        "pillar",
        "wiki_share",
    ]

    if "wiki_views" in df.columns:
        keep_cols.append("wiki_views")

    return df[keep_cols]


# ======================================================
# COMPOSITE HISTORY
# ======================================================

def build_composite_history(
    nyt_df: pd.DataFrame,
    newsapi_df: pd.DataFrame,
    wiki_df: pd.DataFrame,
) -> pd.DataFrame:
    available_dates = sorted(
        set(nyt_df["date"])
        | set(newsapi_df["date"])
        | set(wiki_df["date"])
    )

    if len(available_dates) < 2:
        raise ValueError(
            "Need at least 2 dates across NYT, NewsAPI, and Wikipedia CSVs."
        )

    base = build_complete_grid(available_dates)

    df = base.merge(
        newsapi_df,
        on=["date", "pillar"],
        how="left",
    )

    df = df.merge(
        nyt_df,
        on=["date", "pillar"],
        how="left",
    )

    df = df.merge(
        wiki_df,
        on=["date", "pillar"],
        how="left",
    )

    df["newsapi_available"] = df["newsapi_share"].notna()
    df["nyt_available"] = df["nyt_share"].notna()
    df["wiki_available"] = df["wiki_share"].notna()

    df["available_source_count"] = (
        df["newsapi_available"].astype(int)
        + df["nyt_available"].astype(int)
        + df["wiki_available"].astype(int)
    )

    df["available_weight"] = (
        df["newsapi_available"].astype(float) * SOURCE_WEIGHTS["newsapi"]
        + df["nyt_available"].astype(float) * SOURCE_WEIGHTS["nyt"]
        + df["wiki_available"].astype(float) * SOURCE_WEIGHTS["wiki"]
    )

    df["newsapi_weight_used"] = df.apply(
        lambda row: (
            SOURCE_WEIGHTS["newsapi"] / row["available_weight"]
            if row["newsapi_available"] and row["available_weight"] > 0
            else 0.0
        ),
        axis=1,
    )

    df["nyt_weight_used"] = df.apply(
        lambda row: (
            SOURCE_WEIGHTS["nyt"] / row["available_weight"]
            if row["nyt_available"] and row["available_weight"] > 0
            else 0.0
        ),
        axis=1,
    )

    df["wiki_weight_used"] = df.apply(
        lambda row: (
            SOURCE_WEIGHTS["wiki"] / row["available_weight"]
            if row["wiki_available"] and row["available_weight"] > 0
            else 0.0
        ),
        axis=1,
    )

    df["newsapi_share_filled"] = df["newsapi_share"].fillna(0.0)
    df["nyt_share_filled"] = df["nyt_share"].fillna(0.0)
    df["wiki_share_filled"] = df["wiki_share"].fillna(0.0)

    df["composite_share"] = (
        df["newsapi_weight_used"] * df["newsapi_share_filled"]
        + df["nyt_weight_used"] * df["nyt_share_filled"]
        + df["wiki_weight_used"] * df["wiki_share_filled"]
    )

    df["popularity_score"] = df["composite_share"] * 100

    if "newsapi_article_count" in df.columns:
        df["newsapi_article_count"] = (
            df["newsapi_article_count"]
            .fillna(0)
            .astype(int)
        )

    if "wiki_views" in df.columns:
        df["wiki_views"] = (
            df["wiki_views"]
            .fillna(0)
            .astype(int)
        )

    df = df.sort_values(
        ["date", "pillar"]
    ).reset_index(drop=True)

    df.to_csv(COMPOSITE_HISTORY_CSV, index=False)

    return df


# ======================================================
# MOMENTUM
# ======================================================

def calculate_momentum(
    composite_df: pd.DataFrame,
    target_date: str | None = None,
) -> pd.DataFrame:
    if target_date is None:
        target_date = max(composite_df["date"])

    target_date = pd.to_datetime(target_date).strftime("%Y-%m-%d")

    current_df = composite_df[
        composite_df["date"] == target_date
    ].copy()

    historical_df = composite_df[
        composite_df["date"] < target_date
    ].copy()

    if current_df.empty:
        raise ValueError(f"No composite data found for target date: {target_date}")

    if historical_df.empty:
        raise ValueError(
            "No historical dates available before target_date."
        )

    rows = []

    for _, current_row in current_df.iterrows():
        pillar = current_row["pillar"]
        current_share = current_row["composite_share"]

        history = historical_df[
            (historical_df["pillar"] == pillar)
            & (historical_df["available_source_count"] > 0)
        ]["composite_share"]

        baseline_days = history.count()

        if baseline_days == 0:
            historical_mean = 0.0
            historical_std = 0.0
            z_score = 0.0
            percentile = 50.0
        else:
            historical_mean = history.mean()
            historical_std = safe_std(history)

            if historical_std == 0:
                z_score = 0.0
            else:
                z_score = (
                    current_share - historical_mean
                ) / historical_std

            percentile = normal_cdf(z_score) * 100

        popularity_score = current_row["popularity_score"]

        momentum_score = (
            0.80 * percentile
            + 0.20 * popularity_score
        )

        rows.append(
            {
                "date": target_date,
                "pillar": pillar,

                "newsapi_available": current_row["newsapi_available"],
                "nyt_available": current_row["nyt_available"],
                "wiki_available": current_row["wiki_available"],
                "available_source_count": int(current_row["available_source_count"]),

                "newsapi_weight_used": round(current_row["newsapi_weight_used"], 4),
                "nyt_weight_used": round(current_row["nyt_weight_used"], 4),
                "wiki_weight_used": round(current_row["wiki_weight_used"], 4),

                "newsapi_share": (
                    round(current_row["newsapi_share"], 4)
                    if pd.notna(current_row["newsapi_share"])
                    else None
                ),
                "nyt_share": (
                    round(current_row["nyt_share"], 4)
                    if pd.notna(current_row["nyt_share"])
                    else None
                ),
                "wiki_share": (
                    round(current_row["wiki_share"], 4)
                    if pd.notna(current_row["wiki_share"])
                    else None
                ),
                "composite_share": round(current_share, 4),

                "baseline_days": int(baseline_days),
                "historical_mean": round(historical_mean, 4),
                "historical_std": round(historical_std, 4),

                "z_score": round(z_score, 2),
                "percentile": round(percentile, 2),

                "popularity_score": round(popularity_score, 2),
                "momentum_score": round(momentum_score, 2),
            }
        )

    result_df = pd.DataFrame(rows)

    result_df = result_df.sort_values(
        "momentum_score",
        ascending=False,
    ).reset_index(drop=True)

    result_df.to_csv(FINAL_OUTPUT_CSV, index=False)

    return result_df


# ======================================================
# RUN PIPELINE
# ======================================================

def run_integrated_pipeline(
    target_date: str | None = None,
) -> pd.DataFrame:
    nyt_df = load_nyt_summary()
    newsapi_df = load_newsapi_summary()
    wiki_df = load_wiki_summary()

    composite_df = build_composite_history(
        nyt_df=nyt_df,
        newsapi_df=newsapi_df,
        wiki_df=wiki_df,
    )

    result_df = calculate_momentum(
        composite_df=composite_df,
        target_date=target_date,
    )

    print("\nIntegrated Cultural Momentum")
    print("=" * 80)

    print(
        result_df[
            [
                "pillar",
                "popularity_score",
                "momentum_score",
                "percentile",
                "z_score",
                "baseline_days",
                "newsapi_available",
                "nyt_available",
                "wiki_available",
                "newsapi_weight_used",
                "nyt_weight_used",
                "wiki_weight_used",
            ]
        ]
    )

    print("\nSaved files:")
    print(f"- {COMPOSITE_HISTORY_CSV}")
    print(f"- {FINAL_OUTPUT_CSV}")

    return result_df


if __name__ == "__main__":
    run_integrated_pipeline('2025-01-09')