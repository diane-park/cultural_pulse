import os
import time
import datetime
import requests
import pandas as pd
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


# ======================================================
# CONFIG
# ======================================================

PILLARS = [
    "Politics & Civic Life",
    "Sports & Athletics",
    "Entertainment & Pop Culture",
    "Science & Technology",
    "Business & Finance",
    "Lifestyle & Wellness",
]

EXCLUDED_PAGES = [
    "Main_Page",
    "Special:Search",
    "Wikipedia:",
    "Portal:",
    "Main Page",
    "Special:",
    "Search",
    "File:",
    "Help:",
    "Category:",
    "Talk:",
]

USER_AGENT = os.getenv(
    "WIKIMEDIA_USER_AGENT",
    "ZeitgeistRadarBot/2.0 (your_email@example.com)",
)

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None


class WikiClassification(BaseModel):
    global_id: str
    pillar: str


# ======================================================
# FETCH WIKIPEDIA DATA
# ======================================================

def fetch_wikipedia_historical(
    target_date: str,
    limit: int = 25,
) -> pd.DataFrame:
    """
    Fetch top viewed Wikipedia articles for a single date.

    target_date format:
    YYYY-MM-DD
    """

    date_obj = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m")
    day = date_obj.strftime("%d")

    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
        f"en.wikipedia/all-access/{year}/{month}/{day}"
    )

    headers = {
        "User-Agent": USER_AGENT,
    }

    print(f"Fetching Wikimedia pageviews for {target_date}...")

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    if response.status_code == 404:
        raise Exception(
            "Wikipedia data not found. Pageview data is usually finalized 1-2 days late."
        )

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch Wikipedia data: {response.status_code}\n{response.text}"
        )

    data = response.json()
    articles = data["items"][0]["articles"]

    df = pd.DataFrame(articles)

    for pattern in EXCLUDED_PAGES:
        df = df[
            ~df["article"].str.contains(
                pattern,
                case=False,
                na=False,
                regex=False,
            )
        ]

    df["date"] = target_date
    df["clean_title"] = df["article"].str.replace("_", " ", regex=False)

    df = df.head(limit).copy()

    df["zeitgeist_rank"] = range(1, len(df) + 1)
    df["global_id"] = df.apply(
        lambda row: f"{row['date']}_{int(row['zeitgeist_rank'])}",
        axis=1,
    )

    return df[
        [
            "date",
            "global_id",
            "zeitgeist_rank",
            "article",
            "clean_title",
            "views",
        ]
    ]


def fetch_wikipedia_period(
    end_date: str,
    days: int = 7,
    limit_per_day: int = 25,
) -> pd.DataFrame:
    """
    Fetch Wikipedia top articles for end_date plus previous days - 1.
    """

    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    rows = []

    for i in range(days):
        target_date = (
            end_dt - datetime.timedelta(days=i)
        ).strftime("%Y-%m-%d")

        try:
            day_df = fetch_wikipedia_historical(
                target_date=target_date,
                limit=limit_per_day,
            )

            rows.append(day_df)

            time.sleep(1)

        except Exception as exc:
            print(f"Failed Wikipedia fetch for {target_date}: {exc}")

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# ======================================================
# GEMINI BATCH CLASSIFICATION
# ======================================================

def categorize_wikipedia_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sends all Wikipedia titles across the full period to Gemini once.
    """

    if df.empty:
        df["pillar"] = None
        return df

    if not client:
        raise ValueError(
            "Gemini client is uninitialized. Verify GEMINI_API_KEY in your .env file."
        )

    articles_data = [
        {
            "global_id": row["global_id"],
            "date": row["date"],
            "rank": int(row["zeitgeist_rank"]),
            "title": row["clean_title"],
            "views": int(row["views"]),
        }
        for _, row in df.iterrows()
    ]

    prompt = f"""
You are analyzing American cultural attention using top-viewed English Wikipedia pages.

Categorize each Wikipedia page title into exactly one cultural pillar.

Allowed pillars:
{PILLARS}

Rules:
- Return exactly one classification per page.
- Preserve global_id exactly.
- Use only the allowed pillar names.
- If the title is a person, company, TV show, movie, athlete, politician, or event, classify based on what it is most culturally associated with.
- If unsure, choose the closest pillar.

Wikipedia pages:
{articles_data}
"""

    print(f"Sending {len(articles_data)} Wikipedia pages to Gemini in ONE batch...")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": list[WikiClassification],
            "temperature": 0.1,
        },
    )

    classifications = response.parsed

    pillar_map = {
        item.global_id: item.pillar
        for item in classifications
        if item.pillar in PILLARS
    }

    df = df.copy()
    df["pillar"] = df["global_id"].map(pillar_map)
    df["pillar"] = df["pillar"].fillna("Entertainment & Pop Culture")

    return df


# ======================================================
# SUMMARY CSV
# ======================================================

def create_daily_wiki_pillar_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates weighted daily pillar shares using pageviews.

    Output is intentionally similar to the NYT summary:
    date, pillar, article_count, daily_total, wiki_views, wiki_share
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "pillar",
                "article_count",
                "daily_total",
                "wiki_views",
                "daily_views",
                "wiki_share",
            ]
        )

    summary = (
        df.groupby(["date", "pillar"])
        .agg(
            article_count=("clean_title", "count"),
            wiki_views=("views", "sum"),
        )
        .reset_index()
    )

    daily_totals = (
        summary.groupby("date")
        .agg(
            daily_total=("article_count", "sum"),
            daily_views=("wiki_views", "sum"),
        )
        .reset_index()
    )

    summary = summary.merge(
        daily_totals,
        on="date",
        how="left",
    )

    summary["wiki_share"] = (
        summary["wiki_views"] / summary["daily_views"]
    )

    return summary.sort_values(
        ["date", "wiki_share"],
        ascending=[True, False],
    ).reset_index(drop=True)


# ======================================================
# PUBLIC PIPELINE
# ======================================================

def run_wikipedia_period_pipeline(
    end_date: str,
    days: int = 7,
    limit_per_day: int = 25,
    article_output_path: str = "wiki_7_day_articles_classified.csv",
    summary_output_path: str = "wiki_7_day_pillar_summary.csv",
):
    raw_df = fetch_wikipedia_period(
        end_date=end_date,
        days=days,
        limit_per_day=limit_per_day,
    )

    categorized_df = categorize_wikipedia_batch(raw_df)

    summary_df = create_daily_wiki_pillar_summary(categorized_df)

    categorized_df.to_csv(article_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print("\nSaved files:")
    print(f"- {article_output_path}")
    print(f"- {summary_output_path}")

    print("\nWikipedia daily pillar summary:")
    print(summary_df)

    return categorized_df, summary_df


def get_categorized_wiki_data(
    target_date: str,
    limit: int = 25,
) -> pd.DataFrame:
    """
    Backward-compatible single-day function.
    """

    raw_df = fetch_wikipedia_historical(
        target_date=target_date,
        limit=limit,
    )

    return categorize_wikipedia_batch(raw_df)


# ======================================================
# RUN DIRECTLY
# ======================================================

if __name__ == "__main__":
    # Wikipedia pageview data usually lags by 1-2 days.
    # Use an older date if today's data is unavailable.
    END_DATE = "2026-06-14"

    run_wikipedia_period_pipeline(
        end_date=END_DATE,
        days=30,
        limit_per_day=25,
    )
