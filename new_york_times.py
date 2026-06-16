'''import os
import time
import requests
import pandas as pd
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environmental variables from the local .env file
load_dotenv()

# --- CONFIGURATION & MODEL INITIALIZATION ---
PILLARS = [
    "Politics & Civic Life",
    "Sports & Athletics",
    "Entertainment & Pop Culture",
    "Science & Technology",
    "Business & Finance",
    "Lifestyle & Wellness"
]

# Extract keys safely
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
NYT_KEY = os.getenv('NYT_API_KEY')

# Initialize the Gemini client if the key is present
if GEMINI_KEY:
    client = genai.Client(api_key=GEMINI_KEY)
else:
    client = None

# Enforce Pydantic schema structure for Gemini's response
class ArticleClassification(BaseModel):
    id: int
    pillar: str


# --- CORE PIPELINE FUNCTIONS ---

def fetch_nyt_historical(target_date: str, limit: int = 25) -> pd.DataFrame:
    """
    Fetches top articles from the New York Times for a specific past date.
    Automatically handles pagination to clear the 10-article payload restriction.
    
    Parameters:
        target_date (str): Format 'YYYY-MM-DD'
        limit (int): Total number of articles to return (default 25)
    """
    if not NYT_KEY:
        raise ValueError("Missing NYT_API_KEY environment variable.")

    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    formatted_date = target_date.replace("-", "")
    
    news_list = []
    page = 0
    rank_counter = 1 
    
    print(f"Paginating NYT API to retrieve {limit} articles for {target_date}...")
    
    while len(news_list) < limit:
        params = {
            "begin_date": formatted_date,
            "end_date": formatted_date,
            "sort": "relevance",  # Should we sort by newest instead so we know what the search criteria is?
            "page": page,         
            "api-key": NYT_KEY
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 429:
            print("  [Rate Limit Hit] Pausing for 10 seconds before continuing...")
            time.sleep(10)
            continue
        elif response.status_code != 200:
            raise Exception(f"NYT API Error {response.status_code}: {response.text}")
            
        data = response.json()
        docs = data.get('response', {}).get('docs', [])
        
        if not docs:
            break # Stop if there are no more pages available
            
        for item in docs:
            headline = item.get('headline', {})
            title = headline.get('print_headline') or headline.get('main', 'Unknown Title')
            section = item.get('section_name', item.get('news_desk', 'Unknown'))
            
            news_list.append({
                'zeitgeist_rank': rank_counter, 
                'clean_title': title,
                'source': 'The New York Times',
                'description': item.get('abstract', ''),
                'nyt_section': section
            })
            
            rank_counter += 1
            if len(news_list) == limit:
                break
                
        page += 1
        time.sleep(6) # Protect against API rate limits
        
    return pd.DataFrame(news_list)


def categorize_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sends the gathered headlines and metadata to Gemini to categorize them 
    into the 6 predefined culture pillars.
    """
    if not client:
        raise ValueError("Gemini Client is uninitialized. Verify GEMINI_API_KEY in your .env file.")
        
    if df.empty:
        print("Received an empty DataFrame. Skipping categorization.")
        df['pillar'] = None
        return df

    print(f"Categorizing {len(df)} news headlines using Gemini + NYT Section Tags...")
    
    if 'nyt_section' not in df.columns:
        df['nyt_section'] = 'General'
        
    # Build structural mapping dataset for the prompt payload
    articles_data = [
        {"id": i, "title": row['clean_title'], "section": row.get('nyt_section', 'General')} 
        for i, row in df.iterrows()
    ]
    
    prompt = f"""
    You are analyzing the American zeitgeist. Categorize each of the following New York Times headlines into the most appropriate pillar.
    Use the provided NYT Section tag to help guide your categorization.
    
    The strict pillars are:
    {PILLARS}
    
    Articles to categorize:
    {articles_data}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[ArticleClassification],
            'temperature': 0.1, 
        },
    )
    
    # Map classifications cleanly to the existing DataFrame index
    classifications = response.parsed
    pillar_map = {item.id: item.pillar for item in classifications}
    
    df['pillar'] = df.index.map(pillar_map)
    df['pillar'] = df['pillar'].fillna("Entertainment & Pop Culture") 
    
    return df


def get_categorized_nyt_data(target_date: str, limit: int = 25) -> pd.DataFrame:
    """
    Master pipeline execution module. Call this from external scripts to 
    get fully fetched and categorized news data as a single pandas DataFrame.
    """
    raw_df = fetch_nyt_historical(target_date=target_date, limit=limit)
    categorized_df = categorize_news(raw_df)
    print(categorized_df[['clean_title', 'pillar']])
    return categorized_df

get_categorized_nyt_data("2024-01-01", limit=25)'''

import os
import time
from datetime import datetime, timedelta

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

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
NYT_KEY = os.getenv("NYT_API_KEY")

client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None


SECTION_TO_PILLAR = {
    # Politics / Civic Life
    "Politics": "Politics & Civic Life",
    "U.S.": "Politics & Civic Life",
    "World": "Politics & Civic Life",
    "Washington": "Politics & Civic Life",
    "National": "Politics & Civic Life",

    # Sports
    "Sports": "Sports & Athletics",

    # Entertainment / Culture
    "Arts": "Entertainment & Pop Culture",
    "Movies": "Entertainment & Pop Culture",
    "Theater": "Entertainment & Pop Culture",
    "Television": "Entertainment & Pop Culture",
    "Music": "Entertainment & Pop Culture",
    "Books": "Entertainment & Pop Culture",
    "Style": "Entertainment & Pop Culture",
    "Fashion": "Entertainment & Pop Culture",

    # Science / Tech
    "Technology": "Science & Technology",
    "Science": "Science & Technology",
    "Climate": "Science & Technology",

    # Business
    "Business": "Business & Finance",
    "Business Day": "Business & Finance",
    "Your Money": "Business & Finance",
    "Real Estate": "Business & Finance",

    # Lifestyle / Wellness
    "Health": "Lifestyle & Wellness",
    "Well": "Lifestyle & Wellness",
    "Food": "Lifestyle & Wellness",
    "Travel": "Lifestyle & Wellness",
    "Magazine": "Lifestyle & Wellness",
}


AMBIGUOUS_SECTIONS = {
    "Opinion",
    "Briefing",
    "Podcasts",
    "The Upshot",
    "New York",
    "Obituaries",
    "Reader Center",
    "Corrections",
    "Unknown",
    "",
    None,
}


class ArticleClassification(BaseModel):
    global_id: str
    pillar: str


# ======================================================
# FETCH NYT ARTICLES
# ======================================================

def fetch_nyt_historical(target_date: str, limit: int = 10) -> pd.DataFrame:
    if not NYT_KEY:
        raise ValueError("Missing NYT_API_KEY environment variable.")

    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    formatted_date = target_date.replace("-", "")

    rows = []
    page = 0
    rank_counter = 1

    print(f"Fetching NYT articles for {target_date}...")

    while len(rows) < limit:
        params = {
            "begin_date": formatted_date,
            "end_date": formatted_date,
            "sort": "relevance",
            "page": page,
            "api-key": NYT_KEY,
        }

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 429:
            print("NYT rate limit hit. Sleeping 10 seconds...")
            time.sleep(10)
            continue

        if response.status_code != 200:
            raise Exception(
                f"NYT API Error {response.status_code}: {response.text}"
            )

        data = response.json()
        docs = data.get("response", {}).get("docs", [])

        if not docs:
            break

        for item in docs:
            headline = item.get("headline", {})
            title = headline.get("print_headline") or headline.get("main", "Unknown Title")

            section = (
                item.get("section_name")
                or item.get("news_desk")
                or "Unknown"
            )

            global_id = f"{target_date}_{rank_counter}"

            rows.append(
                {
                    "date": target_date,
                    "global_id": global_id,
                    "zeitgeist_rank": rank_counter,
                    "clean_title": title,
                    "source": "The New York Times",
                    "description": item.get("abstract", ""),
                    "nyt_section": section,
                    "web_url": item.get("web_url", ""),
                }
            )

            rank_counter += 1

            if len(rows) >= limit:
                break

        page += 1
        time.sleep(6)

    return pd.DataFrame(rows)


def fetch_nyt_7_day_period(
    end_date: str,
    days: int = 7,
    limit_per_day: int = 10,
) -> pd.DataFrame:
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    all_days = []

    for i in range(days):
        target_dt = end_dt - timedelta(days=i)
        target_date = target_dt.strftime("%Y-%m-%d")

        day_df = fetch_nyt_historical(
            target_date=target_date,
            limit=limit_per_day,
        )

        all_days.append(day_df)

    if not all_days:
        return pd.DataFrame()

    return pd.concat(all_days, ignore_index=True)


# ======================================================
# CLASSIFICATION
# ======================================================

def classify_by_section(row) -> str | None:
    section = row.get("nyt_section", "Unknown")

    if section in AMBIGUOUS_SECTIONS:
        return None

    return SECTION_TO_PILLAR.get(section)


def classify_ambiguous_batch_with_gemini(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    if not client:
        raise ValueError(
            "Gemini client is uninitialized. Check GEMINI_API_KEY in your .env file."
        )

    articles_data = [
        {
            "global_id": row["global_id"],
            "date": row["date"],
            "title": row["clean_title"],
            "description": row.get("description", ""),
            "section": row.get("nyt_section", "Unknown"),
        }
        for _, row in df.iterrows()
    ]

    prompt = f"""
You are classifying New York Times articles into exactly one cultural pillar.

Allowed pillars:
{PILLARS}

Rules:
- Return exactly one classification per article.
- Use only the allowed pillar names.
- Preserve each article's global_id exactly.
- Use the title, description, date, and NYT section.
- If unsure, choose the closest cultural pillar.

Articles:
{articles_data}
"""

    print(f"Sending {len(articles_data)} ambiguous articles to Gemini in ONE batch...")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": list[ArticleClassification],
            "temperature": 0.1,
        },
    )

    classifications = response.parsed

    return {
        item.global_id: item.pillar
        for item in classifications
        if item.pillar in PILLARS
    }


def categorize_7_day_batch(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["pillar"] = None
        return df

    df = df.copy()

    print(f"Classifying {len(df)} total articles across the 7-day period...")

    df["pillar"] = df.apply(classify_by_section, axis=1)

    ambiguous_df = df[df["pillar"].isna()]

    print(f"Classified locally by NYT section: {len(df) - len(ambiguous_df)}")
    print(f"Ambiguous articles sent to Gemini: {len(ambiguous_df)}")

    gemini_map = classify_ambiguous_batch_with_gemini(ambiguous_df)

    df["gemini_used"] = df["pillar"].isna()

    for global_id, pillar in gemini_map.items():
        df.loc[df["global_id"] == global_id, "pillar"] = pillar

    df["pillar"] = df["pillar"].fillna("Entertainment & Pop Culture")

    return df


# ======================================================
# SUMMARY OUTPUTS
# ======================================================

def create_daily_pillar_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["date", "pillar"])
        .size()
        .reset_index(name="article_count")
    )

    daily_totals = (
        summary.groupby("date")["article_count"]
        .sum()
        .reset_index(name="daily_total")
    )

    summary = summary.merge(daily_totals, on="date", how="left")

    summary["nyt_share"] = (
        summary["article_count"] / summary["daily_total"]
    )

    return summary.sort_values(["date", "nyt_share"], ascending=[True, False])


def run_nyt_7_day_pipeline(
    end_date: str,
    days: int = 7,
    limit_per_day: int = 10,
    article_output_path: str = "nyt_7_day_articles_classified.csv",
    summary_output_path: str = "nyt_7_day_pillar_summary.csv",
):
    raw_df = fetch_nyt_7_day_period(
        end_date=end_date,
        days=days,
        limit_per_day=limit_per_day,
    )

    categorized_df = categorize_7_day_batch(raw_df)

    summary_df = create_daily_pillar_summary(categorized_df)

    categorized_df.to_csv(article_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print("\nSaved files:")
    print(f"- {article_output_path}")
    print(f"- {summary_output_path}")

    print("\nDaily pillar summary:")
    print(summary_df)

    return categorized_df, summary_df


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    # Change this date whenever needed.
    # This will fetch this date plus the previous 6 days.
    END_DATE = "2026-06-16"

    run_nyt_7_day_pipeline(
        end_date=END_DATE,
        days=30,
        limit_per_day=10,
    )