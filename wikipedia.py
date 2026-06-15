import os
import datetime
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

EXCLUDED_PAGES = [
    "Main_Page", "Special:Search", "Wikipedia:", "Portal:", "Main Page",
    "Special:", "Search", "File:", "Help:", "Category:", "Talk:"
]

# Wikimedia requires a descriptive User-Agent 
USER_AGENT = "ZeitgeistRadarBot/2.0 (your_email@example.com)"

# Extract the Gemini key safely
GEMINI_KEY = os.getenv('GEMINI_API_KEY')

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

def fetch_wikipedia_historical(target_date: str, limit: int = 25) -> pd.DataFrame:
    """
    Fetches the top viewed Wikipedia articles for a specific past date.
    Filters out system pages and assigns a prominence rank.
    
    Parameters:
        target_date (str): Format 'YYYY-MM-DD'
        limit (int): Total number of articles to return (default 25)
    """
    date_obj = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m")
    day = date_obj.strftime("%d")
    
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{year}/{month}/{day}"
    headers = {"User-Agent": USER_AGENT}
    
    print(f"Fetching Wikimedia API for {target_date}...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 404:
        raise Exception("Data not found. (Note: Wikipedia data is usually finalized 1-2 days late. Try an older date).")
    elif response.status_code != 200:
        raise Exception(f"Failed to fetch Wikipedia data: {response.status_code}\n{response.text}")
        
    data = response.json()
    articles = data['items'][0]['articles']
    
    df = pd.DataFrame(articles)
    
    # Filter out system and portal pages
    for pattern in EXCLUDED_PAGES:
        df = df[~df['article'].str.contains(pattern, case=False, na=False)]
    
    # Clean titles and enforce the exact limit
    df['clean_title'] = df['article'].str.replace('_', ' ')
    df = df.head(limit).copy()
    
    # Assign a rank position based on the filtered results
    df['zeitgeist_rank'] = range(1, len(df) + 1)
    
    return df


def categorize_wikipedia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sends the gathered Wikipedia titles to Gemini to categorize them 
    into the 6 predefined culture pillars.
    """
    if not client:
        raise ValueError("Gemini Client is uninitialized. Verify GEMINI_API_KEY in your .env file.")
        
    if df.empty:
        print("Received an empty DataFrame. Skipping categorization.")
        df['pillar'] = None
        return df

    print(f"Categorizing {len(df)} Wikipedia articles using Gemini...")
    
    # Build structural mapping dataset for the prompt payload
    articles_data = [{"id": i, "title": row['clean_title']} for i, row in df.iterrows()]
    
    prompt = f"""
    You are analyzing the American zeitgeist. Categorize each of the following Wikipedia article titles into the most appropriate pillar.
    If a title is a person, movie, or company, categorize it based on what they are most famous for right now.
    
    The strict pillars are:
    {PILLARS}
    
    Titles to categorize:
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


def get_categorized_wiki_data(target_date: str, limit: int = 25) -> pd.DataFrame:
    """
    Master pipeline execution module. Call this from external scripts to 
    get fully fetched and categorized Wikipedia data as a single pandas DataFrame.
    
    Returns a DataFrame with columns: ['page rank', 'page title', 'pillar']
    """
    # 1. Pull raw data from API
    raw_df = fetch_wikipedia_historical(target_date=target_date, limit=limit)
    
    # 2. Categorize data using Gemini
    categorized_df = categorize_wikipedia(raw_df)
    
    # 3. Filter and rename to your exact specifications
    final_df = categorized_df[['zeitgeist_rank', 'clean_title', 'pillar']].copy()
    final_df.columns = ['page rank', 'page title', 'pillar']
    
    return final_df