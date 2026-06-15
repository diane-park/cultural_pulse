import os
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
    return categorized_df