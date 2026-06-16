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

import os

import numpy as np
import pandas as pd

from dotenv import load_dotenv

from scipy.stats import norm
from datetime import datetime, timedelta

from news import EventRegistryPillarShareClient
from new_york_times import get_categorized_nyt_data


# ======================================================
# CONFIG
# ======================================================

load_dotenv()

EVENT_REGISTRY_API_KEY = os.getenv("news_api_key")

NYT_LIMIT = 10

PILLAR_MAP = {
    "sports": "Sports & Athletics",
    "business_finance": "Business & Finance",
    "technology": "Science & Technology",
    "health_wellness": "Lifestyle & Wellness",
    "entertainment": "Entertainment & Pop Culture",
    "politics": "Politics & Civic Life",
}

PILLARS = [
    "Politics & Civic Life",
    "Sports & Athletics",
    "Entertainment & Pop Culture",
    "Science & Technology",
    "Business & Finance",
    "Lifestyle & Wellness",
]


# ======================================================
# HELPERS
# ======================================================

def calculate_nyt_share(
    nyt_df: pd.DataFrame
) -> pd.DataFrame:

    counts = (
        nyt_df["pillar"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "pillar",
        "nyt_count"
    ]

    counts["nyt_share"] = (
        counts["nyt_count"]
        / counts["nyt_count"].sum()
    )

    return counts[
        ["pillar", "nyt_share"]
    ]


def calculate_composite_share(
    er_df: pd.DataFrame,
    nyt_df: pd.DataFrame
) -> pd.DataFrame:

    nyt_share_df = calculate_nyt_share(
        nyt_df
    )

    er = er_df.copy()

    er["pillar"] = (
        er["pillar"]
        .map(PILLAR_MAP)
    )

    er = er.rename(
        columns={
            "share_of_tracked_publishing":
            "er_share"
        }
    )

    merged = pd.merge(
        er[["pillar", "er_share"]],
        nyt_share_df,
        on="pillar",
        how="outer"
    )

    merged = merged.fillna(0)

    #
    # Composite Score
    #
    # Event Registry = broad ecosystem
    # NYT = elite editorial signal
    #

    merged["composite_share"] = (
        0.70 * merged["er_share"]
        +
        0.30 * merged["nyt_share"]
    )

    return merged[
        [
            "pillar",
            "er_share",
            "nyt_share",
            "composite_share"
        ]
    ]


def calculate_day_composite(
    target_date: str,
    er_client
) -> pd.DataFrame:

    print(f"\nProcessing {target_date}")

    er_df = er_client.get_daily_category_shares(
        target_date
    )

    nyt_df = get_categorized_nyt_data(
        target_date,
        limit=NYT_LIMIT
    )

    return calculate_composite_share(
        er_df,
        nyt_df
    )


# ======================================================
# MAIN METRIC ENGINE
# ======================================================

def calculate_cultural_metrics(
    target_date: str
):

    er_client = EventRegistryPillarShareClient(
        api_key=EVENT_REGISTRY_API_KEY,
        allow_use_of_archive=False
    )

    #
    # Current Day
    #

    current_df = calculate_day_composite(
        target_date,
        er_client
    )

    #
    # Historical Window
    #
    # Previous 30 days
    #

    historical_results = []

    target_dt = datetime.strptime(
        target_date,
        "%Y-%m-%d"
    )

    for i in range(1, 31):

        hist_day = (
            target_dt
            - timedelta(days=i)
        ).strftime("%Y-%m-%d")

        try:

            day_df = calculate_day_composite(
                hist_day,
                er_client
            )

            day_df["date"] = hist_day

            historical_results.append(
                day_df
            )

        except Exception as e:

            print(
                f"Failed {hist_day}: {e}"
            )

    historical_df = pd.concat(
        historical_results,
        ignore_index=True
    )

    #
    # Momentum Calculation
    #

    output_rows = []

    for _, row in current_df.iterrows():

        pillar = row["pillar"]

        current_share = row[
            "composite_share"
        ]

        history = historical_df[
            historical_df["pillar"]
            == pillar
        ]["composite_share"]

        mean = history.mean()

        std = history.std()

        if pd.isna(std) or std == 0:
            z_score = 0
        else:
            z_score = (
                current_share - mean
            ) / std

        percentile = (
            norm.cdf(z_score)
            * 100
        )

        popularity_score = (
            current_share * 100
        )

        momentum_score = (
            0.80 * percentile
            +
            0.20 * popularity_score
        )

        output_rows.append(
            {
                "pillar": pillar,
                "current_share":
                    round(
                        current_share,
                        4
                    ),

                "historical_mean":
                    round(
                        mean,
                        4
                    ),

                "historical_std":
                    round(
                        std,
                        4
                    ),

                "z_score":
                    round(
                        z_score,
                        2
                    ),

                "percentile":
                    round(
                        percentile,
                        2
                    ),

                "popularity_score":
                    round(
                        popularity_score,
                        2
                    ),

                "momentum_score":
                    round(
                        momentum_score,
                        2
                    ),
            }
        )

    results = pd.DataFrame(
        output_rows
    )

    results = results.sort_values(
        "momentum_score",
        ascending=False
    )

    return results


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":

    today = datetime.today().strftime(
        "%Y-%m-%d"
    )

    results = calculate_cultural_metrics(
        today
    )

    print("\n")
    print("=" * 80)
    print("CULTURAL MOMENTUM")
    print("=" * 80)

    print(
        results[
            [
                "pillar",
                "popularity_score",
                "momentum_score",
                "percentile",
                "z_score",
            ]
        ]
    )

    results.to_csv(
        "cultural_momentum.csv",
        index=False
    )

    print(
        "\nSaved cultural_momentum.csv"
    )