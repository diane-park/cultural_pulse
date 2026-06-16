import os
from datetime import datetime, timedelta

import pandas as pd
from eventregistry import EventRegistry, QueryArticles, RequestArticlesInfo
from dotenv import load_dotenv


load_dotenv()


class EventRegistryPillarShareClient:
    """
    NewsAPI.ai / Event Registry client with CSV caching.

    Main compatible methods:
    - get_daily_category_shares(target_date)
    - get_category_shares(date_start, date_end)

    New cache methods:
    - get_daily_category_shares_cached(target_date)
    - get_daily_category_shares_for_dates_cached(dates)
    """

    DEFAULT_PILLAR_CATEGORIES = {
        "sports": ["dmoz/Sports"],
        "business_finance": ["dmoz/Business"],
        "technology": ["dmoz/Computers"],
        "health_wellness": ["dmoz/Health"],
        "entertainment": [
            "dmoz/Arts/Movies",
            "dmoz/Arts/Music",
            "dmoz/Arts/Television",
        ],
        "politics": ["dmoz/Society/Politics"],
    }

    def __init__(
        self,
        api_key: str,
        pillar_categories: dict | None = None,
        language: str = "eng",
        allow_use_of_archive: bool = False,
        cache_path: str = "newsapi_daily_category_shares.csv",
    ):
        self.api_key = api_key
        self.language = language
        self.cache_path = cache_path

        self.pillar_categories = (
            pillar_categories
            if pillar_categories is not None
            else self.DEFAULT_PILLAR_CATEGORIES
        )

        self.er = EventRegistry(
            apiKey=self.api_key,
            allowUseOfArchive=allow_use_of_archive,
        )

        self._category_uri_cache = {}

    # ======================================================
    # CACHE HELPERS
    # ======================================================

    def load_cache(self) -> pd.DataFrame:
        if not os.path.exists(self.cache_path):
            return pd.DataFrame(
                columns=[
                    "date",
                    "pillar",
                    "article_count",
                    "share_of_tracked_publishing",
                ]
            )

        df = pd.read_csv(self.cache_path)

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        return df

    def save_cache(self, df: pd.DataFrame) -> None:
        df = df.copy()

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        df = (
            df.drop_duplicates(
                subset=["date", "pillar"],
                keep="last",
            )
            .sort_values(["date", "pillar"])
            .reset_index(drop=True)
        )

        df.to_csv(self.cache_path, index=False)

    def append_to_cache(self, new_df: pd.DataFrame) -> pd.DataFrame:
        cache_df = self.load_cache()

        combined = pd.concat(
            [cache_df, new_df],
            ignore_index=True,
        )

        self.save_cache(combined)

        return self.load_cache()

    # ======================================================
    # EVENT REGISTRY CORE
    # ======================================================

    def get_category_uri(self, category_name: str) -> str:
        if category_name in self._category_uri_cache:
            return self._category_uri_cache[category_name]

        uri = self.er.getCategoryUri(category_name)

        if uri is None:
            raise ValueError(f"Could not resolve category: {category_name}")

        self._category_uri_cache[category_name] = uri

        return uri

    def count_articles_for_category(
        self,
        category_uri: str,
        date_start: str,
        date_end: str,
    ) -> int:
        query = QueryArticles(
            categoryUri=category_uri,
            dateStart=date_start,
            dateEnd=date_end,
            lang=self.language,
        )

        query.setRequestedResult(
            RequestArticlesInfo(count=0)
        )

        result = self.er.execQuery(query)

        try:
            return int(result["articles"]["totalResults"])
        except KeyError as exc:
            raise KeyError(
                f"Could not find totalResults in response: {result}"
            ) from exc

    def count_pillar_articles(
        self,
        pillar: str,
        date_start: str,
        date_end: str,
    ) -> int:
        if pillar not in self.pillar_categories:
            raise ValueError(f"Unknown pillar: {pillar}")

        total = 0

        for category_name in self.pillar_categories[pillar]:
            category_uri = self.get_category_uri(category_name)

            total += self.count_articles_for_category(
                category_uri=category_uri,
                date_start=date_start,
                date_end=date_end,
            )

        return total

    def get_category_shares(
        self,
        date_start: str,
        date_end: str,
    ) -> pd.DataFrame:
        rows = []

        for pillar in self.pillar_categories:
            count = self.count_pillar_articles(
                pillar=pillar,
                date_start=date_start,
                date_end=date_end,
            )

            rows.append(
                {
                    "pillar": pillar,
                    "article_count": count,
                }
            )

        df = pd.DataFrame(rows)

        total_count = df["article_count"].sum()

        if total_count == 0:
            df["share_of_tracked_publishing"] = 0.0
        else:
            df["share_of_tracked_publishing"] = (
                df["article_count"] / total_count
            )

        return (
            df.sort_values(
                "share_of_tracked_publishing",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    def get_daily_category_shares(
        self,
        target_date: str,
    ) -> pd.DataFrame:
        start = target_date

        end = (
            datetime.strptime(target_date, "%Y-%m-%d")
            + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        df = self.get_category_shares(
            date_start=start,
            date_end=end,
        )

        return df

    # ======================================================
    # CACHED METHODS
    # ======================================================

    def get_daily_category_shares_cached(
        self,
        target_date: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        target_date = pd.to_datetime(target_date).strftime("%Y-%m-%d")

        cache_df = self.load_cache()

        if not refresh:
            cached_day = cache_df[
                cache_df["date"] == target_date
            ].copy()

            if not cached_day.empty:
                print(f"Using cached NewsAPI.ai data for {target_date}")

                return cached_day[
                    [
                        "pillar",
                        "article_count",
                        "share_of_tracked_publishing",
                    ]
                ].reset_index(drop=True)

        print(f"Fetching NewsAPI.ai data for {target_date}")

        day_df = self.get_daily_category_shares(target_date)
        day_df["date"] = target_date

        self.append_to_cache(day_df)

        return day_df[
            [
                "pillar",
                "article_count",
                "share_of_tracked_publishing",
            ]
        ].reset_index(drop=True)

    def get_daily_category_shares_for_dates_cached(
        self,
        dates: list[str],
        refresh: bool = False,
    ) -> pd.DataFrame:
        rows = []

        for date in dates:
            day_df = self.get_daily_category_shares_cached(
                target_date=date,
                refresh=refresh,
            )

            day_df["date"] = pd.to_datetime(date).strftime("%Y-%m-%d")

            rows.append(day_df)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "date",
                    "pillar",
                    "article_count",
                    "share_of_tracked_publishing",
                ]
            )

        return pd.concat(rows, ignore_index=True)[
            [
                "date",
                "pillar",
                "article_count",
                "share_of_tracked_publishing",
            ]
        ]

    def refresh_cache_for_dates(
        self,
        dates: list[str],
    ) -> pd.DataFrame:
        return self.get_daily_category_shares_for_dates_cached(
            dates=dates,
            refresh=True,
        )


# ======================================================
# OPTIONAL DIRECT RUN
# ======================================================

if __name__ == "__main__":
    api_key = (
        os.getenv("NEWSAPI_KEY")
        or os.getenv("NEWSAPI_AI_KEY")
        or os.getenv("EVENT_REGISTRY_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "Missing API key. Add NEWSAPI_KEY, NEWSAPI_AI_KEY, or EVENT_REGISTRY_API_KEY to .env"
        )

    client = EventRegistryPillarShareClient(
        api_key=api_key,
        allow_use_of_archive=False,
        cache_path="newsapi_daily_category_shares.csv",
    )

    end_date = datetime.today()
    dates = [
        (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(7)
    ]

    df = client.get_daily_category_shares_for_dates_cached(dates)

    print(df)

    print("\nSaved/updated newsapi_daily_category_shares.csv")