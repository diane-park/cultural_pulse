# news
from eventregistry import EventRegistry, QueryArticles, RequestArticlesInfo
from datetime import datetime, timedelta
import pandas as pd


class EventRegistryPillarShareClient:
    """
    Client for calculating share of tracked publishing by cultural pillar
    using Event Registry / NewsAPI.ai category counts.
    """

    DEFAULT_PILLAR_CATEGORIES = {
        "sports": [
            "dmoz/Sports"
        ],
        "business_finance": [
            "dmoz/Business"
        ],
        "technology": [
            "dmoz/Computers"
        ],
        "health_wellness": [
            "dmoz/Health"
        ],
        "entertainment": [
            "dmoz/Arts/Movies",
            "dmoz/Arts/Music",
            "dmoz/Arts/Television"
        ],
        "politics": [
            "dmoz/Society/Politics"
        ]
    }

    def __init__(
        self,
        api_key: str,
        pillar_categories: dict | None = None,
        language: str = "eng",
        allow_use_of_archive: bool = False
    ):
        self.api_key = api_key
        self.language = language
        self.pillar_categories = (
            pillar_categories
            if pillar_categories is not None
            else self.DEFAULT_PILLAR_CATEGORIES
        )

        self.er = EventRegistry(
            apiKey=self.api_key,
            allowUseOfArchive=allow_use_of_archive
        )

        self._category_uri_cache = {}

    def get_category_uri(self, category_name: str) -> str:
        """
        Resolve and cache an Event Registry category URI.
        """

        if category_name in self._category_uri_cache:
            return self._category_uri_cache[category_name]

        uri = self.er.getCategoryUri(category_name)

        if uri is None:
            raise ValueError(
                f"Could not resolve category: {category_name}"
            )

        self._category_uri_cache[category_name] = uri

        return uri

    def count_articles_for_category(
        self,
        category_uri: str,
        date_start: str,
        date_end: str
    ) -> int:
        """
        Count articles for a single Event Registry category URI.
        """

        query = QueryArticles(
            categoryUri=category_uri,
            dateStart=date_start,
            dateEnd=date_end,
            lang=self.language
        )

        query.setRequestedResult(
            RequestArticlesInfo(
                count=0
            )
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
        date_end: str
    ) -> int:
        """
        Sum article counts across all categories assigned to a pillar.
        """

        if pillar not in self.pillar_categories:
            raise ValueError(
                f"Unknown pillar: {pillar}"
            )

        total = 0

        for category_name in self.pillar_categories[pillar]:
            category_uri = self.get_category_uri(category_name)

            total += self.count_articles_for_category(
                category_uri=category_uri,
                date_start=date_start,
                date_end=date_end
            )

        return total

    def get_category_shares(
        self,
        date_start: str,
        date_end: str
    ) -> pd.DataFrame:
        """
        Calculate share of tracked publishing over any date range.

        date_start and date_end should be strings in YYYY-MM-DD format.
        date_end is treated as exclusive in your own usage convention.
        """

        rows = []

        for pillar in self.pillar_categories:
            count = self.count_pillar_articles(
                pillar=pillar,
                date_start=date_start,
                date_end=date_end
            )

            rows.append({
                "pillar": pillar,
                "article_count": count
            })

        df = pd.DataFrame(rows)

        total_count = df["article_count"].sum()

        if total_count == 0:
            df["share_of_tracked_publishing"] = 0.0
        else:
            df["share_of_tracked_publishing"] = (
                df["article_count"] / total_count
            )

        return df.sort_values(
            "share_of_tracked_publishing",
            ascending=False
        ).reset_index(drop=True)

    def get_daily_category_shares(
        self,
        target_date: str
    ) -> pd.DataFrame:
        """
        Calculate category shares for a single day.

        target_date should be a string in YYYY-MM-DD format.
        """

        start = target_date

        end = (
            datetime.strptime(target_date, "%Y-%m-%d")
            + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        return self.get_category_shares(
            date_start=start,
            date_end=end
        )