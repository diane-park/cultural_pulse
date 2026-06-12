# code for app
import streamlit as st
from datetime import date
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

from news import (
    EventRegistryPillarShareClient
)

# -------------------------
# CONFIG
# -------------------------

load_dotenv()

API_KEY = os.getenv("news_api_key")

# -------------------------
# PAGE
# -------------------------

st.set_page_config(
    page_title="Cultural Zeitgeist",
    layout="wide"
)

st.title("Cultural Zeitgeist Tracker")

# -------------------------
# DATE PICKER
# -------------------------

selected_date = st.date_input(
    "Select Date",
    value=date.today()
)

# Convert to format expected by class
target_date = selected_date.strftime(
    "%Y-%m-%d"
)

st.write(
    f"Selected Date: {target_date}"
)

# -------------------------
# BUTTON
# -------------------------

if st.button("Run Analysis"):

    with st.spinner(
        "Fetching Event Registry data..."
    ):

        client = EventRegistryPillarShareClient(
            api_key=API_KEY
        )

        df = client.get_daily_category_shares(
            target_date
        )

    st.success("Complete")

    st.subheader(
        "Share of Tracked Publishing"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader(
        "Pillar Ranking"
    )

    

    categories = df["pillar"].tolist()
    values = df["share_of_tracked_publishing"].tolist()

    # Close the polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Publishing Share"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )
        ),
        showlegend=False,
        title="Cultural Zeitgeist Radar"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )