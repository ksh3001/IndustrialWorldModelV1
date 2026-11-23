"""Industrial World Model — demo UI.

    streamlit run app.py

Sidebar navigation, grouped the way you present it:
  Present    1 Overview · 2 Live demo · 3 Real video · 4 Warehouse · 5 Demo videos
  Evidence   4 Results  · 5 Shift reports
  Sources    Camera alerts
Below the navigation: where you are in the presentation flow, and system status.
"""

import streamlit as st

from ui import pages
from ui.common import sidebar_extras, style

st.set_page_config(page_title="Industrial World Model", page_icon="docs/logo_icon.png", layout="wide",
                   initial_sidebar_state="expanded")
style()
st.logo("docs/logo.png", icon_image="docs/logo_icon.png", size="large")

P = {
    "overview": st.Page(pages.overview, title="Overview", icon=":material/home:", url_path="overview", default=True),
    "live": st.Page(pages.live_demo, title="Live demo", icon=":material/play_circle:", url_path="live"),
    "real": st.Page(pages.real_video, title="Real video", icon=":material/movie:", url_path="real"),
    "warehouse": st.Page(pages.warehouse, title="Warehouse", icon=":material/warehouse:", url_path="warehouse"),
    "videos": st.Page(pages.demo_videos, title="Demo videos", icon=":material/videocam:", url_path="videos"),
    "results": st.Page(pages.results, title="Results", icon=":material/insights:", url_path="results"),
    "reports": st.Page(pages.reports, title="Shift reports", icon=":material/description:", url_path="reports"),
    "camera": st.Page(pages.camera, title="Camera alerts", icon=":material/photo_camera:", url_path="camera"),
}
pages.NAV.update(P)

nav = st.navigation({
    "Present": [P["overview"], P["live"], P["real"], P["warehouse"], P["videos"]],
    "Evidence": [P["results"], P["reports"]],
    "Sources": [P["camera"]],
})
sidebar_extras(nav.title, ["Overview", "Live demo", "Real video", "Warehouse", "Demo videos", "Results",
                          "Shift reports"])
nav.run()
