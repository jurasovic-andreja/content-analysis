import os, io, re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import openai


# === Configuration ===
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=openai.api_key)

# === DataFrameSummary TOOL (pandas only) =========================================
def DataFrameSummaryTool(url: str) -> str:
    """Generate a summary prompt string for the LLM based on the website content."""
    prompt = f"""
        You are tasked with conducting a thorough and strategic analysis of the content on the specified webpage: {url}. Your goal is to provide a multi-dimensional assessment that not only describes the current state but also offers actionable recommendations for optimizing the user experience and achieving specific content objectives.

        Your analysis should encompass the following key areas:

            Comprehensive Page Overview & Strategic Purpose (4-5 sentences):
                Beyond a basic description, provide an in-depth summary that articulates the primary subject matter, the page's intended strategic purpose (e.g., lead generation, information dissemination, product education, brand building), and the target audience it aims to serve.
                Critically evaluate how effectively the current content aligns with this strategic purpose and resonates with the intended audience.

            Detailed Content Element Dissection & Evaluation:
                Conduct a meticulous examination of the following crucial on-page content elements, going beyond mere identification to offer a qualitative assessment of their effectiveness:
                    Meta Title & Meta Description: Evaluate their clarity, keyword relevance, compellingness for click-through, and adherence to best practices for search engine visibility.
                    Internal Links: Identify the types and placement of internal links. Assess their strategic value for user navigation, content discovery, and SEO (e.g., do they guide users to relevant related content or conversion paths?).
                    Bold Text/Emphasis: Analyze the use of bolding. Is it used effectively to highlight key information, improve readability, and guide the user's eye, or is it distracting?
                    Images & Visual Media (including Image Alt Text): Describe the images, their relevance to the content, and their visual quality. Critically assess the presence, accuracy, and descriptive nature of all image alt texts for accessibility and SEO.
                    Blog/Content Structure (if applicable): Evaluate the overall logical flow, use of headings (H1, H2, H3), lists, and other structural elements that enhance readability and information hierarchy. Is the content easy to scan and digest?
                    Paragraphs: Assess paragraph length, readability, sentence structure, and overall flow. Is the language clear, concise, and engaging for the target audience?
                    Banner or Call-to-Action (CTA) Elements: Identify any prominent banners or CTAs. Evaluate their visibility, clarity of message, compellingness, and strategic placement in guiding user behavior.

            Actionable Insights & Optimization Recommendations:
                For each content element analyzed in point 2, provide detailed, data-informed, and actionable insights. This is not just about pointing out what exists, but about why it works or doesn't work from a user experience (UX), conversion optimization, or search engine optimization (SEO) perspective.
                Based on your insights, propose specific, practical, and measurable recommendations for improvement. How can each element be refined to:
                    Enhance User Experience (UX): Improve readability, engagement, accessibility, and overall user satisfaction.
                    Boost Content Effectiveness: Better achieve the page's strategic purpose, whether it's driving conversions, reducing bounce rates, or increasing time on page.
                    Optimize for Search Engines: Improve organic visibility and relevance for target keywords.
                Prioritize your recommendations, suggesting which improvements would yield the most significant impact on user experience and content goals."""
    return prompt

# === DataInsightAgent (upload-time only) ===============================

def DataInsightAgent(url: str) -> str:
    """Uses the LLM to generate a brief summary and possible questions for the uploaded dataset."""
    prompt = DataFrameSummaryTool(url)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "detailed thinking off. You are a content analyst providing brief, focused insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"Error generating dataset insights: {exc}"

# === Helpers ===========================================================

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

# === Main Streamlit App ===============================================

def main():
    st.set_page_config(layout="wide")
    if "insights" not in st.session_state:
        st.session_state.insights = {}
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "page_data" not in st.session_state:
        st.session_state.page_data = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.title("Content Analyst Assistant")
    with header_col2:
        if st.button("🔄 Refresh Session", help="Clear all data and start fresh"):
            st.session_state.clear()
            st.rerun()


    tab1, = st.tabs(["Content Analysis"])

    with tab1:
        left_margin, content_col, right_margin = st.columns([1, 4, 1])
        
        with content_col:
            st.header("1️⃣ Analyse a web page")
            url = st.text_input("Paste a full URL and press Enter", key="url_input")

            if st.button("Analyse") and url:
                with st.spinner("Fetching & analysing…"):
                    try:
                        page = DataFrameSummaryTool(url)
                        summary = DataInsightAgent(url)
                        st.session_state.page_data = page
                        st.session_state.summary = summary
                    except Exception as exc:
                        st.error(f"❌ {exc}")
                        st.stop()

            if st.session_state.summary:
                st.markdown("### Page summary & insights")
                st.markdown(st.session_state.summary)


if __name__ == "__main__":
    main()
