import os
import time
import re
import json
import requests
import pandas as pd
import streamlit as st
import openai

from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from langdetect import detect

# -----------------------------------------------------------------------------
# SECTION 1: HELPER FUNCTIONS (UNCHANGED from your original script)
# -----------------------------------------------------------------------------

def extract_main_content(url: str, max_retries: int = 3, backoff_factor: int = 2) -> BeautifulSoup | None:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SEO-Analyzer/1.0; +https://example.com/bot)"
    }
    wait = 1
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup.find_all(['header','footer','nav','aside','form','noscript','script','style']):
                tag.decompose()
            return soup
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status == 429:
                retry_after = e.response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after)
                st.write(f"⚠️ Received 429 for {url}. Waiting {wait}s before retrying… (Attempt {attempt})")
                time.sleep(wait)
                wait *= backoff_factor
                continue
            else:
                st.write(f"❌ HTTP {status} error fetching {url}. Skipping.")
                return None
        except requests.exceptions.RequestException as e:
            st.write(f"⚠️ Request error fetching {url}: {e}. Retrying in {wait}s… (Attempt {attempt})")
            time.sleep(wait)
            wait *= backoff_factor
            continue
    st.write(f"❌ Failed to fetch {url} after {max_retries} attempts.")
    return None


def list_bold_text(soup: BeautifulSoup) -> list[str]:
    return [
        el.get_text(strip=True)
        for el in soup.find_all(['b','strong'])
        if el.get_text(strip=True)
    ]


def count_bold_text(soup: BeautifulSoup) -> bool:
    bold_texts = list_bold_text(soup)
    return len(bold_texts) < 8


def bold_words(soup: BeautifulSoup) -> bool:
    for txt in list_bold_text(soup):
        if len(txt.split()) > 7:
            return True
    return False


def analyze_images(soup: BeautifulSoup) -> bool:
    image_alt = [
        img.get('alt')
        for img in soup.find_all('img')
        if img.get('alt') and 'logo' not in img.get('alt').lower()
    ]
    return len(image_alt) < 3


def analyze_images_text(soup: BeautifulSoup) -> bool:
    image_alt = [
        img.get('alt')
        for img in soup.find_all('img')
        if img.get('alt') and 'logo' not in img.get('alt').lower()
    ]
    total = sum(1 for img in soup.find_all('img') if img.get('alt') and 'logo' not in img.get('alt').lower())
    return (total > 0 and not image_alt)


def show_images_text(soup: BeautifulSoup) -> list[str]:
    return [
        img.get('alt')
        for img in soup.find_all('img')
        if img.get('alt') and 'logo' not in img.get('alt').lower()
    ]


def analyze_meta_title(soup: BeautifulSoup) -> int:
    title = soup.find('title')
    title_text = title.get_text(strip=True) if title else ""
    char_count = len(title_text)
    if char_count < 30:
        return 0
    elif char_count > 60:
        return 1
    else:
        return 10


def meta_title_show(soup: BeautifulSoup) -> str:
    title = soup.find('title')
    return title.get_text(strip=True) if title else ""


def analyze_meta_description(soup: BeautifulSoup) -> int:
    meta = soup.find('meta', attrs={'name': 'description'})
    desc = meta['content'].strip() if (meta and 'content' in meta.attrs) else ""
    char_count = len(desc)
    if char_count < 120:
        return 0
    elif char_count > 160:
        return 1
    else:
        return 10


def meta_description(soup: BeautifulSoup) -> str:
    meta = soup.find('meta', attrs={'name': 'description'})
    return meta['content'].strip() if (meta and 'content' in meta.attrs) else ""


def analyze_h1(soup: BeautifulSoup) -> bool:
    h1_tags = soup.find_all('h1')
    return len(h1_tags) > 1


def h1_show(soup: BeautifulSoup) -> str:
    h1 = soup.find('h1')
    return h1.get_text(strip=True) if h1 else ""


def analyze_h3(soup: BeautifulSoup) -> bool:
    return soup.find('h3') is None


def analyze_h5_and_h6(soup: BeautifulSoup) -> bool:
    return bool(soup.find('h5') or soup.find('h6'))


def analyze_paragraphs(soup: BeautifulSoup) -> dict[str, str]:
    issues = {}
    for p in soup.find_all('p'):
        text = p.get_text(strip=True)
        raw_sentences = re.split(r'[.!?]+(?=\s|$)', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if len(sentences) > 3:
            issues[text] = "space off after 2-3 sentences"
    return issues


def analyze_bullet_lists(soup: BeautifulSoup) -> bool:
    return not bool(soup.find_all(['ul','ol']))


def analyze_banner(soup: BeautifulSoup) -> bool:
    text = soup.get_text(strip=True).lower()
    for token in ['sign up','subscribe','discount']:
        if token in text:
            return False
    return True


def analyze_internal_links(url: str) -> tuple[bool, bool, str]:
    content_div = extract_main_content(url)
    base_domain = urlparse(url).netloc

    internal_links = []
    for a in content_div.find_all("a", href=True):
        href = a["href"]
        # Make absolute URL if relative
        full_url = urljoin(url, href)
        link_domain = urlparse(full_url).netloc
        # Check if link is internal
        if link_domain == base_domain:
            internal_links.append((full_url, a.get_text(strip=True)))

    too_little_internal_links = len(internal_links) < 5
    too_much_linked_seq = False
    example_long_anchor = ""

    for url, anchor_text in internal_links:
        if len(anchor_text.split()) > 6:
            too_much_linked_seq = True
            example_long_anchor = anchor_text
            break


    return too_little_internal_links, too_much_linked_seq, example_long_anchor


def analyze_primary(client: openai.Client, primary: str, text: str) -> bool:
    if not primary or not text:
        return False

    prompt = (
        "You are a helpful assistant specialized in analyzing the content of the website. "
        f"Try to find this element {primary} by finding its contextual and semantic match in the following text: {text} "
        "A match is considered even if the words don't match exactly but the concept or idea matches. "
        "Answer only 'yes' or 'no'."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a semantic analysis tool for SEO keywords."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0
        )
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
    except Exception:
        return False


def analyze_sec(client: openai.Client, secondaries: list[str], text: str) -> bool:
    if not secondaries or not text:
        return False

    prompt = (
        "You are a helpful assistant specialized in analyzing the content of the website. "
        f"Try to find each element of {secondaries} by finding its contextual and semantic match in the following text: {text} "
        "A match is considered even if the words don't match exactly but the concept or idea matches. "
        "Answer only 'yes' or 'no'."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a semantic analysis tool for SEO secondary keywords."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0
        )
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
    except Exception:
        return False


# -----------------------------------------------------------------------------
# SECTION 2: CSV PARSER (UNCHANGED)
# -----------------------------------------------------------------------------

def parse_keywords_csv(csv_file) -> dict[str, dict[str, any]]:
    df = pd.read_csv(csv_file, dtype=str).fillna("")
    grouped: dict[str, dict[str, any]] = {}
    current_url = None

    for _, row in df.iterrows():
        url = row["url"].strip()
        prim = row["primary kw"].strip()
        sec = row["secundary kw"].strip()

        if url:
            current_url = url
            grouped[current_url] = {
                "primary_kw": prim,
                "secondary_kw": []
            }
            if sec:
                grouped[current_url]["secondary_kw"].append(sec)
        else:
            if current_url and sec:
                grouped[current_url]["secondary_kw"].append(sec)

    return grouped


# -----------------------------------------------------------------------------
# SECTION 3: MERGED ANALYSIS FUNCTION (slightly trimmed)
# -----------------------------------------------------------------------------

LANGUAGE_MAP = {
    "en": "English",
    "hr": "Croatian",
    "sr": "Serbian",
    "bs": "Bosnian",
    "de": "German",
    "fr": "French",
    "es": "Spanish"
}

def analyze_kws_from_csv(urls: list[str], keywords_dict: dict[str, dict[str, any]], client: openai.Client) -> dict[str, dict[str, any]]:
    results: dict[str, dict[str, any]] = {}

    for url in urls:
        info = keywords_dict.get(url, {})
        primary = info.get("primary_kw", "")
        secondaries = info.get("secondary_kw", [])

        content_soup = extract_main_content(url)
        if content_soup is None:
            results[url] = {"error": "Could not fetch page (HTTP error or 429) – skipped analysis"}
            continue

        full_text = content_soup.get_text(strip=True)

        html_tag = content_soup.find('html')
        if html_tag and html_tag.get('lang'):
            lang_code = html_tag.get('lang').split('-')[0].lower()
        else:
            try:
                lang_code = detect(full_text)
            except Exception:
                lang_code = "en"
        lang_name = LANGUAGE_MAP.get(lang_code, "English")

        metrics: dict[str, dict[str, str]] = {}
        metrics["_lang"] = lang_code

        # 1) Page Title
        meta_title_text = meta_title_show(content_soup)
        title_flag = analyze_meta_title(content_soup)
        if not meta_title_text:
            rec = "Add a page title of ~45 characters that includes your primary keyword."
            example = "(none found)"
        elif title_flag == 0:
            rec = "Expand the page title to ~45 characters to improve SEO visibility."
            example = meta_title_text
        elif title_flag == 1:
            rec = "Shorten the page title to ~45 characters to ensure it displays fully in search results."
            example = meta_title_text
        else:
            rec = "/"
            example = meta_title_text
        metrics["Page Title"] = {"recommendation": rec, "example": example}

        # Primary KW in Title
        has_primary_in_title = analyze_primary(client, primary, meta_title_text)
        if primary and not has_primary_in_title:
            rec = f"Include your primary keyword (“{primary}”) in the page title for better relevance."
            example = meta_title_text if meta_title_text else "(no title to show)"
        else:
            rec = "/"
            example = meta_title_text if meta_title_text else ""
        metrics["Primary KW in Title"] = {"recommendation": rec, "example": example}

        # 2) Meta Description
        meta_desc_text = meta_description(content_soup)
        desc_flag = analyze_meta_description(content_soup)
        if not meta_desc_text:
            rec = "Add a meta description of ~130 characters that summarizes the page and includes a CTA."
            example = "(none found)"
        elif desc_flag == 0:
            rec = "Expand the meta description to ~130 characters to improve click-through rates."
            example = meta_desc_text
        elif desc_flag == 1:
            rec = "Shorten the meta description to ~130 characters so it doesn’t get cut off in search results."
            example = meta_desc_text
        else:
            rec = "/"
            example = meta_desc_text
        metrics["Meta Description"] = {"recommendation": rec, "example": example}

        # Primary KW in Description
        has_primary_in_desc = analyze_primary(client, primary, meta_desc_text)
        if primary and not has_primary_in_desc:
            rec = f"Include your primary keyword (“{primary}”) in the meta description for better relevance."
            example = meta_desc_text if meta_desc_text else ""
        else:
            rec = "/"
            example = meta_desc_text if meta_desc_text else ""
        metrics["Primary KW in Description"] = {"recommendation": rec, "example": example}

        # 3) H1 Structure
        h1_tags = content_soup.find_all("h1")
        if not h1_tags:
            rec = "Add exactly one <h1> tag that clearly states the page’s topic."
            example = "(no H1 found)"
        elif len(h1_tags) > 1:
            rec = "Remove extra <h1> tags so there is only one main heading."
            example = "; ".join([h.get_text(strip=True) for h in h1_tags])
        else:
            rec = "/"
            example = h1_tags[0].get_text(strip=True)
        metrics["H1 Structure"] = {"recommendation": rec, "example": example}

        # Primary KW in H1
        h1_text = h1_tags[0].get_text(strip=True) if h1_tags else ""
        has_primary_in_h1 = analyze_primary(client, primary, h1_text)
        if primary and not has_primary_in_h1:
            rec = f"Include your primary keyword (“{primary}”) in the <h1> tag to signal relevance."
            example = h1_text if h1_text else ""
        else:
            rec = "/"
            example = h1_text if h1_text else ""
        metrics["Primary KW in H1"] = {"recommendation": rec, "example": example}

        # 4) H3 Presence & H5/H6 Depth
        no_h3 = analyze_h3(content_soup)
        if no_h3 and content_soup.find("h2"):
            rec = "Add at least one <h3> subsection under each <h2> to improve hierarchy."
            example = "(no H3 tags found)"
        else:
            rec = "/"
            example = "(H3 present)" if not no_h3 else ""
        metrics["H3 Presence"] = {"recommendation": rec, "example": example}

        h5h6_flag = analyze_h5_and_h6(content_soup)
        if h5h6_flag:
            found = [h.get_text(strip=True) for h in content_soup.find_all(["h5","h6"])]
            rec = "Remove <h5> and <h6> tags; stop heading depth at <h4>."
            example = "; ".join(found)
        else:
            rec = "/"
            example = ""
        metrics["H5/H6 Depth"] = {"recommendation": rec, "example": example}

        # 5) Paragraph Length & Bullet Lists
        para_issues = analyze_paragraphs(content_soup)
        if para_issues:
            first_para = next(iter(para_issues))
            rec = (
                "Break long paragraphs into 2–3 sentences each for readability. "
                f"For example, the paragraph starting “{first_para[:100]}…” could be split."
            )
            example = first_para[:100] + "…"
        else:
            rec = "/"
            example = ""
        metrics["Paragraph Length"] = {"recommendation": rec, "example": example}

        no_lists = analyze_bullet_lists(content_soup)
        if no_lists:
            rec = "Add a bullet or numbered list where appropriate to improve scannability."
            example = "(no <ul> or <ol> tags found)"
        else:
            rec = "/"
            example = ""
        metrics["Bullet List Presence"] = {"recommendation": rec, "example": example}

        # 6) Internal Links
        too_few_links, too_long_anchors, long_anchor_example = analyze_internal_links(url)
        if too_few_links:
            rec = "Add at least 5 internal links to relevant pages for better navigation."
            example = "(found fewer than 5 valid internal links)"
        else:
            rec = "/"
            example = ""
        metrics["Internal Links Count"] = {"recommendation": rec, "example": example}

        if too_long_anchors:
            rec = (
                "Shorten link anchor text to 6 words or fewer. "
                f"For example, the anchor “{long_anchor_example}” is too long."
            )
            example = long_anchor_example
        else:
            rec = "/"
            example = ""
        metrics["Internal Link Anchor Length"] = {"recommendation": rec, "example": example}

        # 7) Bold Text / Emphasis
        too_few_bold = count_bold_text(content_soup)
        if too_few_bold:
            total_bold = len(list_bold_text(content_soup))
            rec = (
                "Bold at least 8 phrases to improve scannability. "
                f"Currently only {total_bold} phrases are bolded."
            )
            example = f"(found {total_bold} bolded phrases)"
        else:
            rec = "/"
            example = ""
        metrics["Bold Text Count"] = {"recommendation": rec, "example": example}

        too_long_bold = bold_words(content_soup)
        if too_long_bold:
            for txt in list_bold_text(content_soup):
                if len(txt.split()) > 7:
                    long_bold_example = txt
                    break
            rec = (
                "Shorten lengthy bolded phrases to 7 words or fewer. "
                f"For example: “{long_bold_example}”."
            )
            example = long_bold_example
        else:
            rec = "/"
            example = ""
        metrics["Bold Sequence Length"] = {"recommendation": rec, "example": example}

        # 8) Secondary Keywords in Content
        has_secondaries = analyze_sec(client, secondaries, full_text)
        if secondaries and not has_secondaries:
            rec = (
                "Include your secondary keywords somewhere in the main content. "
                f"Current secondaries: {secondaries}."
            )
            example = ", ".join(secondaries)
        else:
            rec = "/"
            example = ""
        metrics["Secondary KWs in Content"] = {"recommendation": rec, "example": example}

        # 9) Images & Alt Text
        non_logo_alts = show_images_text(content_soup)
        total_non_logo_imgs = len(non_logo_alts)
        word_count = len(full_text.split())

        images_all = [
            img
            for img in content_soup.find_all('img')
            if img.get('alt') is not None and 'logo' not in img.get('alt').lower()
        ]
        no_img = len(images_all)


        if (word_count > 1500 and no_img < 3) or (400 <= word_count <= 1000 and no_img < 2):
            needed = 3 if word_count > 1500 else 2
            rec = (
                f"At {word_count} words but only {no_img} images, add {needed - no_img} more images. "
                "For instance: a chart of key data (alt: “Key data chart”) and a photo illustrating the topic."
            )
            example = f"({no_img} images found)"
        else:
            rec = "/"
            example = ""
        metrics["Images & Word Count Ratio"] = {"recommendation": rec, "example": example}

        # missing_alts = [img.get("src") for img in content_soup.find_all("img") if not img.get("alt")]
        # if missing_alts:
        #     rec = (
        #         f"{len(missing_alts)} image(s) lack alt text, which hurts accessibility. "
        #         f"Add alt attributes like “Description of image” for each."
        #     )
        #     example = missing_alts[0] if missing_alts else ""
        # else:
        #     rec = "/"
        #     example = ""
        # metrics["Image Alt Text Presence"] = {"recommendation": rec, "example": example}

        # missing_alts = [
        #     img.get("src")
        #     for img in content_soup.find_all("img")
        #     # 1) Skip images whose alt contains “logo”
        #     # 2) Then catch any image with no alt (or alt="")
        #     if ('logo' not in (img.get("alt") or "").lower())
        #     and not img.get("alt")
        # ]
        # if missing_alts:
        #     rec = (
        #         f"{len(missing_alts)} image(s) lack alt text, which hurts accessibility. "
        #         f"Add alt attributes like “Description of image” for each."
        #     )
        #     example = missing_alts[0]
        # else:
        #     rec = "/"
        #     example = ""

        # metrics["Image Alt Text Presence"] = {"recommendation": rec, "example": example}

        has_empty_alt = False
        first_empty_alt_img = None
        empty_alt_count = 0

        # Iterate to find the first image with empty alt text
        for img in images_all:
            if img.get('alt').strip() == '':
                has_empty_alt = True
                first_empty_alt_img = img
                empty_alt_count += 1

        if has_empty_alt:
            rec = (
                f"{empty_alt_count} image(s) lack alt text, which hurts accessibility. "
                f"Add alt attributes like “Description of image” for each."
            )
            example = first_empty_alt_img
        else:
            rec = "/"
            example = ""

        metrics["Image Alt Text Presence"] = {"recommendation": rec, "example": example}



        if primary:
            has_primary_in_alt = analyze_primary(client, primary, " ".join(non_logo_alts))
            if not has_primary_in_alt:
                rec = (
                    "Include your primary keyword in at least one image’s alt text. "
                    f"Current alts: {non_logo_alts if non_logo_alts else '(none)'}."
                )
                example = non_logo_alts[0] if non_logo_alts else ""
            else:
                rec = "/"
                example = ""
        else:
            rec = "/"
            example = ""
        metrics["Primary KW in Image Alts"] = {"recommendation": rec, "example": example}

        results[url] = metrics

    return results


# -----------------------------------------------------------------------------
# SECTION 4: NEW FUNCTION TO GET A CONVERSATIONAL TIP FOR A SINGLE ISSUE
# -----------------------------------------------------------------------------

def get_conversational_tip(client: openai.Client, metric_name: str, issue_text: str, current_text: str, kw_list: list[str] | None = None) -> str:

    prompt = (f"You are an SEO consultant, specialized in clear communication and practical solutions. A page owner sees this raw issue for '{metric_name}':\n\n"
        f"    {issue_text} because this is the current vesrsion of the metric: {current_text} \n\n")
    
    # If this is a keyword‐related metric, tell the AI exactly which keyword(s) it should use:
    if kw_list:
        kw_text = ", ".join(kw_list)
        prompt += (f"If there are keyword related issues, here are the keywords to include: {kw_text}")
   
    # Build a prompt that shows the current content,
    # explains the raw issue, and asks for an actionable fix with examples.
    prompt += (
        "Your task is to transform this technical issue into a straightforward, conversational explanation for the page owner. Clearly articulate what the problem is and, more importantly, **how they can fix it themselves with concrete, actionable steps and examples.**"

        "When explaining and suggesting, ensure you:"
        "-   **Explain the 'Why':** Briefly touch on *why* this problem matters for their website's SEO and user experience."
        "-   **Direct Address:** Speak directly to the page owner (e.g., 'Your page title...', 'You can do this by...')."
        "-   **Actionable Steps:** Provide step-by-step instructions or clear tasks they can perform."
        f"-   **Concrete Examples:** Offer specific, ready-to-use examples for titles, descriptions, content snippets, or structural changes. Make sure that all examples are directly connected to the {current_text} and not some general examples."
        "-   **Quantifiable Advice:** Include numbers, lengths, or counts where relevant (e.g., 'aim for 45-60 characters', 'add 2-3 bullet points')."
        "-   **Focus on 'How-To':** The primary goal is to empower the page owner to take immediate, effective action."

        f"For every {metric_name} write max 7 sentences."
    )


    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly SEO content advisor."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"(Error getting tip: {exc})"


# -----------------------------------------------------------------------------
# SECTION 5: STREAMLIT APP
# -----------------------------------------------------------------------------

# Make sure your OPENAI_API_KEY is set in environment
# === Configuration ===
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=openai.api_key)
# client = openai.Client(api_key=openai.api_key)

def main():
    st.set_page_config(layout="wide")
    st.title("🔍 SEO Content Analyzer with On‐Topic Fix Suggestions")

    st.write("""
    1. Paste a full URL.  
    2. Upload a CSV file with columns: url, primary kw, secundary kw.  
    3. Click **Analyze**.  

    The app will run an SEO analysis and then, for each issue found, fetch a **conversational tip** from OpenAI that:
    - Explains why it matters,
    - Shows the **actual current text** (e.g. current meta title) so the AI can stay on-topic,
    - Gives step-by-step, concrete examples tailored to that content.
    """)

    url_input = st.text_input("📥 Paste your URL here", key="url_input")
    csv_file = st.file_uploader("📑 Upload keywords CSV", type="csv")

    if st.button("📈 Analyze"):
        if not url_input:
            st.error("Please paste a URL before clicking Analyze.")
            return
        if csv_file is None:
            st.error("Please upload a keywords CSV file before clicking Analyze.")
            return

        try:
            keywords_dict = parse_keywords_csv(csv_file)
        except Exception as e:
            st.error(f"Could not parse CSV file: {e}")
            return

        if url_input not in keywords_dict:
            st.error("The pasted URL was not found in your CSV. Please ensure it is included.")
            return
        
        primary_kw    = keywords_dict[url_input]["primary_kw"]       # a single string
        secondary_kws = keywords_dict[url_input]["secondary_kw"]     # a list of strings

        with st.spinner("Fetching page and running SEO analysis…"):
            analysis_results = analyze_kws_from_csv([url_input], keywords_dict, client)

        single_result = analysis_results.get(url_input, {})
        if "error" in single_result:
            st.error(single_result["error"])
            return

        # Now, for each metric whose recommendation != "/", call OpenAI to get a conversational tip.
        st.subheader("💬 Smart Fix Suggestions (Examples Stay On‐Topic)")
        any_issue = False

        for metric, data in single_result.items():
            if metric == "_lang":
                continue  # skip language code

            raw_rec = data.get("recommendation", "")
            example = data.get("example", "")  # this is the current text, e.g. current meta title/description

            if raw_rec and raw_rec != "/":
                any_issue = True
                st.markdown(f"**{metric}:** {raw_rec}")
                if example:
                    st.markdown(f"_Current content:_ “{example}”")
                else:
                    st.markdown("_Current content: (none found)_")
                print("Example: ")
                print(example)

                if "Primary KW" in metric:
                    kw_list = [primary_kw] if primary_kw else None
                elif "Secondary KWs" in metric:
                    kw_list = secondary_kws if secondary_kws else None
                else:
                    kw_list = None

                # Fetch a conversational tip, passing along the actual current example
                with st.spinner(f"Generating tip for '{metric}'…"):
                    tip = get_conversational_tip(client, metric, raw_rec, example, kw_list)
                st.markdown(f"> {tip}")
                st.markdown("---")

        if not any_issue:
            st.success("🎉 No actionable issues found! Your page meets the main SEO criteria we checked.")

if __name__ == "__main__":
    main()
