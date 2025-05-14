import os
import re
import streamlit as st
from collections import Counter
from datetime import datetime
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
import textstat
from docx import Document
from striprtf.striprtf import rtf_to_text
import nltk
import requests

# ── LICENSE / PRO CHECK SETUP ────────────────────────────────
# These pull from Streamlit Community Cloud Secrets, or from .streamlit/secrets.toml locally
ACCESS_TOKEN = st.secrets["GUMROAD_ACCESS_TOKEN"]
PERMALINK    = st.secrets["GUMROAD_PRODUCT_PERMALINK"]

if "pro_unlocked" not in st.session_state:
    st.session_state.pro_unlocked = False

def verify_license(license_key: str) -> bool:
    """Verify a Gumroad license key via their API."""
    resp = requests.post(
        "https://api.gumroad.com/v2/licenses/verify",
        data={
            "product_permalink": PERMALINK,
            "license_key": license_key,
            "access_token": ACCESS_TOKEN
        },
        timeout=10
    ).json()
    return resp.get("success", False)

# ── NLTK SETUP ───────────────────────────────────────────────
nltk.download('punkt')
nltk.download('stopwords')

# ── CONSTANTS & PRESETS ─────────────────────────────────────
FILLER_WORDS = ["just", "really", "very", "that", "actually",
                "like", "maybe", "somewhat", "perhaps", "quite"]

CLICHES = [
    "needle in a haystack", "cold sweat", "chill ran down",
    "time stood still", "dead silence", "at the end of the day",
    "low-hanging fruit", "the calm before the storm", "head over heels",
    "in the nick of time", "plenty of fish in the sea", "easy as pie",
    "scared stiff", "raining cats and dogs", "think outside the box",
    "every cloud has a silver lining", "pushing up daisies",
    "barking up the wrong tree", "blood ran cold", "fit as a fiddle"
]

STYLE_PRESETS = {
    "Gritty": {
        "emphasis": "Cliché detection, passive voice, long sentences",
        "note": "Highlights harshness and realism — cracks down on clichés and overwritten prose."
    },
    "Snappy": {
        "emphasis": "Filler words, punchy structure, dialogue ratio",
        "note": "Focuses on rhythm, minimalism, and active scenes."
    },
    "Poetic": {
        "emphasis": "Flow, sentence variety, rhythm",
        "note": "Tolerates longer prose, flags broken rhythm or repetition."
    },
    "Technical": {
        "emphasis": "Passive voice, clarity, redundancy",
        "note": "Suited for nonfiction or precise procedural tone."
    },
    "Sparse": {
        "emphasis": "Minimal filler, clarity, clean structure",
        "note": "Zero fluff. Cuts verbosity. Favors clean delivery."
    }
}

# ── CORE FUNCTIONS ───────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("--", "—")
    return ' '.join(text.split())

def suggest_improvements(text: str) -> str:
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    suggestions = []
    for i, sentence in enumerate(sentences):
        issues = []
        if len(wordpunct_tokenize(sentence)) > 30:
            issues.append("⚠️ Consider breaking this long sentence into two or more.")
        if re.search(r'\b(was|were|is being|are being|has been|have been|had been)\b\s+\w+ed\b', sentence):
            issues.append("💡 Try rephrasing in active voice.")
        filler_count = sum(sentence.lower().count(fw) for fw in FILLER_WORDS)
        if filler_count > 2:
            issues.append("✂️ This line may be padded with filler words.")
        if issues:
            suggestions.append(f"\nSentence {i+1}:\n“{sentence.strip()}”\n" + "\n".join(issues))
    return "\n".join(suggestions) if suggestions else "✅ No smart suggestions needed — looking solid!"

def detect_passive_voice(text: str):
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    passive = []
    pattern = re.compile(r'\b(was|were|is being|are being|has been|have been|had been)\b\s+\w+ed\b', re.IGNORECASE)
    for i, sentence in enumerate(sentences):
        if pattern.search(sentence):
            passive.append((i+1, sentence.strip()))
    return passive

def analyze_text(text: str, style="None") -> str:
    report = []
    if style in STYLE_PRESETS:
        report += [
            f"🎨 Style Preset: {style}",
            f"🔎 Focus: {STYLE_PRESETS[style]['emphasis']}",
            f"_Note: {STYLE_PRESETS[style]['note']}_", ""
        ]
    report += [
        "📊 Analysis Report:",
        f"• Total words: {len(text.split())}",
        f"• Total sentences: {textstat.sentence_count(text)}",
        f"• Avg sentence length: {textstat.words_per_sentence(text):.2f} words",
        f"• Reading grade level: {textstat.flesch_kincaid_grade(text):.2f}", ""
    ]
    report.append("🔎 Common Filler Words Found:")
    for w in FILLER_WORDS:
        c = text.lower().split().count(w)
        if c > 0:
            report.append(f"   - {w}: {c}")
    report.append("\n⚠️ Long Sentences (over 30 words):")
    sentences = PunktSentenceTokenizer().tokenize(text)
    for i, s in enumerate(sentences):
        wc = len(wordpunct_tokenize(s))
        if wc > 30:
            report.append(f"\nSentence {i+1} ({wc} words):\n{s}")
    report.append("\n📈 Top 5 Most Frequent Words (excl. stopwords):")
    words = [w for w in wordpunct_tokenize(text.lower())
             if w.isalpha() and w not in stopwords.words('english')]
    for w, c in Counter(words).most_common(5):
        report.append(f"   - {w}: {c}")
    report.append("\n🕵️ Potential Passive Voice Sentences:")
    passive = detect_passive_voice(text)
    report += [f"\nSentence {n}:\n{s}" for n, s in passive] if passive else ["   None found ✅"]
    report += ["\n🤖 Smart Suggestions:", suggest_improvements(text)]
    return "\n".join(report)

def extract_dialogue(text: str) -> str:
    matches = re.findall(r'[“"]([^“”"]+)[”"]', text)
    return "\n".join(m.strip() for m in matches if m.strip())

def dialogue_by_character(text: str) -> str:
    matches = re.findall(
        r'"[^"]+?"\s+(?:said|asked|replied|whispered|shouted|cried|muttered|yelled|snapped|called)\s+([A-Z][a-zA-Z]*)',
        text)
    if not matches:
        return "No named characters found."
    counts = Counter(matches).most_common()
    return "🧍 Dialogue by Character:\n" + "\n".join(f"   - {n}: {c} lines" for n, c in counts)

def find_cliches(text: str) -> str:
    found = [f"• {ph}" for ph in CLICHES if ph in text.lower()]
    return "\n💣 Clichés Found:\n" + "\n".join(found) if found else "✅ No clichés found!"

def load_docx(file) -> str:
    return "\n".join(p.text for p in Document(file).paragraphs)

def export_full_report(text: str, style="None") -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        f"🗂 Plush: Full Report\nGenerated: {now}",
        analyze_text(text, style), "="*50,
        dialogue_by_character(text), "="*50,
        "🗣 Extracted Dialogue:\n" + extract_dialogue(text), "="*50,
        find_cliches(text), "="*50,
        "🖤 Crafted by Bastian — Plush Edition for writtenbybc.com"
    ]
    return "\n\n".join(parts)

# ── STREAMLIT APP LAYOUT ────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Plush: The Writer Toolkit",
        page_icon="plush-favicon.png",  # your crocodile icon file in repo
        layout="centered"
    )
    st.title("🧠 Plush: The Writer Toolkit")
    st.markdown("*Tribute to Bodhi Crocodile — soft on the surface, sharp on the scene.*")

    # ── PRO UNLOCK SIDEBAR ─────────────────────────────
    with st.sidebar:
        st.header("🔐 Plush Pro Unlock")
        if not st.session_state.pro_unlocked:
            key = st.text_input("Enter your license key")
            if st.button("Verify Key"):
                if verify_license(key):
                    st.session_state.pro_unlocked = True
                    st.success("🎉 Plush Pro unlocked!")
                else:
                    st.error("❌ Invalid key—try again or purchase above.")
            st.markdown("[Buy Plush Pro for $19](https://bradleyc51.gumroad.com/l/ccvuo)")
        else:
            st.info("✅ Plush Pro features unlocked!")

    # ── STYLE PRESET ────────────────────────────────────
    if st.session_state.pro_unlocked:
        style = st.selectbox(
            "🎨 Choose a Writing Style Preset",
            ["None"] + list(STYLE_PRESETS.keys())
        )
    else:
        st.markdown("### 🎨 Writing Style Presets (Plush Pro only)")
        st.info("🔒 Unlock Plush Pro to access custom style presets.")
        style = "None"

    # ── FILE UPLOADER ────────────────────────────────────
    if st.session_state.pro_unlocked:
        uploaded_files = st.file_uploader(
            "📂 Upload one or more `.txt`, `.docx`, or `.rtf` files",
            type=["txt", "docx", "rtf"],
            accept_multiple_files=True
        )
    else:
        uploaded_file = st.file_uploader(
            "📂 Upload a single `.txt`, `.docx`, or `.rtf` file",
            type=["txt", "docx", "rtf"]
        )
        st.info("🔒 Upgrade to Plush Pro to analyze entire manuscripts at once.")
        uploaded_files = [uploaded_file] if uploaded_file else []

    # ── PROCESS UPLOADED CONTENT ─────────────────────────
    raw_text = ""
    for uf in uploaded_files:
        if uf is None:
            continue
        ext = os.path.splitext(uf.name)[1].lower()
        if ext == ".txt":
            raw_text += uf.read().decode("utf-8") + "\n"
        elif ext == ".docx":
            raw_text += load_docx(uf) + "\n"
        elif ext == ".rtf":
            raw_text += rtf_to_text(uf.read().decode("utf-8")) + "\n"

    if raw_text:
        cleaned = clean_text(raw_text)

        # ── CLEAN ONLY ───────────────────────────────────
        if st.button("🧼 Clean Only"):
            st.subheader("✅ Cleaned Text")
            st.text_area("Cleaned Output", cleaned, height=400)
            st.download_button("⬇️ Download Cleaned File",
                               cleaned, file_name="cleaned_output.txt")

        # ── ANALYZE ONLY ─────────────────────────────────
        if st.button("🔍 Analyze Only"):
            st.subheader("📊 Analysis Report + 🤖 Smart Suggestions")
            report = analyze_text(raw_text, style)
            st.text_area("Report", report, height=800)
            st.download_button("⬇️ Download Report",
                               report, file_name="analysis_report.txt")

        # ── CLEAN & ANALYZE ───────────────────────────────
        if st.button("🧼 + 🔍 Clean & Analyze"):
            st.subheader("✅ Cleaned Text")
            st.text_area("Cleaned Output", cleaned, height=400)
            st.download_button("⬇️ Download Cleaned File",
                               cleaned, file_name="cleaned_output.txt")
            st.subheader("📊 Analysis Report + 🤖 Smart Suggestions")
            report = analyze_text(cleaned, style)
            st.text_area("Report", report, height=800)
            st.download_button("⬇️ Download Report",
                               report, file_name="analysis_report.txt")

        # ── DIALOGUE ONLY ───────────────────────────────
        if st.button("🗣 Extract Dialogue Only"):
            st.subheader("🗣 Dialogue Only")
            dialogue = extract_dialogue(raw_text)
            if dialogue:
                st.text_area("Extracted Dialogue", dialogue, height=600)
                st.download_button("⬇️ Download Dialogue",
                                   dialogue, file_name="dialogue_only.txt")
            else:
                st.warning("No dialogue found in this manuscript.")

        # ── DIALOGUE BY CHARACTER ─────────────────────────
        if st.button("🧍 Dialogue by Character"):
            st.subheader("🧍 Dialogue by Character")
            char_report = dialogue_by_character(raw_text)
            st.text_area("Speaker Breakdown", char_report, height=400)
            st.download_button("⬇️ Download Character Dialogue Report",
                               char_report, file_name="character_dialogue.txt")

        # ── CLICHÉ BUSTER ────────────────────────────────
        if st.button("💣 Cliché Buster"):
            st.subheader("💣 Cliché Buster")
            cpl = find_cliches(raw_text)
            st.text_area("Cliché Report", cpl, height=400)
            st.download_button("⬇️ Download Cliché Report",
                               cpl, file_name="cliches_found.txt")

        # ── EXPORT FULL REPORT ────────────────────────────
        if st.session_state.pro_unlocked:
            if st.button("📦 Export Full Report"):
                full = export_full_report(raw_text, style)
                st.text_area("Full Report Preview", full, height=800)
                st.download_button("⬇️ Download Full Report",
                                   full, file_name="plush_full_report.txt")
        else:
            st.info("🔒 Unlock Plush Pro to export polished full reports!")

    st.markdown("---")
    st.markdown("🖤 *Crafted by Bastian — Plush Edition for* [**writtenbybc.com**](https://www.writtenbybc.com)")

if __name__ == "__main__":
    main()
