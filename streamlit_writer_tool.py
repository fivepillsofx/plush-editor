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

nltk.download('punkt')
nltk.download('stopwords')

FILLER_WORDS = ["just", "really", "very", "that", "actually", "like", "maybe", "somewhat", "perhaps", "quite"]

CLICHES = [
    "needle in a haystack", "cold sweat", "chill ran down", "time stood still", "dead silence",
    "at the end of the day", "low-hanging fruit", "the calm before the storm", "head over heels",
    "in the nick of time", "plenty of fish in the sea", "easy as pie", "scared stiff",
    "raining cats and dogs", "think outside the box", "every cloud has a silver lining",
    "pushing up daisies", "barking up the wrong tree", "blood ran cold", "fit as a fiddle"
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

def clean_text(text):
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("--", "—")
    text = ' '.join(text.split())
    return text

def suggest_improvements(text):
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

def detect_passive_voice(text):
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    passive_sentences = []
    pattern = re.compile(r'\b(was|were|is being|are being|has been|have been|had been)\b\s+\w+ed\b', re.IGNORECASE)
    for i, sentence in enumerate(sentences):
        if pattern.search(sentence):
            passive_sentences.append((i + 1, sentence.strip()))
    return passive_sentences

def analyze_text(text, style="None"):
    report = []
    if style in STYLE_PRESETS:
        report.append(f"🎨 Style Preset: {style}")
        report.append(f"🔎 Focus: {STYLE_PRESETS[style]['emphasis']}")
        report.append(f"_Note: {STYLE_PRESETS[style]['note']}_")
        report.append("")

    report.append("📊 Analysis Report:")
    report.append(f"• Total words: {len(text.split())}")
    report.append(f"• Total sentences: {textstat.sentence_count(text)}")
    report.append(f"• Avg sentence length: {textstat.words_per_sentence(text):.2f} words")
    report.append(f"• Reading grade level: {textstat.flesch_kincaid_grade(text):.2f}")

    report.append("\n🔎 Common Filler Words Found:")
    for word in FILLER_WORDS:
        count = text.lower().split().count(word)
        if count > 0:
            report.append(f"   - {word}: {count}")

    report.append("\n⚠️ Long Sentences (over 30 words):")
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    for i, sentence in enumerate(sentences):
        word_count = len(wordpunct_tokenize(sentence))
        if word_count > 30:
            report.append(f"\nSentence {i + 1} ({word_count} words):\n{sentence}")

    report.append("\n📈 Top 5 Most Frequent Words (excluding stopwords):")
    words = wordpunct_tokenize(text.lower())
    filtered = [w for w in words if w.isalpha() and w not in stopwords.words('english')]
    most_common = Counter(filtered).most_common(5)
    for word, count in most_common:
        report.append(f"   - {word}: {count}")

    report.append("\n🕵️ Potential Passive Voice Sentences:")
    passive = detect_passive_voice(text)
    if passive:
        for num, sentence in passive:
            report.append(f"\nSentence {num}:\n{sentence}")
    else:
        report.append("   None found ✅")

    report.append("\n🤖 Smart Suggestions:")
    report.append(suggest_improvements(text))

    return "\n".join(report)

def extract_dialogue(text):
    pattern = r'[“"]([^“”"]+)[”"]'
    matches = re.findall(pattern, text)
    return "\n".join(match.strip() for match in matches if match.strip())

def dialogue_by_character(text):
    pattern = r'"[^"]+?"\s+(?:said|asked|replied|whispered|shouted|cried|muttered|yelled|snapped|called)\s+([A-Z][a-zA-Z]*)'
    matches = re.findall(pattern, text)
    if not matches:
        return "No named characters found using direct dialogue attribution."
    counts = Counter(matches)
    result = ["🧍 Dialogue by Character:"]
    for name, count in counts.most_common():
        result.append(f"   - {name}: {count} lines")
    return "\n".join(result)

def find_cliches(text):
    found = []
    text_lower = text.lower()
    for phrase in CLICHES:
        if phrase in text_lower:
            found.append(f"• {phrase}")
    if found:
        return "\n💣 Clichés Found:\n" + "\n".join(found)
    return "✅ No clichés found!"

def load_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def export_full_report(text, style="None", filename="full_report.txt"):
    sections = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections.append(f"🗂 Plush: Full Report\nGenerated: {now}")
    if style in STYLE_PRESETS:
        sections.append(f"🎨 Style Preset: {style}")
        sections.append(f"🔎 Focus: {STYLE_PRESETS[style]['emphasis']}")
        sections.append(f"_Note: {STYLE_PRESETS[style]['note']}_")
    sections.append("=" * 50)
    sections.append(analyze_text(text, style))
    sections.append("\n" + "=" * 50)
    sections.append(dialogue_by_character(text))
    sections.append("\n" + "=" * 50)
    sections.append("🗣 Extracted Dialogue:\n" + extract_dialogue(text))
    sections.append("\n" + "=" * 50)
    sections.append(find_cliches(text))
    sections.append("\n" + "=" * 50)
    sections.append("🖤 Crafted by Bastian — Plush Edition for writtenbybc.com")
    return "\n\n".join(sections)

def main():
    st.set_page_config(page_title="Plush: The Writer Toolkit", layout="centered")
    st.title("🧠 Plush: The Writer Toolkit")
    st.markdown("*Soft on the surface, sharp on the scene.*")

with st.sidebar:
    st.header("🔐 Plush Pro Unlock")
    if not st.session_state.pro_unlocked:
        key = st.text_input("Enter your license key")
        if st.button("Verify Key"):
            if verify_license(key):
                st.session_state.pro_unlocked = True
                st.success("🎉 Plush Pro unlocked!")
            else:
                st.error("❌ Invalid key—please try again or purchase above.")
        st.markdown("[Buy Plush Pro for $19](https://gumroad.com/l/plush-pro)")
    else:
        st.info("✅ Plush Pro features unlocked!")
    style = st.selectbox("🎨 Choose a Writing Style Preset", ["None"] + list(STYLE_PRESETS.keys()))
    uploaded_file = st.file_uploader("📂 Upload a `.txt`, `.docx`, or `.rtf` file", type=["txt", "docx", "rtf"])

    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        raw_text = ""
        if ext == ".txt":
            raw_text = uploaded_file.read().decode("utf-8")
        elif ext == ".docx":
            raw_text = load_docx(uploaded_file)
        elif ext == ".rtf":
            raw_text = rtf_to_text(uploaded_file.read().decode("utf-8"))

        cleaned_text = clean_text(raw_text)

        if st.button("🧼 Clean Only"):
            st.subheader("✅ Cleaned Text")
            st.text_area("Cleaned Output", cleaned_text, height=400)
            st.download_button("⬇️ Download Cleaned File", cleaned_text, file_name="cleaned_output.txt")

        if st.button("🔍 Analyze Only"):
            st.subheader("📊 Analysis Report + 🤖 Smart Suggestions")
            report = analyze_text(raw_text, style)
            st.text_area("Report", report, height=800)
            st.download_button("⬇️ Download Report", report, file_name="analysis_report.txt")

        if st.button("🧼 + 🔍 Clean & Analyze"):
            st.subheader("✅ Cleaned Text")
            st.text_area("Cleaned Output", cleaned_text, height=400)
            st.download_button("⬇️ Download Cleaned File", cleaned_text, file_name="cleaned_output.txt")

            st.subheader("📊 Analysis Report + 🤖 Smart Suggestions")
            report = analyze_text(cleaned_text, style)
            st.text_area("Report", report, height=800)
            st.download_button("⬇️ Download Report", report, file_name="analysis_report.txt")

        if st.button("🗣 Extract Dialogue Only"):
            st.subheader("🗣 Dialogue Only")
            dialogue = extract_dialogue(raw_text)
            if dialogue:
                st.text_area("Extracted Dialogue", dialogue, height=600)
                st.download_button("⬇️ Download Dialogue", dialogue, file_name="dialogue_only.txt")
            else:
                st.warning("No dialogue found in this manuscript.")

        if st.button("🧍 Dialogue by Character"):
            st.subheader("🧍 Dialogue by Character")
            character_report = dialogue_by_character(raw_text)
            st.text_area("Speaker Breakdown", character_report, height=400)
            st.download_button("⬇️ Download Character Dialogue Report", character_report, file_name="character_dialogue.txt")

        if st.button("💣 Cliché Buster"):
            st.subheader("💣 Cliché Buster")
            cliche_report = find_cliches(raw_text)
            st.text_area("Cliché Report", cliche_report, height=400)
            st.download_button("⬇️ Download Cliché Report", cliche_report, file_name="cliches_found.txt")

        if st.button("📦 Export Full Report"):
            st.subheader("📦 Full Report + 🤖 Suggestions")
            full_report = export_full_report(raw_text, style)
            st.text_area("Full Report Preview", full_report, height=800)
            st.download_button("⬇️ Download Full Report", full_report, file_name="plush_full_report.txt")

    st.markdown("---")
    st.markdown("🖤 *Crafted by Bastian — Plush Edition for* [**writtenbybc.com**](https://www.writtenbybc.com)")

if __name__ == "__main__":
    main()
