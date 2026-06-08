import streamlit as st
import re
import json
import os
import time
import tempfile
from datetime import datetime
from collections import Counter

from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import db

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ✅ Fixed: corrected model names (gemini-3.x doesn't exist)
_CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

def _get_pkg_major() -> int:
    try:
        import importlib.metadata
        ver = importlib.metadata.version("langchain-google-genai")
        return int(ver.split(".")[0])
    except Exception:
        return 0

_PKG_MAJOR   = _get_pkg_major()
_EMBED_MODEL = "gemini-embedding-001" if _PKG_MAJOR >= 4 else "models/embedding-001"

# ─────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────
st.set_page_config(page_title="Meeting Intelligence Agent", page_icon="📋", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
section[data-testid="stSidebar"] { display: none !important; }

.main-title { font-family:'DM Serif Display',serif; font-size:2.2rem; color:#1a1a2e; letter-spacing:-0.5px; margin-bottom:0; }
.sub-title  { color:#6b7280; font-size:0.9rem; margin-top:3px; margin-bottom:1.2rem; }

.stat-box   { background:linear-gradient(135deg,#667eea18,#764ba218); border:1px solid #667eea28;
              border-radius:10px; padding:0.75rem 1rem; text-align:center; }
.stat-num   { font-size:1.55rem; font-weight:700; color:#4f46e5; line-height:1; }
.stat-label { font-size:0.7rem; color:#6b7280; text-transform:uppercase; letter-spacing:.06em; margin-top:3px; }

.an-card  { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:1.1rem 1.3rem; margin-bottom:.8rem; }
.an-title { font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:#94a3b8; margin-bottom:.6rem; }
.an-big   { font-size:2rem; font-weight:700; color:#1a1a2e; }

.bar-row   { display:flex; align-items:center; gap:8px; margin-bottom:5px; font-size:.8rem; }
.bar-label { width:120px; color:#374151; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bar-track { flex:1; background:#f1f5f9; border-radius:4px; height:10px; }
.bar-fill  { height:10px; border-radius:4px; }
.bar-val   { width:30px; color:#6b7280; text-align:right; font-size:.75rem; }

.donut-wrap { display:flex; gap:10px; flex-wrap:wrap; margin-top:.4rem; }
.donut-chip { background:#ede9fe; color:#5b21b6; border-radius:20px; padding:3px 12px; font-size:.78rem; font-weight:500; }

.tl-item { display:flex; gap:12px; margin-bottom:10px; align-items:flex-start; }
.tl-dot  { width:10px; height:10px; border-radius:50%; margin-top:4px; flex-shrink:0; }
.tl-text { font-size:.82rem; color:#374151; line-height:1.45; }
.tl-time { font-size:.7rem; color:#9ca3af; margin-top:1px; }

.priority-high   { color:#dc2626; font-weight:600; }
.priority-medium { color:#d97706; font-weight:600; }
.priority-low    { color:#16a34a; font-weight:600; }

.sentiment-pos { background:#dcfce7; color:#166534; border-radius:4px; padding:1px 6px; font-size:.75rem; }
.sentiment-neg { background:#fee2e2; color:#991b1b; border-radius:4px; padding:1px 6px; font-size:.75rem; }
.sentiment-neu { background:#f1f5f9; color:#475569; border-radius:4px; padding:1px 6px; font-size:.75rem; }

.sh { font-family:'DM Serif Display',serif; font-size:1.05rem; color:#1a1a2e;
      border-bottom:2px solid #e2e8f0; padding-bottom:5px; margin:1rem 0 .5rem; }

.db-ok  { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px;
          padding:.5rem .9rem; font-size:.82rem; color:#166534; }
.db-err { background:#fef2f2; border:1px solid #fecaca; border-radius:8px;
          padding:.5rem .9rem; font-size:.82rem; color:#991b1b; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📋 Meeting Intelligence Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Gemini RAG · MySQL Storage · Multi-model fallback · Analytics Dashboard</p>', unsafe_allow_html=True)

# ── Init DB once per session ──
if "db_status" not in st.session_state:
    ok, msg = db.init_db()
    st.session_state["db_status"] = (ok, msg)

_db_ok, _db_msg = st.session_state["db_status"]

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────
with st.expander("⚙️ Settings & Filters", expanded=False):
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)

    with fc1:
        st.markdown("**📝 Metadata**")
        meeting_title = st.text_input("Title",     placeholder="Q2 Sprint Planning", label_visibility="collapsed")
        meeting_date  = st.date_input("Date",      value=datetime.today(),           label_visibility="collapsed")
        attendees_raw = st.text_input("Attendees", placeholder="Alice, Bob, Carol",  label_visibility="collapsed")

    with fc2:
        st.markdown("**🎚️ Output**")
        output_language = st.selectbox("Language",
            ["English","Hindi","Spanish","French","German","Portuguese","Japanese","Arabic"],
            label_visibility="collapsed")
        summary_length = st.select_slider("Summary",
            options=["1–2 sentences","3–5 sentences","Full paragraph"],
            value="3–5 sentences", label_visibility="collapsed")
        tone = st.selectbox("Tone",
            ["Formal (corporate)","Concise (bullet-heavy)","Casual (plain English)","Technical (engineering)"],
            label_visibility="collapsed")

    with fc3:
        st.markdown("**📦 Sections**")
        inc_summary   = st.checkbox("Summary",      value=True)
        inc_topics    = st.checkbox("Topics",       value=True)
        inc_keypoints = st.checkbox("Key Points",   value=True)
        inc_decisions = st.checkbox("Decisions",    value=True)
        inc_actions   = st.checkbox("Action Items", value=True)
        inc_nextsteps = st.checkbox("Next Steps",   value=True)
        inc_speakers  = st.checkbox("Speaker map",  value=True)
        inc_sentiment = st.checkbox("Sentiment",    value=True)

    with fc4:
        st.markdown("**🔬 Processing**")
        min_priority  = st.selectbox("Min priority", ["Low (all)","Medium","High only"], label_visibility="collapsed")
        chunk_size    = st.slider("Chunk size",    500, 2000, 1000, 100)
        chunk_overlap = st.slider("Overlap",        50,  400,  150,  50)
        retriever_k   = st.slider("Retriever k",     3,   10,    6)
        dedup         = st.checkbox("Deduplicate key points", value=True)

    with fc5:
        st.markdown("**🗄️ MySQL**")
        use_mysql     = st.checkbox("Save to MySQL",                   value=_db_ok)
        use_past_ctx  = st.checkbox("Use past transcripts as context", value=_db_ok)
        past_ctx_limit= st.slider("Past transcripts to load", 1, 10, 3)
        if _db_ok:
            st.markdown(f'<div class="db-ok">✅ Connected · {db.count_transcripts()} stored</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="db-err">❌ {_db_msg}</div>', unsafe_allow_html=True)
            st.caption("Set MySQL env vars in Railway dashboard (Variables tab)")

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
PRIORITY_MAP     = {"high": 0, "medium": 1, "low": 2}
MIN_PRIORITY_MAP = {"Low (all)": 2, "Medium": 1, "High only": 0}

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "to","of","in","for","on","with","at","by","from","as","it","its","this",
    "that","these","those","and","or","but","not","we","i","you","they","he","she",
    "our","their","your","my","his","her","all","more","also","about","up","out",
    "so","if","then","there","than","into","over","just","some","what","how",
}

def _filter_by_priority(items, threshold_label):
    threshold = MIN_PRIORITY_MAP[threshold_label]
    out = []
    for item in items:
        m = re.search(r"\[(HIGH|MEDIUM|LOW)\]", item, re.IGNORECASE)
        if m:
            if PRIORITY_MAP.get(m.group(1).lower(), 2) <= threshold:
                out.append(item)
        else:
            out.append(item)
    return out

def _deduplicate(items, threshold=0.6):
    seen, result = [], []
    for item in items:
        words  = set(item.lower().split())
        is_dup = any(
            len(words & set(s.lower().split())) / max(len(words), 1) > threshold
            for s in seen
        )
        if not is_dup:
            seen.append(item)
            result.append(item)
    return result

def _extract_speakers(text):
    patterns = [
        r"^([A-Z][a-zA-Z\s]{1,25}):\s",
        r"\[([A-Z][a-zA-Z\s]{1,25})\]",
        r"^([A-Z][A-Z\s]{1,20})\s*[-–—]\s",
    ]
    speakers = set()
    for line in text.split("\n"):
        for pat in patterns:
            m = re.match(pat, line.strip())
            if m:
                name = m.group(1).strip().title()
                if len(name) > 1:
                    speakers.add(name)
    return sorted(speakers)

def _word_count(text):     return len(text.split())
def _reading_time(words):  return f"{max(1, round(words / 200))} min"
def _sentence_count(text): return max(1, len(re.split(r'[.!?]+', text)))

def _avg_sentence_len(text):
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return round(sum(len(s.split()) for s in sents) / max(len(sents), 1), 1)

def _top_keywords(text, n=10):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq  = Counter(w for w in words if w not in STOPWORDS)
    return freq.most_common(n)

def _count_questions(text):
    return len(re.findall(r'\?', text))

def _speaker_word_counts(text, speakers):
    counts  = {s: 0 for s in speakers}
    current = None
    for line in text.split("\n"):
        for sp in speakers:
            if re.match(rf"^{re.escape(sp)}\s*:", line, re.IGNORECASE):
                current = sp
                break
        if current:
            counts[current] += len(line.split())
    return {k: v for k, v in counts.items() if v > 0}

def _sentiment_distribution(key_points):
    dist = {"positive": 0, "negative": 0, "neutral": 0}
    for kp in key_points:
        m = re.search(r"\[(POSITIVE|NEGATIVE|NEUTRAL)\]", kp, re.IGNORECASE)
        dist[m.group(1).lower() if m else "neutral"] += 1
    return dist

def _priority_distribution(action_items):
    dist = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNTAGGED": 0}
    for item in action_items:
        m = re.search(r"\[(HIGH|MEDIUM|LOW)\]", item, re.IGNORECASE)
        if m:
            dist[m.group(1).upper()] += 1
        else:
            dist["UNTAGGED"] += 1
    return dist

def _html_bar(label, value, max_val, color="#667eea"):
    pct = int(value / max(max_val, 1) * 100)
    return (
        f'<div class="bar-row">'
        f'<span class="bar-label" title="{label}">{label}</span>'
        f'<div class="bar-track">'
        f'<div class="bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
        f'<span class="bar-val">{value}</span>'
        f'</div>'
    )

# ─────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────
@st.cache_data
def clean_transcript(text):
    fillers = [
        r"\bum+\b", r"\buh+\b", r"\bhmm+\b", r"\byeah\b", r"\bbasically\b",
        r"\bactually\b", r"\byou know\b", r"\bokay so\b", r"\bso yeah\b",
        r"\bkind of\b", r"\bsort of\b", r"\blike\b(?!\s+[a-z]+ed\b)",
        r"\bright\b(?=\s*[,?])", r"\bi mean\b",
    ]
    for pat in fillers:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^([A-Z]{2,})\s*:", lambda m: m.group(1).title() + ":", text, flags=re.MULTILINE)
    return text.strip()

def extract_json(raw):
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON:\n{raw}")

# ─────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────
def build_prompt(summary_length, tone, output_language, has_past_context=False):
    lmap = {
        "1–2 sentences": "1-2 sentences only — extremely terse.",
        "3–5 sentences": "3-5 sentences — dense, no fluff.",
        "Full paragraph": "one detailed paragraph capturing all key context.",
    }
    tmap = {
        "Formal (corporate)":    "Use formal, professional corporate language.",
        "Concise (bullet-heavy)":"Use very short clipped phrases. Prefer fragments.",
        "Casual (plain English)":"Use plain conversational English. Avoid jargon.",
        "Technical (engineering)":"Use precise technical language; include metrics, system names, ticket IDs.",
    }
    lang = f"Write ALL output in {output_language}." if output_language != "English" else ""

    past_rule = ""
    past_slot = ""
    if has_past_context:
        past_rule = (
            "- You are also given PAST MEETING CONTEXT from a database of prior meetings. "
            "Use it to identify recurring topics, carry-over action items, and historical patterns. "
            "Do NOT copy-paste — synthesise insights only.\n"
        )
        past_slot = "Past Meeting Context (from database):\n{past_context}\n---\n"

    template = f"""
You are an expert meeting analyst. Produce precise meeting minutes.

RULES:
- {tmap[tone]}
- {lang}
- Summary: {lmap[summary_length]}
- Topics: max 6, each ≤ 6 words.
- Key Points: concrete facts, numbers, names. Append [POSITIVE], [NEGATIVE], or [NEUTRAL].
- Decisions: ONLY firm agreed decisions — not proposals.
- Action Items: "[PRIORITY] Owner: Task (Deadline)". PRIORITY = HIGH/MEDIUM/LOW.
- Next Steps: specific follow-ups with timeline if mentioned.
- Do NOT invent anything not in the context.
{past_rule}- Return ONLY valid JSON — no markdown, no preamble.

Output (strict JSON):
{{{{
  "summary": "string",
  "topics": ["string"],
  "keyPoints": ["string [SENTIMENT]"],
  "decisions": ["string"],
  "actionItems": ["[PRIORITY] Owner: task (deadline)"],
  "nextSteps": ["string"],
  "overallSentiment": "positive|neutral|negative",
  "meetingType": "standup|planning|review|retrospective|brainstorm|decision|other"
}}}}

{past_slot}Current Meeting Context: {{context}}
Task: {{input}}
"""
    return ChatPromptTemplate.from_template(template)

# ─────────────────────────────────────────
# EMBEDDINGS + CHAIN
# ─────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=_EMBED_MODEL, google_api_key=GOOGLE_API_KEY)

def build_chain(chunks, past_chunks, llm, prompt_template, k, has_past_context):
    embeddings = get_embeddings()

    vdb = Chroma.from_texts(texts=chunks, embedding=embeddings)
    ret = vdb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    fmt = lambda docs: "\n\n".join(d.page_content for d in docs)

    if has_past_context and past_chunks:
        past_vdb = Chroma.from_texts(texts=past_chunks, embedding=embeddings)
        past_ret = past_vdb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": min(3, len(past_chunks))}
        )
        fmt_past = lambda docs: "\n\n".join(d.page_content for d in docs)
        chain_input = {
            "context":      ret      | fmt,
            "past_context": past_ret | fmt_past,
            "input":        RunnablePassthrough(),
        }
    else:
        chain_input = {
            "context": ret | fmt,
            "input":   RunnablePassthrough(),
        }

    return chain_input | prompt_template | llm | StrOutputParser()

def invoke_with_fallback(chunks, past_chunks, prompt_text, prompt_template, k, has_past_context):
    last_error = None
    for model_name in _CANDIDATE_MODELS:
        llm   = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY, temperature=0)
        chain = build_chain(chunks, past_chunks, llm, prompt_template, k, has_past_context)
        for attempt in range(1, 4):
            try:
                result = chain.invoke(prompt_text)
                st.session_state["active_model"] = model_name
                return result
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    dm   = re.search(r"retry[^\d]*(\d+)", err, re.IGNORECASE)
                    wait = min(int(dm.group(1)) if dm else attempt * 20, 60)
                    if attempt < 3:
                        st.warning(f"⏳ `{model_name}` quota hit ({attempt}/3) — waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        st.warning(f"⚠️ `{model_name}` exhausted — trying next model...")
                        last_error = e
                        break
                else:
                    raise
    raise RuntimeError(f"All models exhausted. Last error: {last_error}")

# ─────────────────────────────────────────
# MARKDOWN EXPORT
# ─────────────────────────────────────────
def build_markdown(data, meta, sections):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    md  = f"# {meta.get('title', 'Meeting Minutes')}\n_Generated: {now}_\n\n"
    if meta.get("date"):        md += f"**Date:** {meta['date']}  \n"
    if meta.get("attendees"):   md += f"**Attendees:** {meta['attendees']}  \n"
    if data.get("meetingType"): md += f"**Type:** {data['meetingType'].title()}  \n"
    md += "\n"

    def sec(title, items, emoji=""):
        return (f"## {emoji} {title}\n" + "\n".join(f"- {i}" for i in items) + "\n\n") if items else ""

    if sections.get("summary"):   md += f"## 📌 Summary\n{data.get('summary', '')}\n\n"
    if sections.get("topics"):    md += sec("Topics",       data.get("topics", []),      "🏷")
    if sections.get("keypoints"): md += sec("Key Points",   data.get("keyPoints", []),   "💡")
    if sections.get("decisions"): md += sec("Decisions",    data.get("decisions", []),   "✅")
    if sections.get("actions"):   md += sec("Action Items", data.get("actionItems", []), "⚡")
    if sections.get("nextsteps"): md += sec("Next Steps",   data.get("nextSteps", []),   "➡️")
    if sections.get("speakers") and meta.get("speakers"):
        md += "## 🎤 Speakers\n" + "\n".join(f"- {s}" for s in meta["speakers"]) + "\n\n"
    return md.strip()

# ─────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    text_input = st.text_area("Paste transcript", height=200,
                              placeholder="Paste your meeting transcript here...")
with col2:
    uploaded = st.file_uploader("Or upload TXT / PDF", type=["txt", "pdf"])
    st.caption("Plain text or PDF supported")

transcript = ""
if uploaded:
    with st.spinner("Reading file..."):
        if uploaded.name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded.read())
                transcript = extract_text(f.name)
        else:
            transcript = uploaded.read().decode("utf-8", errors="ignore")
    st.success(f"Loaded **{uploaded.name}** ({len(transcript):,} chars)")
elif text_input.strip():
    transcript = text_input.strip()

# ── Pre-processing stats ──
if transcript:
    spk = _extract_speakers(transcript)
    wc  = _word_count(transcript)
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, num, lbl in [
        (s1, f"{wc:,}",                    "Words"),
        (s2, _reading_time(wc),            "Read time"),
        (s3, len(spk),                     "Speakers"),
        (s4, _sentence_count(transcript),  "Sentences"),
        (s5, _count_questions(transcript), "Questions"),
    ]:
        col.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-num">{num}</div>'
            f'<div class="stat-label">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if spk:
        st.caption("🎤 " + "  ·  ".join(f"`{s}`" for s in spk))

# ─────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────
st.markdown("")
if st.button("🚀 Generate Minutes + Analytics", type="primary", disabled=not transcript):

    cleaned  = clean_transcript(transcript)
    speakers = _extract_speakers(transcript)
    wc_clean = _word_count(cleaned)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(cleaned)
    st.info(f"📦 {len(chunks)} chunks · {wc_clean:,} words after cleaning")

    # ── Load past transcripts from MySQL for RAG context ──
    past_chunks = []
    if use_past_ctx and _db_ok:
        with st.spinner("Loading past transcripts from MySQL..."):
            past_rows, err = db.fetch_past_transcripts(limit=past_ctx_limit)
            if err:
                st.warning(f"Could not load past transcripts: {err}")
            elif past_rows:
                for row in past_rows:
                    txt = row.get("cleaned_text") or ""
                    if txt:
                        past_chunks += splitter.split_text(txt)
                st.success(f"📚 Loaded {len(past_rows)} past meeting(s) · {len(past_chunks)} context chunks")

    has_past        = bool(past_chunks)
    prompt_template = build_prompt(summary_length, tone, output_language, has_past_context=has_past)

    with st.spinner("Building vector index..."):
        get_embeddings()

    with st.spinner("Calling Gemini..."):
        try:
            raw  = invoke_with_fallback(
                chunks, past_chunks,
                "Generate meeting minutes",
                prompt_template, retriever_k, has_past,
            )
            data = extract_json(raw)

            # Post-processing
            if dedup and "keyPoints" in data:
                before = len(data["keyPoints"])
                data["keyPoints"] = _deduplicate(data["keyPoints"])
                removed = before - len(data["keyPoints"])
                if removed:
                    st.caption(f"🔁 Removed {removed} duplicate key point(s)")

            if "actionItems" in data:
                data["actionItems"] = _filter_by_priority(data["actionItems"], min_priority)

            data["_speakers"] = speakers
            data["_raw_text"] = transcript

            sections = dict(
                summary=inc_summary, topics=inc_topics, keypoints=inc_keypoints,
                decisions=inc_decisions, actions=inc_actions, nextsteps=inc_nextsteps,
                speakers=inc_speakers,
            )
            meta = dict(
                title=meeting_title or "Meeting Minutes",
                date=str(meeting_date),
                attendees=attendees_raw,
                speakers=speakers,
            )
            st.session_state.update(
                data=data, sections=sections, meta=meta,
                markdown=build_markdown(data, meta, sections),
            )

            active_model = st.session_state.get("active_model", "?")
            st.success(f"✅ Done · model: `{active_model}`")

            # ── Persist to MySQL ──
            if use_mysql and _db_ok:
                tid, err = db.save_transcript(
                    meeting_title, meeting_date, attendees_raw,
                    transcript, cleaned, wc_clean, speakers,
                )
                if err:
                    st.warning(f"⚠️ Transcript save failed: {err}")
                else:
                    ok2, err2 = db.save_minutes(tid, data, active_model)
                    if err2:
                        st.warning(f"⚠️ Minutes save failed: {err2}")
                    else:
                        st.info(f"🗄️ Saved to MySQL · transcript id = {tid}")

        except RuntimeError as e:
            st.error("🚫 All quota limits reached.")
            st.info(str(e))
        except ValueError as e:
            st.error("JSON parse failed.")
            st.exception(e)
        except Exception as e:
            st.error("Unexpected error.")
            st.exception(e)

# ─────────────────────────────────────────
# OUTPUT TABS
# ─────────────────────────────────────────
if "data" in st.session_state:
    data     = st.session_state["data"]
    sections = st.session_state.get("sections", {})
    meta     = st.session_state.get("meta", {})
    raw_text = data.get("_raw_text", "")
    speakers = data.get("_speakers", [])

    st.divider()

    # Metadata bar
    mbits = []
    if meta.get("title"):       mbits.append(f"📌 **{meta['title']}**")
    if meta.get("date"):        mbits.append(f"📅 {meta['date']}")
    if meta.get("attendees"):   mbits.append(f"👥 {meta['attendees']}")
    if data.get("meetingType"): mbits.append(f"🏷 `{data['meetingType'].title()}`")
    if data.get("overallSentiment"):
        s   = data["overallSentiment"].lower()
        css = {"positive": "sentiment-pos", "negative": "sentiment-neg"}.get(s, "sentiment-neu")
        mbits.append(f'<span class="{css}">{s.upper()}</span>')
    if mbits:
        st.markdown("  ·  ".join(mbits), unsafe_allow_html=True)
        st.write("")

    tab_minutes, tab_analytics, tab_history = st.tabs([
        "📄 Meeting Minutes",
        "📊 Analytics Dashboard",
        "🗄️ Meeting History",
    ])

    # ══════════════════════════════════════
    # TAB 1 — MINUTES
    # ══════════════════════════════════════
    with tab_minutes:
        if sections.get("summary", True) and data.get("summary"):
            st.markdown('<p class="sh">Summary</p>', unsafe_allow_html=True)
            st.info(data["summary"])

        col_a, col_b = st.columns(2)

        with col_a:
            if sections.get("topics", True):
                with st.expander("🏷 Topics Discussed", expanded=True):
                    for t in data.get("topics", []):
                        st.markdown(f"- {t}")

            if sections.get("keypoints", True):
                with st.expander("💡 Key Points", expanded=True):
                    for p in data.get("keyPoints", []):
                        clean_p = re.sub(r"\[(POSITIVE|NEGATIVE|NEUTRAL)\]", "", p).strip()
                        m = re.search(r"\[(POSITIVE|NEGATIVE|NEUTRAL)\]", p, re.IGNORECASE)
                        if inc_sentiment and m:
                            tag = m.group(1).lower()
                            css = {"positive": "sentiment-pos", "negative": "sentiment-neg"}.get(tag, "sentiment-neu")
                            st.markdown(f'- {clean_p} <span class="{css}">{tag}</span>', unsafe_allow_html=True)
                        else:
                            st.markdown(f"- {clean_p}")

            if sections.get("decisions", True):
                with st.expander("✅ Decisions Made", expanded=True):
                    decs = data.get("decisions", [])
                    if decs:
                        for d in decs:
                            st.markdown(f"- {d}")
                    else:
                        st.caption("No firm decisions recorded.")

        with col_b:
            if sections.get("actions", True):
                with st.expander("⚡ Action Items", expanded=True):
                    items = data.get("actionItems", [])
                    if items:
                        for a in items:
                            pm = re.match(r"\[(HIGH|MEDIUM|LOW)\]\s*(.*)", a, re.IGNORECASE)
                            if pm:
                                lvl  = pm.group(1).upper()
                                rest = pm.group(2)
                                css  = {"HIGH": "priority-high", "MEDIUM": "priority-medium", "LOW": "priority-low"}[lvl]
                                st.markdown(f'<span class="{css}">[{lvl}]</span> {rest}', unsafe_allow_html=True)
                            else:
                                st.markdown(f"- {a}")
                    else:
                        st.caption("No action items (or all filtered).")

            if sections.get("nextsteps", True):
                with st.expander("➡️ Next Steps", expanded=True):
                    steps = data.get("nextSteps", [])
                    if steps:
                        for s in steps:
                            st.markdown(f"- {s}")
                    else:
                        st.caption("No next steps recorded.")

            if sections.get("speakers", True) and speakers:
                with st.expander("🎤 Speakers", expanded=False):
                    chips = "".join(f'<span class="donut-chip">🎤 {sp}</span>' for sp in speakers)
                    st.markdown(f'<div class="donut-wrap">{chips}</div>', unsafe_allow_html=True)

        st.divider()
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "📥 Download Markdown",
                st.session_state.get("markdown", ""),
                file_name=f"minutes_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with d2:
            raw_json = json.dumps(
                {k: v for k, v in data.items() if not k.startswith("_")}, indent=2
            )
            st.download_button(
                "📥 Download JSON",
                raw_json,
                file_name=f"minutes_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )

    # ══════════════════════════════════════
    # TAB 2 — ANALYTICS
    # ══════════════════════════════════════
    with tab_analytics:
        wc        = _word_count(raw_text)
        sc        = _sentence_count(raw_text)
        avg_sl    = _avg_sentence_len(raw_text)
        n_q       = _count_questions(raw_text)
        keywords  = _top_keywords(raw_text, 12)
        sent_dist = _sentiment_distribution(data.get("keyPoints", []))
        prio_dist = _priority_distribution(data.get("actionItems", []))
        spk_wc    = _speaker_word_counts(raw_text, speakers)

        st.markdown("### 📈 Transcript Statistics")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        for col, val, lbl in [
            (k1, f"{wc:,}",                            "Total Words"),
            (k2, sc,                                   "Sentences"),
            (k3, f"{avg_sl}",                          "Avg Sent. Length"),
            (k4, n_q,                                  "Questions Asked"),
            (k5, len(data.get("actionItems", [])),     "Action Items"),
            (k6, len(data.get("decisions", [])),       "Decisions"),
        ]:
            col.markdown(
                f'<div class="an-card"><div class="an-title">{lbl}</div>'
                f'<div class="an-big">{val}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        an1, an2, an3 = st.columns([1.3, 1.3, 1.4])

        with an1:
            st.markdown("### 🔑 Top Keywords")
            if keywords:
                max_kw = keywords[0][1]
                st.markdown("".join(_html_bar(w, c, max_kw) for w, c in keywords), unsafe_allow_html=True)
            else:
                st.caption("No keywords extracted.")

        with an2:
            st.markdown("### 😐 Sentiment Breakdown")
            total_s = sum(sent_dist.values()) or 1
            for label, color, css in [
                ("Positive", "#16a34a", "sentiment-pos"),
                ("Neutral",  "#475569", "sentiment-neu"),
                ("Negative", "#dc2626", "sentiment-neg"),
            ]:
                cnt = sent_dist.get(label.lower(), 0)
                pct = round(cnt / total_s * 100)
                st.markdown(
                    f'<div style="margin-bottom:8px">'
                    f'<span class="{css}">{label}</span>'
                    f'<span style="float:right;font-size:.8rem;color:#6b7280">{cnt} pts · {pct}%</span>'
                    f'<div style="background:#f1f5f9;border-radius:4px;height:8px;margin-top:4px">'
                    f'<div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("---")
            overall = data.get("overallSentiment", "neutral").lower()
            css_o   = {"positive": "sentiment-pos", "negative": "sentiment-neg"}.get(overall, "sentiment-neu")
            st.markdown(f'**Overall:** <span class="{css_o}">{overall.upper()}</span>', unsafe_allow_html=True)

        with an3:
            st.markdown("### ⚡ Action Item Priorities")
            total_p = sum(prio_dist.values()) or 1
            for label, color in [
                ("HIGH",    "#dc2626"),
                ("MEDIUM",  "#d97706"),
                ("LOW",     "#16a34a"),
                ("UNTAGGED","#94a3b8"),
            ]:
                cnt = prio_dist.get(label, 0)
                pct = round(cnt / total_p * 100)
                st.markdown(
                    f'<div style="margin-bottom:8px">'
                    f'<span style="font-weight:600;color:{color}">{label}</span>'
                    f'<span style="float:right;font-size:.8rem;color:#6b7280">{cnt} · {pct}%</span>'
                    f'<div style="background:#f1f5f9;border-radius:4px;height:8px;margin-top:4px">'
                    f'<div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            if data.get("meetingType"):
                st.markdown("---")
                st.markdown(f"**Meeting type:** `{data['meetingType'].title()}`")

        st.markdown("")
        sp1, sp2 = st.columns(2)

        with sp1:
            st.markdown("### 🎤 Speaker Talk-Time (estimated)")
            if spk_wc:
                max_sw = max(spk_wc.values())
                st.markdown(
                    "".join(
                        _html_bar(sp, wc_s, max_sw, "#764ba2")
                        for sp, wc_s in sorted(spk_wc.items(), key=lambda x: -x[1])
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(f"Total tracked: {sum(spk_wc.values()):,} words · {len(spk_wc)} speaker(s)")
            else:
                st.caption("No speaker labels detected (needs 'Name: …' format).")

        with sp2:
            st.markdown("### 📋 Decisions & Actions Timeline")
            decisions    = data.get("decisions", [])
            action_items = data.get("actionItems", [])
            if decisions or action_items:
                html_tl = ""
                for i, d in enumerate(decisions, 1):
                    html_tl += (
                        f'<div class="tl-item">'
                        f'<div class="tl-dot" style="background:#16a34a"></div>'
                        f'<div><div class="tl-text">✅ {d}</div>'
                        f'<div class="tl-time">Decision #{i}</div></div></div>'
                    )
                for i, a in enumerate(action_items, 1):
                    pm  = re.match(r"\[(HIGH|MEDIUM|LOW)\]\s*(.*)", a, re.IGNORECASE)
                    txt = pm.group(2) if pm else a
                    lvl = pm.group(1).upper() if pm else "—"
                    dot_color = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}.get(lvl, "#667eea")
                    html_tl += (
                        f'<div class="tl-item">'
                        f'<div class="tl-dot" style="background:{dot_color}"></div>'
                        f'<div><div class="tl-text">⚡ {txt}</div>'
                        f'<div class="tl-time">Action · {lvl}</div></div></div>'
                    )
                st.markdown(html_tl, unsafe_allow_html=True)
            else:
                st.caption("No decisions or action items to display.")

        st.markdown("")
        tk1, tk2 = st.columns(2)

        with tk1:
            st.markdown("### 📊 Keyword Frequency Table")
            if keywords:
                kw_data = [
                    {"Keyword": w, "Count": c, "% of top": f"{round(c / keywords[0][1] * 100)}%"}
                    for w, c in keywords
                ]
                st.dataframe(kw_data, use_container_width=True, hide_index=True)
            else:
                st.caption("No keywords.")

        with tk2:
            st.markdown("### 🏷 Topic Coverage")
            topics = data.get("topics", [])
            if topics:
                coverage = []
                for topic in topics:
                    t_words = set(topic.lower().split()) - STOPWORDS
                    hits    = sum(raw_text.lower().count(w) for w in t_words if len(w) > 3)
                    coverage.append((topic, hits))
                max_cov = max(c for _, c in coverage) if coverage else 1
                st.markdown(
                    "".join(_html_bar(t[:30], c, max_cov, "#4f46e5") for t, c in coverage),
                    unsafe_allow_html=True,
                )
                st.caption("Coverage = keyword hit count in transcript")
            else:
                st.caption("No topics extracted.")

    # ══════════════════════════════════════
    # TAB 3 — HISTORY
    # ══════════════════════════════════════
    with tab_history:
        st.markdown("### 🗄️ Meeting History")

        if not _db_ok:
            st.error(f"MySQL not connected: {_db_msg}")
            st.caption("Set MySQL env vars in Railway dashboard under Variables tab.")
        else:
            h1, h2 = st.columns([3, 1])
            with h1:
                search_q = st.text_input(
                    "🔍 Search past meetings",
                    placeholder="e.g. sprint, Q2, budget…",
                )
            with h2:
                hist_limit = st.number_input("Show last N", min_value=1, max_value=50, value=10)

            if st.button("🔄 Refresh History"):
                st.rerun()

            rows, err = db.fetch_past_transcripts(limit=hist_limit, search_term=search_q or None)

            if err:
                st.error(f"Query error: {err}")
            elif not rows:
                st.info("No meetings stored yet — generate your first set of minutes above!")
            else:
                st.caption(f"Showing {len(rows)} meeting(s)")
                for row in rows:
                    tid   = row["id"]
                    title = row.get("meeting_title") or f"Meeting #{tid}"
                    date  = str(row.get("meeting_date") or "")
                    mtype = row.get("meeting_type") or "—"
                    sent  = row.get("overall_sentiment") or "—"
                    summ  = row.get("summary") or "_No summary stored._"

                    with st.expander(f"📋 [{tid}] {title}  —  {date}  ·  {mtype.title()}"):
                        css_s = {"positive": "sentiment-pos", "negative": "sentiment-neg"}.get(
                            sent.lower(), "sentiment-neu"
                        )
                        st.markdown(
                            f'**Sentiment:** <span class="{css_s}">{sent.upper()}</span>  '
                            f'· **Type:** `{mtype.title()}`  · **Date:** {date}',
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Summary:**")
                        st.info(summ)

                        try:
                            spk_stored = json.loads(row.get("speakers") or "[]")
                            if spk_stored:
                                st.markdown("**Speakers:** " + "  ·  ".join(f"`{s}`" for s in spk_stored))
                        except Exception:
                            pass

                        if st.button(f"🗑 Delete meeting #{tid}", key=f"del_{tid}"):
                            ok, derr = db.delete_transcript(tid)
                            if ok:
                                st.success(f"Deleted meeting #{tid}")
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {derr}")