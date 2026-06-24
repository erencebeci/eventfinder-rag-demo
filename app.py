# app.py
import math
import os
import gradio as gr
from retriever import EventRetriever, parse_filters

# ── Load once at startup ──────────────────────────────────────────────────────
print("Loading EventRetriever …")
retriever = EventRetriever()
print("Ready.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_price(price):
    if price is None:
        return "—"
    try:
        if math.isnan(float(price)):
            return "—"
        return f"{float(price):.0f} TL"
    except (TypeError, ValueError):
        return "—"

def fmt_performers(performers):
    if not performers:
        return ""
    if isinstance(performers, list):
        return ", ".join(performers[:3])
    return str(performers)

def event_card(ev: dict, rank: int) -> str:
    title       = ev.get("title", "Untitled")
    date        = ev.get("date", "")
    time_str    = ev.get("time", "")
    venue       = ev.get("venue_name", "")
    city        = ev.get("city_display") or ev.get("city", "")
    category    = ev.get("category", "")
    subcategory = ev.get("subcategory", "")
    price       = fmt_price(ev.get("price"))
    url         = ev.get("url", "#")
    performers  = fmt_performers(ev.get("performers", []))

    cat_label = f"{category} / {subcategory}" if subcategory and subcategory != category else category

    score_str = ""
    if "_rrf_score" in ev:
        score_str = f'<span class="score">RRF {ev["_rrf_score"]:.4f}</span>'
    elif "_vec_score" in ev:
        score_str = f'<span class="score">cos {ev["_vec_score"]:.3f}</span>'
    elif "_bm25_score" in ev:
        score_str = f'<span class="score">BM25 {ev["_bm25_score"]:.2f}</span>'

    performers_html = f'<div class="detail">🎤 {performers}</div>' if performers else ""
    date_html = f'<div class="detail">📅 {date} {time_str}'.rstrip() + '</div>' if date else ""
    venue_html = f'<div class="detail">📍 {venue}, {city}'.rstrip(", ") + '</div>' if venue else ""

    return f"""
    <div class="card">
      <div class="card-header">
        <span class="rank">#{rank}</span>
        <a href="{url}" target="_blank" class="title">{title}</a>
        {score_str}
      </div>
      <div class="card-body">
        {date_html}
        {venue_html}
        {performers_html}
        <div class="tags">
          <span class="tag cat">{cat_label}</span>
          <span class="tag price">{price}</span>
        </div>
      </div>
    </div>"""

CSS = """
body { font-family: 'Inter', sans-serif; }
.card {
    background: #1e2130;
    border: 1px solid #3d4166;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.rank { background: #7c83fd; color: white; border-radius: 50%;
        width: 26px; height: 26px; display: flex; align-items: center;
        justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; }
.title { color: #e5e7eb; font-size: 15px; font-weight: 600;
         text-decoration: none; flex: 1; }
.title:hover { color: #a5b4fc; text-decoration: underline; }
.score { font-size: 11px; color: #6b7280; font-family: monospace;
         background: #111827; padding: 2px 6px; border-radius: 4px; }
.card-body { color: #9ca3af; font-size: 13px; }
.detail { margin-bottom: 3px; }
.tags { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
.tag { font-size: 11px; padding: 2px 8px; border-radius: 12px; font-weight: 500; }
.tag.cat { background: #312e81; color: #a5b4fc; }
.tag.price { background: #064e3b; color: #6ee7b7; }
.filter-box {
    background: #111827; border: 1px solid #374151; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 14px; font-size: 13px; color: #9ca3af;
}
.filter-box strong { color: #e5e7eb; }
.no-results { color: #6b7280; text-align: center; padding: 40px;
              font-size: 15px; }
"""

def search(query: str, method: str, top_k: int) -> str:
    if not query.strip():
        return '<div class="no-results">Enter a query above to find events.</div>'

    filters = parse_filters(query)
    active_filters = {k: v for k, v in filters.items() if v is not None}

    results = retriever.retrieve(
        query,
        method=method.lower(),
        top_k=top_k,
        filters=filters,
        pool_k=150,
    )

    # Filter badge
    if active_filters:
        badges = "  ".join(
            f"<strong>{k}:</strong> {v}" for k, v in active_filters.items()
        )
        filter_html = f'<div class="filter-box">🔍 Detected filters — {badges}</div>'
    else:
        filter_html = '<div class="filter-box">🔍 No structured filters detected — full semantic search</div>'

    if not results:
        return filter_html + '<div class="no-results">No events matched your query and filters.<br>Try broader terms or remove city/price constraints.</div>'

    cards = "".join(event_card(ev, i + 1) for i, ev in enumerate(results))
    count_line = f'<div style="color:#6b7280;font-size:12px;margin-bottom:10px;">{len(results)} result(s) · method: {method}</div>'
    return filter_html + count_line + cards


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="EventFinder-RAG") as demo:
    gr.Markdown("""
# 🎭 EventFinder-RAG
**Turkish Event Discovery via Hybrid BM25 + Dense Retrieval**

Type a query in Turkish or English. Filters (city, category, price) are extracted automatically.
_Retrieval only — no LLM generation. Uses [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) + BM25 + RRF fusion._
    """)

    with gr.Row():
        query_box = gr.Textbox(
            placeholder='e.g. "İstanbul\'da 500 TL altı rock konser" or "jazz concerts in Ankara"',
            label="Query",
            scale=4,
        )
        method_box = gr.Radio(
            choices=["Hybrid", "Keyword", "Vector"],
            value="Hybrid",
            label="Method",
            scale=1,
        )

    with gr.Row():
        top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-K results")
        search_btn   = gr.Button("Search 🔍", variant="primary")

    output_html = gr.HTML()

    gr.Examples(
        examples=[
            ["İstanbul'da rock konseri", "Hybrid", 5],
            ["Ankara'da 300 TL altı tiyatro", "Hybrid", 5],
            ["çocuklar için eğlenceli etkinlik", "Hybrid", 5],
            ["jazz concerts in Istanbul", "Hybrid", 5],
            ["İzmir'de stand-up komedi", "Hybrid", 5],
            ["müze ve sergi gezisi", "Vector", 5],
        ],
        inputs=[query_box, method_box, top_k_slider],
        outputs=output_html,
        fn=search,
        cache_examples=False,
    )

    search_btn.click(fn=search, inputs=[query_box, method_box, top_k_slider], outputs=output_html)
    query_box.submit(fn=search, inputs=[query_box, method_box, top_k_slider], outputs=output_html)

demo.launch()