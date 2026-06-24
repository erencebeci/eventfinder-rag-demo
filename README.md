# EventFinder-RAG — Live Demo

> Retrieval-only demo for [EventFinder-RAG](https://github.com/erencebeci/eventfinder-rag) — try the hybrid BM25 + dense search pipeline live in your browser.

**[🤗 Open on Hugging Face Spaces](https://huggingface.co/spaces/erencebeci/eventfinder-rag)**

---

## What this demo does

Type a query in Turkish or English and get ranked Turkish events back instantly. Filters for city, category, and price are extracted automatically from your query — no dropdowns needed.

| Query example | Auto-detected filters |
|---|---|
| `İstanbul'da 500 TL altı rock konser` | city: istanbul · category: muzik · max_price: 500 |
| `Ankara'da tiyatro etkinlikleri` | city: ankara · category: tiyatro |
| `jazz concerts in Istanbul` | city: istanbul · subcategory: jazz |
| `çocuklar için eğlenceli etkinlik` | category: cocuk |

---

## Demo vs. full project

| | This demo | [Main project](https://github.com/erencebeci/eventfinder-rag) |
|---|---|---|
| BM25 retrieval | ✅ | ✅ |
| Dense retrieval (multilingual-e5-base) | ✅ | ✅ |
| Hybrid RRF fusion | ✅ | ✅ |
| Structured filter extraction | ✅ | ✅ |
| **LLM answer generation (Qwen2.5)** | ❌ CPU-only demo | ✅ Full implementation |
| **LLM query parser** | ❌ | ✅ |

The LLM generator and LLM-based query parser are implemented in the main project notebook but omitted here to run within Hugging Face Spaces' free CPU tier. The retrieval quality and filter logic are identical.

---

## How it works
Query (Turkish or English)

│

▼

parse_filters()          — rule-based city / category / price extraction

│

▼

┌───────────────────────────────────────┐

│           Hybrid Retrieval            │

│  BM25 (10,235-term inverted index)    │

│  +                                    │

│  FAISS (768-dim multilingual-e5-base) │

│  → fused via Reciprocal Rank Fusion   │

└───────────────┬───────────────────────┘

│

▼

Ranked event cards

---

## Repo structure
eventfinder-rag-demo/

├── app.py                  # Gradio interface

├── retriever.py            # EventRetriever class + parse_filters()

├── requirements.txt

├── README.md

├── data/

│   └── events_processed.json

└── models/

├── bm25_index.pkl

├── faiss_index.bin

└── event_id_map.json

---

## Run locally

```bash
git clone https://github.com/erencebeci/eventfinder-rag-demo
cd eventfinder-rag-demo
pip install -r requirements.txt
python app.py
```

Opens at `http://localhost:7860`.

---

## Main project

Full pipeline with LLM generation, LLM query parser, evaluation suite, and all preprocessing scripts:
**[github.com/erencebeci/eventfinder-rag](https://github.com/erencebeci/eventfinder-rag)**
