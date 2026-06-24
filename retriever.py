# retriever.py
# Extracted from cs455_project.ipynb — EventRetriever + parse_filters
import json
import math
import pickle
import re
import time
from datetime import date
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_JSON = Path("data/events_processed.json")
BM25_PATH      = Path("models/bm25_index.pkl")
FAISS_PATH     = Path("models/faiss_index.bin")
ID_MAP_PATH    = Path("models/event_id_map.json")
EMBED_MODEL    = "intfloat/multilingual-e5-base"

# ── Turkish tokenizer (Cell 18) ───────────────────────────────────────────────
TR_STOPWORDS = set("""
ve bir bu için ile de da bir o var yok bu şu ne kim nasıl nerede
olan her gibi kadar daha çok az hem veya ama ancak fakat
mi mı mu mü ya en ben sen biz siz onlar bunlar
şunlar hepsi hiç birçok bazı tüm bütün çeşitli herhangi
ile olan ki ise de da ki mı mi mu mü
""".split())

def tokenize_turkish(text: str) -> list:
    if not text:
        return []
    text = (
        text
        .replace("İ", "i").replace("I", "ı")
        .replace("Ş", "ş").replace("Ğ", "ğ")
        .replace("Ü", "ü").replace("Ö", "ö")
        .replace("Ç", "ç")
        .lower()
    )
    text = text.replace("'", " ").replace("'", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) >= 2 and t not in TR_STOPWORDS]

# ── Filter extraction (Cell 26) ───────────────────────────────────────────────
_ALL_CITIES = {
    "istanbul": "istanbul", "i̇stanbul": "istanbul",
    "ankara": "ankara",
    "izmir": "izmir", "i̇zmir": "izmir",
    "antalya": "antalya",
    "mugla": "mugla", "muğla": "mugla",
    "kocaeli": "kocaeli",
    "nevsehir": "nevsehir", "nevşehir": "nevsehir",
    "bursa": "bursa",
    "eskisehir": "eskisehir", "eskişehir": "eskisehir",
    "adana": "adana",
    "denizli": "denizli",
    "zonguldak": "zonguldak",
    "balikesir": "balikesir", "balıkesir": "balikesir",
    "mersin": "mersin",
    "konya": "konya",
    "gaziantep": "gaziantep",
    "canakkale": "canakkale", "çanakkale": "canakkale",
    "samsun": "samsun",
    "sakarya": "sakarya",
    "sivas": "sivas",
    "kayseri": "kayseri",
    "trabzon": "trabzon",
    "diyarbakir": "diyarbakir", "diyarbakır": "diyarbakir",
    "edirne": "edirne",
    "malatya": "malatya",
    "manisa": "manisa",
    "yalova": "yalova",
    "rize": "rize",
    "bolu": "bolu",
}

_CATEGORY_KEYWORDS = {
    "muzik":     ["konser", "müzik", "muzik", "concert", "music", "festival"],
    "tiyatro":   ["tiyatro", "theatre", "theater", "sahne", "oyun"],
    "stand-up":  ["stand up", "stand-up", "standup", "komedi", "comedy"],
    "cocuk":     ["çocuk", "cocuk", "çocuklar", "kids", "children"],
    "dans":      ["dans", "bale", "dance", "ballet"],
    "muze-sergi":["müze", "muze", "sergi", "exhibition", "fuar"],
    "atolye":    ["atölye", "atolye", "workshop", "eğitim"],
    "sinema":    ["sinema", "film", "cinema", "movie"],
    "spor":      ["spor", "maç", "futbol", "basketbol", "sport"],
    "eglence":   ["eğlence", "eglence", "parti", "party"],
    "tur-gezi":  ["tur", "gezi", "tour", "trip"],
}

_SUBCATEGORY_KEYWORDS = {
    "rock":         ["rock"],
    "pop":          ["pop"],
    "jazz":         ["jazz"],
    "classical":    ["klasik", "classical", "klasik müzik"],
    "rap-hiphop":   ["rap", "hip hop", "hiphop"],
    "metal":        ["metal", "heavy metal"],
    "electronic":   ["elektronik", "electronic", "dj", "edm"],
    "alternative":  ["alternatif", "alternative"],
    "turk-sanat-halk": ["türk sanat", "halk müziği", "arabesk"],
    "futbol":       ["futbol", "football", "soccer"],
    "basketbol":    ["basketbol", "basketball"],
    "bale":         ["bale", "ballet"],
}

def parse_filters(query: str) -> dict:
    filters = {
        "city": None, "category": None, "subcategory": None,
        "max_price": None, "date_from": None, "date_to": None,
    }
    q_lower = (
        query
        .replace("İ", "i").replace("I", "ı")
        .replace("Ş", "ş").replace("Ğ", "ğ")
        .replace("Ü", "ü").replace("Ö", "ö")
        .replace("Ç", "ç")
        .lower()
    )

    # City
    for raw, normalized in _ALL_CITIES.items():
        if raw in q_lower:
            filters["city"] = normalized
            break

    # Price — "500 TL altı", "500 lira altında", "under 500"
    price_match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:tl|lira|₺)?\s*(?:altı|altında|under|max|en fazla)",
        q_lower
    )
    if not price_match:
        price_match = re.search(
            r"(?:under|max|en fazla|altında)\s*(\d+(?:[.,]\d+)?)",
            q_lower
        )
    if price_match:
        try:
            filters["max_price"] = float(price_match.group(1).replace(",", "."))
        except ValueError:
            pass

    # Subcategory (before category so category can be inferred)
    for subcat, keywords in _SUBCATEGORY_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            filters["subcategory"] = subcat
            # infer parent category
            _subcat_to_cat = {
                "rock": "muzik", "pop": "muzik", "jazz": "muzik",
                "classical": "muzik", "rap-hiphop": "muzik", "metal": "muzik",
                "electronic": "muzik", "alternative": "muzik",
                "turk-sanat-halk": "muzik",
                "futbol": "spor", "basketbol": "spor",
                "bale": "dans",
            }
            filters["category"] = _subcat_to_cat.get(subcat)
            break

    # Category (only if not already set by subcategory)
    if not filters["category"]:
        for cat, keywords in _CATEGORY_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                filters["category"] = cat
                break

    return filters


# ── EventRetriever (Cell 27) ──────────────────────────────────────────────────
class EventRetriever:
    def __init__(
        self,
        processed_json: Path = PROCESSED_JSON,
        bm25_path:      Path = BM25_PATH,
        faiss_path:     Path = FAISS_PATH,
        id_map_path:    Path = ID_MAP_PATH,
        embed_model:    str  = EMBED_MODEL,
    ):
        print("[Retriever] Loading artifacts …")
        t0 = time.time()

        with open(processed_json, encoding="utf-8") as f:
            self.events = json.load(f)
        self.id_to_event = {ev["id"]: ev for ev in self.events}
        print(f"  events     : {len(self.events):,}")

        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.tokenized_corpus = bm25_data["tokenized_corpus"]
        print(f"  BM25       : vocab={len(self.bm25.idf):,}")

        self.index = faiss.read_index(str(faiss_path))
        print(f"  FAISS      : {self.index.ntotal:,} vectors")

        with open(id_map_path, encoding="utf-8") as f:
            id_map = json.load(f)
        # idx_to_id maps integer index → event id (may be stored as str keys)
        self.idx_to_id = {int(k): v for k, v in id_map["idx_to_id"].items()}

        print(f"  Loading embed model: {embed_model} …")
        self.embed_model = SentenceTransformer(embed_model)
        print(f"  Ready in {time.time()-t0:.1f}s")

    def encode_query(self, query: str) -> np.ndarray:
        prefixed = f"query: {query}"
        vec = self.embed_model.encode(
            [prefixed], convert_to_numpy=True, normalize_embeddings=True
        )
        return vec.astype(np.float32)

    def keyword_retrieve(self, query: str, top_k: int = 100) -> list:
        tokens = tokenize_turkish(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                ev = self.events[idx].copy()
                ev["_bm25_score"] = float(scores[idx])
                results.append(ev)
        return results

    def vector_retrieve(self, query: str, top_k: int = 100) -> list:
        query_vec = self.encode_query(query)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                event_id = self.idx_to_id.get(int(idx))
                if event_id and event_id in self.id_to_event:
                    ev = self.id_to_event[event_id].copy()
                    ev["_vec_score"] = float(score)
                    results.append(ev)
        return results

    def hybrid_retrieve(self, query: str, top_k: int = 10, pool_k: int = 100, rrf_k: int = 60) -> list:
        bm25_results = self.keyword_retrieve(query, top_k=pool_k)
        vec_results  = self.vector_retrieve(query,  top_k=pool_k)

        rrf_scores: dict = {}
        for rank, ev in enumerate(bm25_results):
            eid = ev["id"]
            rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, ev in enumerate(vec_results):
            eid = ev["id"]
            rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (rrf_k + rank + 1)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]
        results = []
        for eid in sorted_ids:
            ev = self.id_to_event[eid].copy()
            ev["_rrf_score"] = rrf_scores[eid]
            results.append(ev)
        return results

    def apply_filters(self, events: list, filters: dict) -> list:
        if not filters:
            return events
        out = []
        for ev in events:
            if filters.get("city") and ev.get("city") != filters["city"]:
                continue
            if filters.get("category") and ev.get("category") != filters["category"]:
                continue
            if filters.get("subcategory") and ev.get("subcategory") != filters["subcategory"]:
                continue
            if filters.get("max_price") is not None:
                price = ev.get("price")
                if price is not None and not math.isnan(float(price)) and float(price) > filters["max_price"]:
                    continue
            if filters.get("date_from") and ev.get("date") and ev["date"] < filters["date_from"]:
                continue
            if filters.get("date_to") and ev.get("date") and ev["date"] > filters["date_to"]:
                continue
            out.append(ev)
        return out

    def retrieve(self, query: str, method: str = "hybrid", top_k: int = 5,
                 filters: Optional[dict] = None, pool_k: int = 100) -> list:
        if method == "keyword":
            candidates = self.keyword_retrieve(query, top_k=pool_k)
        elif method == "vector":
            candidates = self.vector_retrieve(query, top_k=pool_k)
        else:
            candidates = self.hybrid_retrieve(query, top_k=pool_k, pool_k=pool_k)

        if filters:
            candidates = self.apply_filters(candidates, filters)

        return candidates[:top_k]