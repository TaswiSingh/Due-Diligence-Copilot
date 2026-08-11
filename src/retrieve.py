"""
Due Diligence Copilot - Retrieval + Risk-Specific Evidence Analysis

Pipeline:

    Question
        |
        v
    Topic + intent detection
        |
        v
    Semantic + keyword + topic retrieval
        |
        v
    Risk-category scoring
        |
        v
    Risk-specific evidence extraction
        |
        v
    Compact evidence context
        |
        v
    Local Ollama LLM
        |
        v
    Citation validation
        |
        +---- invalid --> deterministic evidence-grounded fallback

Designed for:
    - Apple SEC filing chunks
    - sentence-transformers embeddings
    - local Ollama / llama3.2:3b
    - evidence-grounded due-diligence answers
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed"

CHUNKS_PATH = PROCESSED_DIR / "apple_chunks.json"
EMBEDDINGS_PATH = PROCESSED_DIR / "apple_embeddings.npy"


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

import os

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

OLLAMA_TIMEOUT = 180

# Retrieval configuration

RETRIEVAL_CANDIDATES = 16

LLM_CONTEXT_CHUNKS = 5

MAX_RISK_FINDINGS = 4

MIN_FINAL_SCORE = 0.42


# Retrieval weights

SEMANTIC_WEIGHT = 0.35
KEYWORD_WEIGHT = 0.20
TOPIC_WEIGHT = 0.15
PRIORITY_WEIGHT = 0.15
RISK_CATEGORY_WEIGHT = 0.15


# ============================================================
# TOPIC DEFINITIONS
# ============================================================

TOPIC_KEYWORDS: Dict[str, List[str]] = {

    "supply_chain": [
        "supply chain",
        "supplier",
        "suppliers",
        "component",
        "components",
        "manufacturing",
        "manufacture",
        "assembly",
        "outsourcing",
        "outsourced",
        "logistics",
        "shipping",
        "shortage",
        "shortages",
        "single source",
        "limited source",
        "raw materials",
        "semiconductor",
        "semiconductors",
        "inventory",
        "tariff",
        "tariffs",
        "trade restrictions",
        "trade dispute",
        "production",
    ],

    "regulatory": [
        "regulatory",
        "regulation",
        "regulations",
        "government",
        "government investigation",
        "investigations",
        "law",
        "laws",
        "legal",
        "legislation",
        "compliance",
        "penalties",
        "competition law",
        "antitrust",
        "privacy",
        "data protection",
        "app store",
        "tariff",
        "tariffs",
        "trade restrictions",
    ],

    "cybersecurity": [
        "cybersecurity",
        "cyber security",
        "cyber",
        "ransomware",
        "malware",
        "hacking",
        "hack",
        "unauthorized access",
        "data breach",
        "breach",
        "information security",
        "security incident",
        "cyber attack",
        "cyber attacks",
        "security attacks",
    ],

    "financial": [
        "financial risk",
        "financial risks",
        "credit risk",
        "credit",
        "receivables",
        "collectibility",
        "liquidity",
        "debt",
        "interest rate",
        "interest rates",
        "currency",
        "foreign exchange",
        "exchange rate",
        "market volatility",
        "financial instruments",
        "cash",
        "cash equivalents",
        "securities",
        "tax",
        "taxes",
        "tax rate",
        "margin",
        "margins",
    ],

    "competitive": [
        "competition",
        "competitor",
        "competitors",
        "competitive",
        "market share",
        "pricing",
        "price competition",
        "innovation",
        "technology",
    ],

    "ip": [
        "intellectual property",
        "patent",
        "patents",
        "copyright",
        "copyrights",
        "trademark",
        "trademarks",
        "license",
        "licensing",
        "infringement",
    ],
}


# ============================================================
# RISK CATEGORY DEFINITIONS
# ============================================================

RISK_CATEGORIES: Dict[str, Dict[str, object]] = {

    "single_source": {
        "topics": ["supply_chain"],
        "keywords": [
            "single source",
            "single-source",
            "single or limited sources",
            "limited sources",
            "limited source",
            "only one source",
            "one source",
            "custom components",
            "sole source",
        ],
        "label": "Single-source and limited-source component dependence",
        "why": (
            "Dependence on a single or limited number of sources "
            "can make component availability vulnerable to supplier "
            "disruption, capacity constraints or unfavorable "
            "commercial terms."
        ),
    },

    "manufacturing_concentration": {
        "topics": ["supply_chain"],
        "keywords": [
            "manufacturing",
            "manufactured",
            "contract manufacturers",
            "outsourcing partners",
            "outsourcing",
            "final assembly",
            "assembly",
            "manufacturing capacities",
            "geographically concentrated",
            "primarily in china",
            "primarily in asia",
        ],
        "label": "Manufacturing and geographic concentration",
        "why": (
            "Reliance on external manufacturing partners and "
            "concentrated production locations can reduce direct "
            "control and increase exposure to regional or operational "
            "disruptions."
        ),
    },

    "component_shortage": {
        "topics": ["supply_chain"],
        "keywords": [
            "component shortages",
            "component shortage",
            "shortages of supply",
            "industry-wide shortages",
            "shortage of supply",
            "shortages",
            "commodity pricing",
            "commodity pricing fluctuations",
            "semiconductor",
            "semiconductors",
            "sufficient quantities",
            "capacity constraints",
            "pricing risks",
        ],
        "label": "Component shortages and pricing volatility",
        "why": (
            "Component shortages can constrain production and delay "
            "deliveries, while component price increases can raise "
            "costs and pressure margins."
        ),
    },

    "logistics_disruption": {
        "topics": ["supply_chain"],
        "keywords": [
            "transportation and logistics",
            "logistics providers",
            "logistics",
            "transportation",
            "delivery",
            "deliver products",
            "distribution",
            "supply and manufacturing chain",
            "delays and inefficiencies",
        ],
        "label": "Logistics and distribution disruption",
        "why": (
            "Disruption to transportation, logistics or distribution "
            "can delay products, increase costs and interfere with "
            "the company's ability to serve customers."
        ),
    },

    "geopolitical_trade": {
        "topics": ["supply_chain", "regulatory"],
        "keywords": [
            "trade restrictions",
            "trade disputes",
            "trade dispute",
            "tariffs",
            "tariff",
            "international disputes",
            "geopolitical tensions",
            "geopolitical",
            "political events",
            "conflict",
            "restrictions on international trade",
            "imports or exports",
        ],
        "label": "Geopolitical and trade disruption",
        "why": (
            "Trade restrictions and geopolitical events can increase "
            "costs, restrict sourcing options and interrupt manufacturing "
            "or product delivery."
        ),
    },

    "natural_disaster": {
        "topics": ["supply_chain"],
        "keywords": [
            "natural disasters",
            "natural disaster",
            "earthquakes",
            "extreme weather",
            "climate change",
            "fire",
            "power shortages",
            "industrial accidents",
        ],
        "label": "Natural-disaster and physical disruption",
        "why": (
            "Natural disasters and physical disruptions at company or "
            "supplier facilities can interrupt production, delay "
            "deliveries and increase operating costs."
        ),
    },

    "labor_public_health": {
        "topics": ["supply_chain"],
        "keywords": [
            "labor disputes",
            "labor dispute",
            "public health issues",
            "public health",
            "pandemic",
            "pandemics",
            "health issues",
            "workforce",
        ],
        "label": "Labor and public-health disruption",
        "why": (
            "Labor disruptions and major public-health events can "
            "interrupt supplier and manufacturing operations and "
            "reduce supply-chain capacity."
        ),
    },

    "cyber_disruption": {
        "topics": ["supply_chain", "cybersecurity"],
        "keywords": [
            "cybersecurity attacks",
            "cyber attack",
            "cyber attacks",
            "ransomware",
            "malware",
            "security incidents",
            "unauthorized access",
            "security attacks",
        ],
        "label": "Cybersecurity-related operational disruption",
        "why": (
            "Cybersecurity incidents affecting Apple or its suppliers "
            "can interrupt operations, expose information and disrupt "
            "manufacturing or service availability."
        ),
    },

    "regulatory_scrutiny": {
        "topics": ["regulatory"],
        "keywords": [
            "regulatory scrutiny",
            "increasing regulation",
            "government investigations",
            "government investigation",
            "legal actions",
            "regulatory requirements",
            "regulatory action",
            "regulatory changes",
            "government entities",
            "penalties",
        ],
        "label": "Regulatory scrutiny and government action",
        "why": (
            "Regulatory action can increase compliance costs, require "
            "changes to products or business practices and create "
            "exposure to investigations, litigation or penalties."
        ),
    },

    "privacy_data": {
        "topics": ["regulatory", "cybersecurity"],
        "keywords": [
            "privacy",
            "data protection",
            "data-protection",
            "personal data",
            "personal information",
            "privacy laws",
            "privacy regulations",
        ],
        "label": "Privacy and data-protection regulation",
        "why": (
            "Privacy and data-protection requirements can increase "
            "compliance costs and require changes to products, "
            "services or business practices."
        ),
    },

    "competition_antitrust": {
        "topics": ["regulatory", "competitive"],
        "keywords": [
            "competition-related laws",
            "competition-related regulations",
            "antitrust",
            "competition law",
            "competition laws",
            "competition regulations",
            "competitive practices",
        ],
        "label": "Competition and antitrust regulation",
        "why": (
            "Competition-related investigations and regulation can "
            "require changes to Apple's business practices and may "
            "result in litigation, penalties or restrictions."
        ),
    },

    "credit_collectibility": {
        "topics": ["financial"],
        "keywords": [
            "credit risk",
            "collectibility risk",
            "trade receivables",
            "vendor non-trade receivables",
            "receivables",
            "collectibility",
        ],
        "label": "Credit and collectibility risk",
        "why": (
            "Credit deterioration or reduced collectibility can affect "
            "receivables and adversely affect cash flows or financial "
            "results."
        ),
    },

    "market_financial": {
        "topics": ["financial"],
        "keywords": [
            "marketable securities",
            "financial instruments",
            "interest rates",
            "currency fluctuations",
            "foreign exchange",
            "exchange rates",
            "market volatility",
        ],
        "label": "Financial-market and financial-instrument exposure",
        "why": (
            "Changes in financial markets, interest rates or currencies "
            "can adversely affect financial instruments, earnings or "
            "cash flows."
        ),
    },

    "tax": {
        "topics": ["financial", "regulatory"],
        "keywords": [
            "tax laws",
            "tax law",
            "effective tax rates",
            "tax rate",
            "tax rates",
            "tax regulations",
            "tax uncertainty",
        ],
        "label": "Tax-related uncertainty",
        "why": (
            "Changes in tax laws, tax rates or tax positions can "
            "increase tax expense and adversely affect financial results."
        ),
    },

    "competitive_pressure": {
        "topics": ["competitive"],
        "keywords": [
            "competition",
            "competitors",
            "competitive",
            "market share",
            "price competition",
            "pricing",
            "innovation",
        ],
        "label": "Competitive and pricing pressure",
        "why": (
            "Intense competition can pressure pricing, market share, "
            "innovation requirements and financial performance."
        ),
    },

    "intellectual_property": {
        "topics": ["ip"],
        "keywords": [
            "intellectual property",
            "patent",
            "patents",
            "copyright",
            "trademark",
            "licensing",
            "infringement",
        ],
        "label": "Intellectual-property and infringement exposure",
        "why": (
            "Intellectual-property disputes or infringement claims can "
            "increase costs, restrict products or require changes to "
            "business practices."
        ),
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text for matching."""

    text = str(text or "").lower()

    text = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text: str) -> List[str]:
    """Return normalized tokens."""

    return normalize_text(text).split()


# ============================================================
# LOAD DATA
# ============================================================

def load_chunks() -> List[dict]:
    """Load chunks from processed Apple chunks file."""

    print(
        f"Loading chunks from: {CHUNKS_PATH}"
    )

    if not CHUNKS_PATH.exists():

        raise FileNotFoundError(
            f"Could not find chunks file:\n"
            f"{CHUNKS_PATH}\n\n"
            f"Expected file:\n"
            f"data/processed/apple_chunks.json"
        )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if isinstance(data, dict):

        if "chunks" in data:

            chunks = data["chunks"]

        else:

            chunks = list(
                data.values()
            )

    elif isinstance(data, list):

        chunks = data

    else:

        raise ValueError(
            "apple_chunks.json must contain "
            "a list or dictionary."
        )

    if not chunks:

        raise ValueError(
            "The chunks file contains no chunks."
        )

    return chunks


def load_embeddings() -> np.ndarray:
    """Load embedding vectors."""

    print(
        f"Loading embeddings from: {EMBEDDINGS_PATH}"
    )

    if not EMBEDDINGS_PATH.exists():

        raise FileNotFoundError(
            f"Could not find embeddings file:\n"
            f"{EMBEDDINGS_PATH}\n\n"
            f"Expected file:\n"
            f"data/processed/apple_embeddings.npy"
        )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    if not isinstance(
        embeddings,
        np.ndarray,
    ):

        embeddings = np.asarray(
            embeddings
        )

    if embeddings.ndim != 2:

        raise ValueError(
            "Embeddings must be a 2-dimensional "
            f"numpy array. Received shape: "
            f"{embeddings.shape}"
        )

    return embeddings


# ============================================================
# OLLAMA
# ============================================================

def check_ollama() -> None:
    """Check whether Ollama is running."""

    try:

        response = requests.get(
            OLLAMA_TAGS_URL,
            timeout=10,
        )

        response.raise_for_status()

    except Exception as exc:

        raise RuntimeError(
            "Could not connect to Ollama at "
            "http://localhost:11434.\n"
            "Make sure Ollama is running."
        ) from exc


def generate_with_ollama(
    prompt: str,
) -> str:
    """Generate an answer using local Ollama."""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "num_predict": 900,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    answer = data.get(
        "response",
        "",
    ).strip()

    if not answer:

        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return answer


# ============================================================
# TOPIC DETECTION
# ============================================================

def detect_topics(
    question: str,
) -> List[str]:
    """Detect question topics."""

    q = normalize_text(
        question
    )

    scores: Dict[str, int] = {}

    for topic, keywords in TOPIC_KEYWORDS.items():

        score = 0

        for keyword in keywords:

            keyword_normalized = normalize_text(
                keyword
            )

            if keyword_normalized in q:

                score += (
                    2
                    if " " in keyword_normalized
                    else 1
                )

        if score > 0:

            scores[topic] = score

    if not scores:

        return []

    ordered = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        topic
        for topic, score in ordered
        if score > 0
    ][:3]


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(
    question: str,
) -> str:
    """
    Detect broad question intent.

    Returns:
        ranking
        exposure
        explanation
        comparison
        general
    """

    q = normalize_text(
        question
    )

    ranking_phrases = [
        "biggest",
        "largest",
        "top",
        "key",
        "major",
        "most significant",
        "highest",
        "main",
        "greatest",
        "primary",
    ]

    comparison_phrases = [
        "compare",
        "comparison",
        "versus",
        "vs",
        "difference",
    ]

    explanation_phrases = [
        "why",
        "how",
        "explain",
        "explanation",
        "what causes",
    ]

    if any(
        phrase in q
        for phrase in ranking_phrases
    ):

        return "ranking"

    if any(
        phrase in q
        for phrase in comparison_phrases
    ):

        return "comparison"

    if any(
        phrase in q
        for phrase in explanation_phrases
    ):

        return "explanation"

    if any(
        phrase in q
        for phrase in [
            "risk",
            "risks",
            "exposure",
            "exposures",
        ]
    ):

        return "exposure"

    return "general"


# ============================================================
# KEYWORD SCORING
# ============================================================

def keyword_score(
    question: str,
    text: str,
) -> float:
    """Calculate question/evidence keyword overlap."""

    q = normalize_text(
        question
    )

    t = normalize_text(
        text
    )

    if not q or not t:

        return 0.0

    q_tokens = set(
        tokenize(q)
    )

    valid_tokens = [
        token
        for token in q_tokens
        if len(token) >= 3
    ]

    if not valid_tokens:

        return 0.0

    matched = sum(
        token in t
        for token in valid_tokens
    )

    base = (
        matched
        / len(valid_tokens)
    )

    phrase_bonus = 0.0

    for topic_keywords in TOPIC_KEYWORDS.values():

        for phrase in topic_keywords:

            p = normalize_text(
                phrase
            )

            if (
                " " in p
                and p in q
                and p in t
            ):

                phrase_bonus += 0.10

    return min(
        1.0,
        base + phrase_bonus,
    )


# ============================================================
# TOPIC SCORING
# ============================================================

def topic_score(
    text: str,
    topics: List[str],
) -> float:

    if not topics:

        return 0.0

    normalized = normalize_text(
        text
    )

    scores = []

    for topic in topics:

        keywords = TOPIC_KEYWORDS.get(
            topic,
            [],
        )

        if not keywords:

            continue

        matched = 0

        for keyword in keywords:

            if normalize_text(
                keyword
            ) in normalized:

                matched += 1

        scores.append(
            min(
                1.0,
                matched
                / max(
                    3,
                    len(keywords) * 0.25,
                ),
            )
        )

    if not scores:

        return 0.0

    return min(
        1.0,
        max(scores),
    )


# ============================================================
# RISK CATEGORY MATCHING
# ============================================================

def risk_category_matches(
    text: str,
    topics: List[str],
) -> Dict[str, float]:
    """
    Calculate how strongly a chunk supports each risk category.
    """

    normalized = normalize_text(
        text
    )

    scores: Dict[str, float] = {}

    for category, definition in RISK_CATEGORIES.items():

        category_topics = definition.get(
            "topics",
            [],
        )

        if topics and not any(
            topic in topics
            for topic in category_topics
        ):

            continue

        keywords = definition.get(
            "keywords",
            [],
        )

        matched_phrases = 0

        weighted_score = 0.0

        for keyword in keywords:

            phrase = normalize_text(
                keyword
            )

            if phrase in normalized:

                matched_phrases += 1

                if " " in phrase:

                    weighted_score += 0.25

                else:

                    weighted_score += 0.10

        if matched_phrases:

            scores[category] = min(
                1.0,
                weighted_score,
            )

    return scores


def best_risk_category_score(
    text: str,
    topics: List[str],
) -> float:

    scores = risk_category_matches(
        text,
        topics,
    )

    if not scores:

        return 0.0

    return max(
        scores.values()
    )


# ============================================================
# QUESTION-SPECIFIC PRIORITY
# ============================================================

def question_type_priority(
    question: str,
    chunk: dict,
    topics: List[str],
) -> float:
    """
    Additional relevance signal.

    Gives priority to evidence that directly addresses
    the requested topic rather than generic risk language.
    """

    text = normalize_text(
        chunk.get(
            "text",
            "",
        )
    )

    q = normalize_text(
        question
    )

    bonus = 0.0

    # --------------------------------------------------------
    # Supply chain
    # --------------------------------------------------------

    if "supply_chain" in topics:

        phrases = [
            "single or limited sources",
            "single source",
            "limited sources",
            "supply shortages",
            "component shortages",
            "component suppliers",
            "supply chain",
            "manufacturing",
            "outsourcing partners",
            "final assembly",
            "transportation and logistics",
            "commodity pricing",
            "semiconductor",
            "manufacturing capacities",
            "alternative source",
            "trade restrictions",
            "geopolitical tensions",
            "natural disasters",
        ]

        matches = sum(
            phrase in text
            for phrase in phrases
        )

        bonus += min(
            0.35,
            matches * 0.035,
        )

    # --------------------------------------------------------
    # Regulatory
    # --------------------------------------------------------

    if "regulatory" in topics:

        phrases = [
            "regulatory scrutiny",
            "increasing regulation",
            "government investigations",
            "legal actions",
            "penalties",
            "competition-related laws",
            "competition-related regulations",
            "privacy",
            "data protection",
            "government entities",
            "regulatory requirements",
            "court order",
            "app store",
        ]

        matches = sum(
            phrase in text
            for phrase in phrases
        )

        bonus += min(
            0.35,
            matches * 0.035,
        )

    # --------------------------------------------------------
    # Cybersecurity
    # --------------------------------------------------------

    if "cybersecurity" in topics:

        phrases = [
            "cybersecurity",
            "cybersecurity threats",
            "ransomware",
            "malicious attacks",
            "unauthorized access",
            "confidential information",
            "data security",
            "security incidents",
            "suppliers",
            "third parties",
        ]

        matches = sum(
            phrase in text
            for phrase in phrases
        )

        bonus += min(
            0.35,
            matches * 0.035,
        )

    # --------------------------------------------------------
    # Financial
    # --------------------------------------------------------

    if "financial" in topics:

        phrases = [
            "credit risk",
            "collectibility risk",
            "trade receivables",
            "vendor non trade receivables",
            "marketable securities",
            "financial instruments",
            "interest rates",
            "currency fluctuations",
            "tax laws",
            "effective tax rates",
            "liquidity",
        ]

        matches = sum(
            phrase in text
            for phrase in phrases
        )

        bonus += min(
            0.35,
            matches * 0.035,
        )

    # Direct question phrase overlap.

    for topic in topics:

        for phrase in TOPIC_KEYWORDS.get(
            topic,
            [],
        ):

            p = normalize_text(
                phrase
            )

            if (
                p in q
                and p in text
            ):

                bonus += 0.03

    return min(
        1.0,
        bonus,
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:

    query_norm = np.linalg.norm(
        query_embedding
    )

    if query_norm == 0:

        return np.zeros(
            len(embeddings)
        )

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    denominator = (
        embedding_norms
        * query_norm
    )

    denominator = np.where(
        denominator == 0,
        1e-12,
        denominator,
    )

    return (
        np.dot(
            embeddings,
            query_embedding,
        )
        / denominator
    )


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_score_array(
    values: List[float],
) -> List[float]:
    """
    Min-max normalize scores to 0-1.

    This makes retrieval diagnostics easier to interpret.
    """

    if not values:

        return []

    minimum = min(values)

    maximum = max(values)

    if maximum - minimum < 1e-12:

        return [
            1.0
            for _ in values
        ]

    return [
        (
            value - minimum
        )
        / (
            maximum - minimum
        )
        for value in values
    ]


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    question: str,
    chunks: List[dict],
    embeddings: np.ndarray,
    query_embedding_model: SentenceTransformer,
) -> Tuple[List[dict], List[str], str]:

    print(
        "\nDetecting question topic..."
    )

    topics = detect_topics(
        question
    )

    if topics:

        print(
            "Detected topics: "
            + ", ".join(topics)
        )

    else:

        print(
            "Detected topics: general"
        )

    intent = detect_intent(
        question
    )

    print(
        f"Detected intent: {intent}"
    )

    query_embedding = (
        query_embedding_model.encode(
            question,
            normalize_embeddings=True,
        )
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings,
    )

    candidates = []

    for index, chunk in enumerate(
        chunks
    ):

        text = chunk.get(
            "text",
            "",
        )

        semantic = float(
            similarities[index]
        )

        keyword = keyword_score(
            question,
            text,
        )

        topic = topic_score(
            text,
            topics,
        )

        priority = question_type_priority(
            question,
            chunk,
            topics,
        )

        risk_scores = risk_category_matches(
            text,
            topics,
        )

        risk_score = (
            max(
                risk_scores.values()
            )
            if risk_scores
            else 0.0
        )

        candidates.append(
            {
                "chunk": chunk,
                "semantic_score": semantic,
                "keyword_score": keyword,
                "topic_score": topic,
                "priority_score": priority,
                "risk_category_score": risk_score,
                "risk_category_scores": risk_scores,
            }
        )

    # --------------------------------------------------------
    # Normalize the raw semantic signal.
    # --------------------------------------------------------

    semantic_values = [
        item["semantic_score"]
        for item in candidates
    ]

    normalized_semantic = normalize_score_array(
        semantic_values
    )

    # --------------------------------------------------------
    # Calculate final normalized score.
    # --------------------------------------------------------

    for item, semantic_norm in zip(
        candidates,
        normalized_semantic,
    ):

        raw_score = (
            semantic_norm
            * SEMANTIC_WEIGHT
            + item["keyword_score"]
            * KEYWORD_WEIGHT
            + item["topic_score"]
            * TOPIC_WEIGHT
            + item["priority_score"]
            * PRIORITY_WEIGHT
            + item["risk_category_score"]
            * RISK_CATEGORY_WEIGHT
        )

        item["raw_final_score"] = raw_score

    raw_scores = [
        item["raw_final_score"]
        for item in candidates
    ]

    normalized_final = normalize_score_array(
        raw_scores
    )

    for item, normalized in zip(
        candidates,
        normalized_final,
    ):

        item["final_score"] = normalized

    candidates.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    selected = [
        item
        for item in candidates[
            :RETRIEVAL_CANDIDATES
        ]
        if item["final_score"]
        >= MIN_FINAL_SCORE
    ]

    if not selected:

        selected = candidates[
            :min(
                LLM_CONTEXT_CHUNKS,
                len(candidates),
            )
        ]

    # --------------------------------------------------------
    # Topic-aware filtering.
    # --------------------------------------------------------

    if topics:

        topic_selected = [
            item
            for item in selected
            if (
                item["topic_score"] >= 0.20
                or item["risk_category_score"] >= 0.20
                or item["priority_score"] >= 0.10
            )
        ]

        if len(topic_selected) >= 2:

            selected = topic_selected

    selected = selected[
        :LLM_CONTEXT_CHUNKS
    ]

    return (
        selected,
        topics,
        intent,
    )


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_into_sentences(
    text: str,
) -> List[str]:
    """
    Lightweight sentence splitter suitable for SEC text.
    """

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:

        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) > 20:

            cleaned.append(
                sentence
            )

    return cleaned


# ============================================================
# RISK-SPECIFIC EVIDENCE SCORING
# ============================================================

def risk_evidence_sentence_score(
    sentence: str,
    question: str,
    category: str,
    topics: List[str],
) -> float:
    """
    Score one sentence for a particular risk category.
    """

    normalized = normalize_text(
        sentence
    )

    q = normalize_text(
        question
    )

    definition = RISK_CATEGORIES.get(
        category,
        {},
    )

    keywords = definition.get(
        "keywords",
        [],
    )

    score = 0.0

    # --------------------------------------------------------
    # Category keyword matches
    # --------------------------------------------------------

    for keyword in keywords:

        phrase = normalize_text(
            keyword
        )

        if phrase in normalized:

            if " " in phrase:

                score += 0.25

            else:

                score += 0.10

    # --------------------------------------------------------
    # Question overlap
    # --------------------------------------------------------

    question_tokens = set(
        tokenize(q)
    )

    for token in question_tokens:

        if len(token) >= 4 and token in normalized:

            score += 0.025

    # --------------------------------------------------------
    # Risk-impact language
    # --------------------------------------------------------

    impact_phrases = [
        "adversely affect",
        "material adverse",
        "materially adversely",
        "could result",
        "could adversely",
        "risk",
        "risks",
        "disruption",
        "shortage",
        "increase costs",
        "increase the company",
        "difficult or impossible",
        "unable to",
        "may not be able",
        "uncertain",
    ]

    for phrase in impact_phrases:

        if phrase in normalized:

            score += 0.04

    # --------------------------------------------------------
    # Topic confirmation
    # --------------------------------------------------------

    for topic in topics:

        for keyword in TOPIC_KEYWORDS.get(
            topic,
            [],
        ):

            if normalize_text(
                keyword
            ) in normalized:

                score += 0.02

    return min(
        1.0,
        score,
    )


# ============================================================
# RISK-SPECIFIC PASSAGE EXTRACTION
# ============================================================

def extract_risk_evidence(
    text: str,
    question: str,
    category: str,
    topics: List[str],
    max_chars: int = 1000,
) -> Optional[str]:
    """
    Extract a short passage specifically supporting one risk.

    Unlike the old extractor, this does not simply choose the
    strongest generic sentence. It scores sentences against the
    requested risk category.
    """

    sentences = split_into_sentences(
        text
    )

    if not sentences:

        return None

    scored = []

    for index, sentence in enumerate(
        sentences
    ):

        score = risk_evidence_sentence_score(
            sentence,
            question,
            category,
            topics,
        )

        scored.append(
            (
                score,
                index,
                sentence,
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    if not scored:

        return None

    best_score = scored[0][0]

    if best_score <= 0:

        return None

    # Take the strongest sentence plus nearby supporting
    # sentences where appropriate.

    best_index = scored[0][1]

    selected_indices = [
        best_index
    ]

    for score, index, _ in scored[1:4]:

        if score >= best_score * 0.55:

            if abs(
                index - best_index
            ) <= 3:

                selected_indices.append(
                    index
                )

    selected_indices = sorted(
        set(selected_indices)
    )

    # Keep evidence compact.

    passage_parts = []

    for index in selected_indices:

        passage_parts.append(
            sentences[index]
        )

    passage = " ".join(
        passage_parts
    )

    if len(passage) > max_chars:

        passage = passage[
            :max_chars
        ]

        if " " in passage:

            passage = passage.rsplit(
                " ",
                1,
            )[0]

        passage += "..."

    return passage


# ============================================================
# BUILD RISK EVIDENCE SET
# ============================================================

def build_risk_evidence(
    question: str,
    selected: List[dict],
    topics: List[str],
    max_findings: int = MAX_RISK_FINDINGS,
) -> List[dict]:
    """
    Build distinct risk findings from retrieved chunks.

    Each finding contains:
        category
        label
        evidence
        chunk_id
        score
    """

    category_candidates: Dict[
        str,
        List[dict]
    ] = {}

    for item in selected:

        chunk = item["chunk"]

        text = chunk.get(
            "text",
            "",
        )

        category_scores = risk_category_matches(
            text,
            topics,
        )

        for category, category_score in category_scores.items():

            if category_score <= 0:

                continue

            evidence = extract_risk_evidence(
                text=text,
                question=question,
                category=category,
                topics=topics,
            )

            if not evidence:

                continue

            evidence_score = risk_evidence_sentence_score(
                evidence,
                question,
                category,
                topics,
            )

            combined_score = (
                item["final_score"] * 0.45
                + category_score * 0.30
                + evidence_score * 0.25
            )

            category_candidates.setdefault(
                category,
                [],
            ).append(
                {
                    "category": category,
                    "category_score": category_score,
                    "evidence_score": evidence_score,
                    "combined_score": combined_score,
                    "chunk": chunk,
                    "chunk_id": str(
                        chunk.get(
                            "chunk_id",
                            "unknown",
                        )
                    ),
                    "evidence": evidence,
                    "label": RISK_CATEGORIES[
                        category
                    ]["label"],
                    "why": RISK_CATEGORIES[
                        category
                    ]["why"],
                }
            )

    # --------------------------------------------------------
    # Choose strongest evidence for each category.
    # --------------------------------------------------------

    best_by_category = []

    for category, items in category_candidates.items():

        items.sort(
            key=lambda x: x["combined_score"],
            reverse=True,
        )

        best_by_category.append(
            items[0]
        )

    best_by_category.sort(
        key=lambda x: x["combined_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Avoid near-duplicate findings.
    # --------------------------------------------------------

    findings = []

    for candidate in best_by_category:

        category = candidate["category"]

        # Avoid two findings whose labels are identical.

        if any(
            finding["label"]
            == candidate["label"]
            for finding in findings
        ):

            continue

        findings.append(
            candidate
        )

        if len(findings) >= max_findings:

            break

    return findings


# ============================================================
# BUILD LLM CONTEXT
# ============================================================

def build_llm_context(
    risk_evidence: List[dict],
) -> str:
    """
    Build compact, risk-specific context for llama3.2:3b.
    """

    sections = []

    for number, item in enumerate(
        risk_evidence,
        start=1,
    ):

        sections.append(
            f"""
RISK {number}
Risk category: {item["label"]}
Chunk ID: {item["chunk_id"]}

Evidence:
{item["evidence"]}
""".strip()
        )

    if not sections:

        return (
            "No risk-specific evidence was extracted."
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    question: str,
    topics: List[str],
    intent: str,
    evidence_context: str,
) -> str:

    topic_text = (
        ", ".join(topics)
        if topics
        else "general business risk"
    )

    return f"""
You are a financial due-diligence analyst.

Answer the user's EXACT question using ONLY the risk-specific evidence below.

USER QUESTION:
{question}

DETECTED TOPIC:
{topic_text}

DETECTED INTENT:
{intent}

STRICT RULES:

1. Answer the user's exact question.
2. Do not say the question is missing.
3. Do not ask for another question.
4. Do not discuss these instructions.
5. Do not invent facts.
6. Do not introduce risks that are not supported by the supplied evidence.
7. For a ranking question, present the strongest supported risks first.
8. The filing does not necessarily provide a numerical ranking. Do not claim
   that a risk is objectively the largest unless the evidence establishes that.
9. Explain why each identified issue matters for due diligence.
10. Every Key Finding MUST contain an Evidence citation.
11. The Evidence citation MUST use the exact Chunk ID supplied below.
12. Do not cite any chunk that is not supplied below.
13. Keep the answer concise.
14. Prefer 3-4 strong findings over many weak findings.

Use EXACTLY this citation format:

**Evidence:** Chunk X

OUTPUT FORMAT:

## Conclusion

2-4 sentences directly answering the question.

## Key Findings

### 1. [Risk]

**What the filing says:** [brief evidence-grounded statement]

**Why it matters:** [brief analytical explanation]

**Evidence:** Chunk X

### 2. [Risk]

**What the filing says:** [brief evidence-grounded statement]

**Why it matters:** [brief analytical explanation]

**Evidence:** Chunk X

### 3. [Risk]

**What the filing says:** [brief evidence-grounded statement]

**Why it matters:** [brief analytical explanation]

**Evidence:** Chunk X

### 4. [Risk]

**What the filing says:** [brief evidence-grounded statement]

**Why it matters:** [brief analytical explanation]

**Evidence:** Chunk X

Only include findings for which evidence is actually supplied.

## Evidence Limitations

One short paragraph explaining what the retrieved evidence cannot establish.

RISK-SPECIFIC EVIDENCE:

{evidence_context}
""".strip()


# ============================================================
# ANSWER CLEANING
# ============================================================

def clean_answer(
    answer: str,
) -> str:

    answer = answer.strip()

    prefixes = [
        "assistant:",
        "answer:",
        "response:",
    ]

    lowered = answer.lower()

    for prefix in prefixes:

        if lowered.startswith(prefix):

            answer = answer[
                len(prefix):
            ].strip()

            break

    unwanted_tail_patterns = [
        "\nIf you have any further questions",
        "\nPlease let me know if you",
        "\nI hope this helps",
    ]

    for pattern in unwanted_tail_patterns:

        position = answer.find(
            pattern
        )

        if position != -1:

            answer = answer[
                :position
            ].rstrip()

    return answer


# ============================================================
# QUESTION CONFUSION VALIDATION
# ============================================================

def answer_has_question_confusion(
    answer: str,
) -> bool:

    normalized = normalize_text(
        answer
    )

    suspicious_phrases = [
        "i do not see any users question",
        "i dont see any users question",
        "i do not see the users question",
        "i dont see the users question",
        "no question was provided",
        "please provide the question",
        "could you provide the question",
        "there is no question",
        "i dont see a question",
        "i do not see a question",
        "there is no user question",
        "you have not provided a question",
    ]

    return any(
        phrase in normalized
        for phrase in suspicious_phrases
    )


# ============================================================
# GENERIC ANSWER VALIDATION
# ============================================================

def answer_is_generic(
    answer: str,
    question: str,
) -> bool:

    normalized = normalize_text(
        answer
    )

    generic_phrases = [
        "i can help you with that",
        "please provide more context",
        "i would be happy to help",
        "here are some general guidelines",
        "review the filing",
        "identify potential risks",
    ]

    generic_hits = sum(
        phrase in normalized
        for phrase in generic_phrases
    )

    if generic_hits >= 2:

        return True

    if (
        "key findings" not in normalized
        and "conclusion" not in normalized
    ):

        return True

    return False


# ============================================================
# CITATION VALIDATION
# ============================================================

def answer_has_evidence_citations(
    answer: str,
    risk_evidence: List[dict],
) -> bool:
    """
    Validate citations against actual retrieved risk evidence.

    We validate each numbered finding independently.
    """

    if not risk_evidence:

        return False

    valid_chunk_ids = {
        str(
            item["chunk_id"]
        ).strip()
        for item in risk_evidence
    }

    if not valid_chunk_ids:

        return False

    citation_pattern = re.compile(
        r"evidence\s*:\s*chunk\s+([A-Za-z0-9_-]+)",
        re.IGNORECASE,
    )

    citations = citation_pattern.findall(
        answer
    )

    if not citations:

        return False

    # Every citation must refer to retrieved evidence.

    for citation in citations:

        if citation.strip() not in valid_chunk_ids:

            return False

    # Locate numbered findings.

    finding_sections = re.findall(
        r"###\s+\d+\..*?(?=###\s+\d+\.|##\s+Evidence Limitations|$)",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not finding_sections:

        return False

    # Every finding must have its own citation.

    for finding in finding_sections:

        if not citation_pattern.search(
            finding
        ):

            return False

    return True


# ============================================================
# DETERMINISTIC CONCLUSION
# ============================================================

def deterministic_conclusion(
    question: str,
    risk_evidence: List[dict],
) -> str:

    if not risk_evidence:

        return (
            "The retrieved evidence does not establish "
            "a sufficiently reliable answer to this question."
        )

    labels = [
        item["label"]
        for item in risk_evidence[:4]
    ]

    if len(labels) == 1:

        return (
            f"The retrieved filing evidence identifies "
            f"{labels[0].lower()} as a material exposure "
            f"relevant to the question."
        )

    if len(labels) == 2:

        return (
            "The strongest risks supported by the retrieved "
            "filing evidence are "
            f"{labels[0].lower()} and "
            f"{labels[1].lower()}."
        )

    return (
        "The strongest supply-chain or business exposures "
        "supported by the retrieved filing evidence are "
        + ", ".join(
            label.lower()
            for label in labels[:-1]
        )
        + ", and "
        + labels[-1].lower()
        + "."
    )


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def fallback_answer(
    question: str,
    risk_evidence: List[dict],
    topics: List[str],
) -> str:

    if not risk_evidence:

        return (
            "## Conclusion\n\n"
            "The retrieved evidence does not establish "
            "a reliable answer to this question.\n\n"
            "## Evidence Limitations\n\n"
            "No sufficiently relevant risk-specific filing "
            "evidence was retrieved."
        )

    output = [
        "## Conclusion",
        "",
        deterministic_conclusion(
            question,
            risk_evidence,
        ),
        "",
        "The filing does not provide enough evidence "
        "to establish a precise numerical ranking of "
        "probability or financial impact.",
        "",
        "## Key Findings",
        "",
    ]

    for number, item in enumerate(
        risk_evidence[:MAX_RISK_FINDINGS],
        start=1,
    ):

        output.extend(
            [
                f"### {number}. {item['label']}",
                "",
                f"**What the filing says:** "
                f"{item['evidence']}",
                "",
                f"**Why it matters:** "
                f"{item['why']}",
                "",
                f"**Evidence:** Chunk {item['chunk_id']}",
                "",
            ]
        )

    output.extend(
        [
            "## Evidence Limitations",
            "",
            "The retrieved filing evidence supports these "
            "exposures but does not provide enough information "
            "to quantify the probability or dollar impact of "
            "each individual risk or establish a precise "
            "numerical ranking.",
        ]
    )

    return "\n".join(
        output
    )


# ============================================================
# ANALYSIS
# ============================================================

def analyze_question(
    question: str,
    selected: List[dict],
    topics: List[str],
    intent: str,
) -> str:

    if not selected:

        return fallback_answer(
            question,
            [],
            topics,
        )

    risk_evidence = build_risk_evidence(
        question=question,
        selected=selected,
        topics=topics,
        max_findings=MAX_RISK_FINDINGS,
    )

    if not risk_evidence:

        print(
            "Warning: no risk-specific evidence "
            "could be extracted. Using fallback."
        )

        return fallback_answer(
            question,
            [],
            topics,
        )

    context = build_llm_context(
        risk_evidence
    )

    prompt = build_prompt(
        question=question,
        topics=topics,
        intent=intent,
        evidence_context=context,
    )

    try:

        answer = generate_with_ollama(
            prompt
        )

        answer = clean_answer(
            answer
        )

        # ----------------------------------------------------
        # Validation 1
        # ----------------------------------------------------

        if answer_has_question_confusion(
            answer
        ):

            print(
                "Warning: LLM returned a "
                "question-confusion response. "
                "Using deterministic evidence answer."
            )

            return fallback_answer(
                question,
                risk_evidence,
                topics,
            )

        # ----------------------------------------------------
        # Validation 2
        # ----------------------------------------------------

        if answer_is_generic(
            answer,
            question,
        ):

            print(
                "Warning: LLM returned a "
                "generic/non-analytical response. "
                "Using deterministic evidence answer."
            )

            return fallback_answer(
                question,
                risk_evidence,
                topics,
            )

        # ----------------------------------------------------
        # Validation 3
        # ----------------------------------------------------

        if not answer_has_evidence_citations(
            answer,
            risk_evidence,
        ):

            print(
                "Warning: LLM response did not contain "
                "valid citations for every finding. "
                "Using deterministic evidence answer."
            )

            return fallback_answer(
                question,
                risk_evidence,
                topics,
            )

        return answer

    except Exception as exc:

        print(
            f"Warning: LLM generation failed: {exc}"
        )

        print(
            "Using deterministic evidence answer."
        )

        return fallback_answer(
            question,
            risk_evidence,
            topics,
        )


# ============================================================
# TERMINAL DISPLAY
# ============================================================

def print_evidence(
    selected: List[dict],
) -> None:

    print(
        "\n" + "=" * 60
    )

    print(
        "SELECTED EVIDENCE"
    )

    print(
        "=" * 60
    )

    for item in selected:

        chunk = item["chunk"]

        print()

        print(
            f"Chunk {chunk.get('chunk_id', 'unknown')}"
        )

        print(
            f"Section: "
            f"{chunk.get('section', 'Unknown')}"
        )

        print(
            f"Section Chunk: "
            f"{chunk.get('section_chunk', 'unknown')}"
        )

        print(
            f"Semantic similarity: "
            f"{item['semantic_score']:.4f}"
        )

        print(
            f"Keyword score: "
            f"{item['keyword_score']:.4f}"
        )

        print(
            f"Topic score: "
            f"{item['topic_score']:.4f}"
        )

        print(
            f"Priority score: "
            f"{item['priority_score']:.4f}"
        )

        print(
            f"Risk-category score: "
            f"{item['risk_category_score']:.4f}"
        )

        print(
            f"Final score: "
            f"{item['final_score']:.4f}"
        )

        risk_scores = item.get(
            "risk_category_scores",
            {},
        )

        if risk_scores:

            ordered = sorted(
                risk_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            categories = [
                category
                for category, score in ordered
                if score > 0
            ]

            print(
                "Risk categories: "
                + ", ".join(
                    categories
                )
            )


def print_risk_evidence(
    risk_evidence: List[dict],
) -> None:

    print(
        "\n" + "=" * 60
    )

    print(
        "RISK-SPECIFIC EVIDENCE"
    )

    print(
        "=" * 60
    )

    for number, item in enumerate(
        risk_evidence,
        start=1,
    ):

        print()

        print(
            f"{number}. {item['label']}"
        )

        print(
            f"Chunk: {item['chunk_id']}"
        )

        print(
            f"Risk score: "
            f"{item['combined_score']:.4f}"
        )

        print(
            "Evidence:"
        )

        print(
            item["evidence"]
        )


def print_source_summary(
    selected: List[dict],
) -> None:

    print(
        "\n" + "=" * 60
    )

    print(
        "RETRIEVED SOURCE SUMMARY"
    )

    print(
        "=" * 60
    )

    for item in selected:

        chunk = item["chunk"]

        print(
            f"Chunk {chunk.get('chunk_id')} | "
            f"{chunk.get('section')} | "
            f"section_chunk="
            f"{chunk.get('section_chunk')} | "
            f"final="
            f"{item['final_score']:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "Loading chunks..."
    )

    chunks = load_chunks()

    print(
        "Loading embeddings..."
    )

    embeddings = load_embeddings()

    if len(chunks) != len(embeddings):

        raise ValueError(
            "Number of chunks does not match "
            "number of embeddings.\n"
            f"Chunks: {len(chunks)}\n"
            f"Embeddings: {len(embeddings)}"
        )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Embeddings: {embeddings.shape}"
    )

    print(
        "Loading embedding model..."
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print()

    print(
        f"Local LLM: {LLM_MODEL}"
    )

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Retrieval candidates: "
        f"{RETRIEVAL_CANDIDATES}"
    )

    print(
        f"LLM context chunks: "
        f"{LLM_CONTEXT_CHUNKS}"
    )

    print(
        f"Maximum risk findings: "
        f"{MAX_RISK_FINDINGS}"
    )

    print(
        f"Minimum final score: "
        f"{MIN_FINAL_SCORE}"
    )

    print()

    print(
        "=" * 60
    )

    print(
        "              DUE DILIGENCE COPILOT"
    )

    print(
        "=" * 60
    )

    try:

        check_ollama()

        print(
            "Ollama: connected"
        )

    except RuntimeError as exc:

        print(
            str(exc)
        )

        sys.exit(1)

    while True:

        try:

            question = input(
                "\nAsk a question (or type 'exit'): "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print(
                "\nGoodbye."
            )

            break

        if not question:

            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:

            print(
                "Goodbye."
            )

            break

        try:

            print(
                "\nRetrieving relevant filing sections..."
            )

            selected, topics, intent = retrieve(
                question=question,
                chunks=chunks,
                embeddings=embeddings,
                query_embedding_model=embedding_model,
            )

            print_evidence(
                selected
            )

            # Build risk evidence once for diagnostics.

            risk_evidence = build_risk_evidence(
                question=question,
                selected=selected,
                topics=topics,
                max_findings=MAX_RISK_FINDINGS,
            )

            print_risk_evidence(
                risk_evidence
            )

            print(
                "\n" + "=" * 60
            )

            print(
                "GENERATING DUE-DILIGENCE ANALYSIS"
            )

            print(
                "=" * 60
            )

            answer = analyze_question(
                question=question,
                selected=selected,
                topics=topics,
                intent=intent,
            )

            print()

            print(
                answer
            )

            print_source_summary(
                selected
            )

            print(
                "\n" + "=" * 60
            )

            print(
                "END OF ANALYSIS"
            )

            print(
                "=" * 60
            )

        except Exception as exc:

            print(
                "\nERROR:"
            )

            print(
                str(exc)
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()