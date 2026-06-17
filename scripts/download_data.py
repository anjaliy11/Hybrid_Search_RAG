"""
Data Download Pipeline for Agentic RAG System.
Downloads three data sources into separate folders:
  1. ArXiv papers (AI/ML domain)
  2. Technical documentation (markdown)
  3. Wikipedia articles (diverse knowledge base)

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --skip-arxiv    # Skip ArXiv if slow
    python scripts/download_data.py --wiki-only     # Only Wikipedia

Author: Anjali Yadav
"""

import json
import logging
import time
import argparse
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

BASE_DATA_DIR = Path("./data/raw")
ARXIV_DIR = BASE_DATA_DIR / "papers"
DOCS_DIR = BASE_DATA_DIR / "docs"
WIKI_DIR = BASE_DATA_DIR / "wiki"
EVAL_DIR = Path("./data/eval_sets")

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_DELAY = 3  # seconds between requests (ArXiv rate limit)
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────


@dataclass
class Paper:
    title: str
    authors: List[str]
    abstract: str
    published: str
    arxiv_id: str
    categories: List[str]
    pdf_url: str


@dataclass
class WikiArticle:
    title: str
    content: str
    summary: str
    url: str
    word_count: int


@dataclass
class DownloadStats:
    source: str
    attempted: int
    succeeded: int
    failed: int
    total_chars: int

    def __str__(self) -> str:
        return (
            f"  [{self.source}] "
            f"{self.succeeded}/{self.attempted} succeeded | "
            f"{self.failed} failed | "
            f"{self.total_chars:,} chars total"
        )


# ─────────────────────────────────────────────────────────────
# ArXiv Downloader (No external library — uses ArXiv REST API)
# ─────────────────────────────────────────────────────────────


class ArxivDownloader:
    """Download AI/ML papers from ArXiv using their public API."""

    NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}

    QUERIES = [
        "retrieval augmented generation",
        "LLM agents agentic",
        "hallucination detection large language models",
        "vector database semantic search",
        "multi agent orchestration",
        "prompt engineering chain of thought",
        "reinforcement learning human feedback",
    ]

    def __init__(self, output_dir: Path, max_per_query: int = 8):
        self.output_dir = output_dir
        self.max_per_query = max_per_query
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self) -> DownloadStats:
        """Download papers for all queries."""
        stats = DownloadStats(
            source="ArXiv Papers",
            attempted=0,
            succeeded=0,
            failed=0,
            total_chars=0,
        )

        seen_ids = set()

        for query in self.QUERIES:
            logger.info(f"ArXiv query: '{query}'")
            papers = self._fetch_papers(query, self.max_per_query)

            for paper in papers:
                if paper.arxiv_id in seen_ids:
                    continue
                seen_ids.add(paper.arxiv_id)

                stats.attempted += 1
                if self._save_paper(paper):
                    stats.succeeded += 1
                    stats.total_chars += len(paper.abstract) + len(paper.title)
                else:
                    stats.failed += 1

            # Respect rate limit
            time.sleep(ARXIV_DELAY)

        return stats

    def _fetch_papers(self, query: str, max_results: int) -> List[Paper]:
        """Fetch papers from ArXiv API."""
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })

        url = f"{ARXIV_API_URL}?{params}"

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                xml_data = response.read().decode("utf-8")
        except Exception as e:
            logger.warning(f"ArXiv API request failed: {e}")
            return []

        return self._parse_response(xml_data)

    def _parse_response(self, xml_data: str) -> List[Paper]:
        """Parse ArXiv Atom XML response."""
        papers = []

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")
            return []

        for entry in root.findall("atom:entry", self.NAMESPACE):
            try:
                title = entry.find("atom:title", self.NAMESPACE)
                summary = entry.find("atom:summary", self.NAMESPACE)
                published = entry.find("atom:published", self.NAMESPACE)
                entry_id = entry.find("atom:id", self.NAMESPACE)

                if title is None or summary is None:
                    continue

                # Extract authors
                authors = []
                for author in entry.findall("atom:author", self.NAMESPACE):
                    name = author.find("atom:name", self.NAMESPACE)
                    if name is not None and name.text:
                        authors.append(name.text)

                # Extract categories
                categories = [
                    cat.get("term", "")
                    for cat in entry.findall("atom:category", self.NAMESPACE)
                ]

                # Extract PDF link
                pdf_url = ""
                for link in entry.findall("atom:link", self.NAMESPACE):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                        break

                # Clean text
                title_text = " ".join(title.text.strip().split())
                abstract_text = " ".join(summary.text.strip().split())

                arxiv_id = ""
                if entry_id is not None and entry_id.text:
                    arxiv_id = entry_id.text.split("/abs/")[-1]

                papers.append(Paper(
                    title=title_text,
                    authors=authors[:5],  # Limit authors
                    abstract=abstract_text,
                    published=published.text if published is not None else "",
                    arxiv_id=arxiv_id,
                    categories=categories[:3],
                    pdf_url=pdf_url,
                ))

            except Exception as e:
                logger.debug(f"Failed to parse entry: {e}")
                continue

        return papers

    def _save_paper(self, paper: Paper) -> bool:
        """Save paper metadata as JSON."""
        try:
            safe_id = paper.arxiv_id.replace("/", "_").replace(".", "_")
            filepath = self.output_dir / f"{safe_id}.json"

            if filepath.exists():
                return True  # Already downloaded

            # Create document in our standard format
            document = {
                "title": paper.title,
                "content": f"Title: {paper.title}\n\nAbstract: {paper.abstract}",
                "metadata": {
                    "source": f"arxiv_{safe_id}",
                    "type": "paper",
                    "authors": paper.authors,
                    "published": paper.published,
                    "arxiv_id": paper.arxiv_id,
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                },
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(document, f, indent=2, ensure_ascii=False)

            logger.debug(f"  Saved: {paper.title[:50]}...")
            return True

        except Exception as e:
            logger.warning(f"  Failed to save {paper.arxiv_id}: {e}")
            return False


# ─────────────────────────────────────────────────────────────
# Technical Docs Downloader
# ─────────────────────────────────────────────────────────────


class DocsDownloader:
    """Download public technical documentation from GitHub."""

    SOURCES = [
    {
        "name": "langchain_introduction",
        "url": "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/introduction.mdx",
        "title": "LangChain Introduction",
    },
    {
        "name": "langgraph_readme",
        "url": "https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md",
        "title": "LangGraph Overview",
    },
    {
        "name": "langchain_rag",
        "url": "https://raw.githubusercontent.com/langchain-ai/langchain/master/cookbook/rag_with_quantized_embeddings.ipynb",
        "title": "LangChain RAG Cookbook",
    },
    {
        "name": "chromadb_guide",
        "url": "https://raw.githubusercontent.com/chroma-core/chroma/main/README.md",
        "title": "ChromaDB Documentation",
    },
    {
        "name": "fastapi_tutorial",
        "url": "https://raw.githubusercontent.com/fastapi/fastapi/master/README.md",
        "title": "FastAPI Documentation",
    },
    {
        "name": "pinecone_overview",
        "url": "https://raw.githubusercontent.com/pinecone-io/pinecone-python-client/main/README.md",
        "title": "Pinecone Python Client",
    },
    {
        "name": "sentence_transformers",
        "url": "https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/README.md",
        "title": "Sentence Transformers Documentation",
    },
    {
        "name": "ragas_docs",
        "url": "https://raw.githubusercontent.com/explodinggradients/ragas/main/README.md",
        "title": "RAGAS Evaluation Framework",
    },
    {
        "name": "rank_bm25",
        "url": "https://raw.githubusercontent.com/dorianbrown/rank_bm25/master/README.md",
        "title": "BM25 Ranking Algorithm",
    },
    {
        "name": "huggingface_transformers",
        "url": "https://raw.githubusercontent.com/huggingface/transformers/main/README.md",
        "title": "HuggingFace Transformers",
    },
    {
        "name": "ollama_readme",
        "url": "https://raw.githubusercontent.com/ollama/ollama/main/README.md",
        "title": "Ollama Local LLM Runner",
    },
    {
        "name": "docker_getting_started",
        "url": "https://raw.githubusercontent.com/docker/getting-started/master/README.md",
        "title": "Docker Getting Started",
    },
]

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self) -> DownloadStats:
        """Download all technical documentation."""
        stats = DownloadStats(
            source="Technical Docs",
            attempted=0,
            succeeded=0,
            failed=0,
            total_chars=0,
        )

        for source in self.SOURCES:
            stats.attempted += 1

            content = self._fetch_url(source["url"])
            if content and len(content.strip()) > 100:
                if self._save_doc(source, content):
                    stats.succeeded += 1
                    stats.total_chars += len(content)
                else:
                    stats.failed += 1
            else:
                stats.failed += 1
                logger.warning(f"  Empty or failed: {source['name']}")

        return stats

    def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch content from URL."""
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.read().decode("utf-8")
        except Exception as e:
            logger.warning(f"  Fetch failed ({url[:50]}...): {e}")
            return None

    def _save_doc(self, source: Dict, content: str) -> bool:
        """Save documentation as JSON."""
        try:
            filepath = self.output_dir / f"{source['name']}.json"

            document = {
                "title": source["title"],
                "content": content,
                "metadata": {
                    "source": source["name"],
                    "type": "documentation",
                    "url": source["url"],
                    "char_count": len(content),
                },
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(document, f, indent=2, ensure_ascii=False)

            logger.info(f"  ✅ {source['title']} ({len(content):,} chars)")
            return True

        except Exception as e:
            logger.warning(f"  Failed to save {source['name']}: {e}")
            return False


# ─────────────────────────────────────────────────────────────
# Wikipedia Downloader (No external library — uses MediaWiki API)
# ─────────────────────────────────────────────────────────────


class WikipediaDownloader:
    """Download Wikipedia articles using the MediaWiki API directly."""

    TOPICS = [
        "Retrieval-augmented generation",
        "Large language model",
        "Transformer (deep learning architecture)",
        "Vector database",
        "Prompt engineering",
        "Reinforcement learning from human feedback",
        "Natural language processing",
        "Attention (machine learning)",
        "Fine-tuning (deep learning)",
        "Hallucination (artificial intelligence)",
        "Semantic search",
        "Knowledge graph",
        "Machine learning",
        "Artificial neural network",
        "Generative artificial intelligence",
        "Multi-agent system",
        "Information retrieval",
        "Word embedding",
        "Cosine similarity",
        "GPT-4",
        "BERT (language model)",
        "Recurrent neural network",
        "Convolutional neural network",
        "Backpropagation",
        "Gradient descent",
        "Overfitting",
        "Cross-validation (statistics)",
        "Mixture of experts",
        "Transfer learning",
        "Named-entity recognition",
    ]

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self) -> DownloadStats:
        """Download all Wikipedia articles."""
        stats = DownloadStats(
            source="Wikipedia",
            attempted=0,
            succeeded=0,
            failed=0,
            total_chars=0,
        )

        for topic in self.TOPICS:
            stats.attempted += 1

            article = self._fetch_article(topic)
            if article:
                if self._save_article(article):
                    stats.succeeded += 1
                    stats.total_chars += len(article.content)
                    logger.info(
                        f"  ✅ {article.title} ({article.word_count:,} words)"
                    )
                else:
                    stats.failed += 1
            else:
                stats.failed += 1
                logger.warning(f"  ❌ Failed: {topic}")

            # Small delay to be polite
            time.sleep(0.5)

        return stats

    def _fetch_article(self, title: str) -> Optional[WikiArticle]:
        """Fetch article from Wikipedia MediaWiki API."""
        params = urllib.parse.urlencode({
            "action": "query",
            "titles": title,
            "prop": "extracts|info",
            "explaintext": "true",
            "exsectionformat": "plain",
            "inprop": "url",
            "format": "json",
            "redirects": "1",
        })

        url = f"{WIKI_API_URL}?{params}"

        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "AgenticRAG/1.0 (research project; anjaliyadavknp9450@gmail.com)",
                },
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))

        except Exception as e:
            logger.debug(f"  API error for '{title}': {e}")
            return None

        # Parse response
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                # Page not found — try search
                return self._search_and_fetch(title)

            content = page_data.get("extract", "")
            if not content or len(content) < 200:
                continue

            # Extract summary (first paragraph)
            summary = content.split("\n\n")[0] if "\n\n" in content else content[:500]

            return WikiArticle(
                title=page_data.get("title", title),
                content=content,
                summary=summary,
                url=page_data.get("fullurl", f"https://en.wikipedia.org/wiki/{title}"),
                word_count=len(content.split()),
            )

        return None

    def _search_and_fetch(self, query: str) -> Optional[WikiArticle]:
        """Search Wikipedia and fetch the top result."""
        params = urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": "1",
            "format": "json",
        })

        url = f"{WIKI_API_URL}?{params}"

        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "AgenticRAG/1.0 (research project; anjaliyadavknp9450@gmail.com)",
                },
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))

            results = data.get("query", {}).get("search", [])
            if results:
                return self._fetch_article(results[0]["title"])

        except Exception as e:
            logger.debug(f"  Search failed for '{query}': {e}")

        return None

    def _save_article(self, article: WikiArticle) -> bool:
        """Save Wikipedia article as JSON."""
        try:
            safe_name = (
                article.title.lower()
                .replace(" ", "_")
                .replace("/", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
            )
            filepath = self.output_dir / f"{safe_name}.json"

            document = {
                "title": article.title,
                "content": article.content,
                "metadata": {
                    "source": safe_name,
                    "type": "wiki",
                    "url": article.url,
                    "word_count": article.word_count,
                    "summary": article.summary[:300],
                },
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(document, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.warning(f"  Save failed for '{article.title}': {e}")
            return False


# ─────────────────────────────────────────────────────────────
# Evaluation Set Generator
# ─────────────────────────────────────────────────────────────


def create_evaluation_set() -> None:
    """Create golden QA pairs for benchmarking the RAG system."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    eval_set = [
        {
            "question": "What is retrieval-augmented generation and how does it reduce hallucinations?",
            "ground_truth": (
                "RAG combines a retrieval system with a generative model, fetching relevant "
                "documents to ground the generation in factual information, thereby reducing "
                "hallucinations by constraining outputs to retrieved evidence."
            ),
            "type": "single_hop",
            "difficulty": "easy",
        },
        {
            "question": "How do transformers use attention mechanisms differently from RNNs for sequence processing?",
            "ground_truth": (
                "Transformers use self-attention to process all positions in parallel and capture "
                "long-range dependencies directly, while RNNs process sequences sequentially and "
                "struggle with long-range dependencies due to vanishing gradients."
            ),
            "type": "multi_hop",
            "difficulty": "medium",
        },
        {
            "question": "What is the relationship between RLHF and fine-tuning in making LLMs safer?",
            "ground_truth": (
                "RLHF is a fine-tuning technique that uses human preference data to train a reward "
                "model, which then guides the LLM policy optimization (usually via PPO) to generate "
                "outputs aligned with human values and safety requirements."
            ),
            "type": "multi_hop",
            "difficulty": "medium",
        },
        {
            "question": "Compare vector databases and traditional databases for semantic search.",
            "ground_truth": (
                "Vector databases store high-dimensional embeddings and support similarity search "
                "using metrics like cosine similarity, enabling semantic matching. Traditional "
                "databases use exact matching on structured data and cannot capture semantic meaning."
            ),
            "type": "comparison",
            "difficulty": "easy",
        },
        {
            "question": "What are Chain-of-Thought and Few-Shot prompting techniques?",
            "ground_truth": (
                "Chain-of-Thought prompting asks the model to show intermediate reasoning steps "
                "before answering. Few-Shot provides example input-output pairs in the prompt to "
                "demonstrate the desired format and behavior."
            ),
            "type": "single_hop",
            "difficulty": "easy",
        },
        {
            "question": "How does a knowledge graph enhance retrieval-augmented generation systems?",
            "ground_truth": (
                "Knowledge graphs provide structured relationships between entities that augment "
                "RAG by enabling more precise retrieval, supporting multi-hop reasoning across "
                "connected facts, and providing explicit context about entity relationships."
            ),
            "type": "multi_hop",
            "difficulty": "hard",
        },
        {
            "question": "Explain overfitting and how cross-validation helps prevent it.",
            "ground_truth": (
                "Overfitting occurs when a model learns noise in training data and fails to "
                "generalize. Cross-validation splits data into multiple folds, training and "
                "evaluating on different subsets, giving a more reliable estimate of generalization."
            ),
            "type": "single_hop",
            "difficulty": "easy",
        },
        {
            "question": "What are the key architectural differences between GPT-4 and BERT?",
            "ground_truth": (
                "GPT-4 is a decoder-only autoregressive model optimized for generation, while "
                "BERT is an encoder-only model using masked language modeling, optimized for "
                "understanding tasks like classification. GPT-4 is much larger and more versatile."
            ),
            "type": "comparison",
            "difficulty": "medium",
        },
        {
            "question": "How does the Mixture of Experts architecture improve model efficiency?",
            "ground_truth": (
                "Mixture of Experts uses a gating network to route inputs to specialized sub-networks "
                "(experts), activating only a subset for each input. This allows scaling model capacity "
                "without proportionally increasing computation cost."
            ),
            "type": "single_hop",
            "difficulty": "medium",
        },
        {
            "question": "What is transfer learning and how does fine-tuning relate to it in NLP?",
            "ground_truth": (
                "Transfer learning reuses knowledge from a model trained on one task for a different "
                "task. In NLP, models are pre-trained on large corpora then fine-tuned on specific "
                "downstream tasks with smaller labeled datasets, leveraging learned representations."
            ),
            "type": "multi_hop",
            "difficulty": "easy",
        },
    ]

    # Save complete eval set
    with open(EVAL_DIR / "eval_set.json", "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)

    # Save by type
    for eval_type in ["single_hop", "multi_hop", "comparison"]:
        subset = [e for e in eval_set if e["type"] == eval_type]
        if subset:
            with open(EVAL_DIR / f"{eval_type}.json", "w", encoding="utf-8") as f:
                json.dump(subset, f, indent=2, ensure_ascii=False)

    # Save adversarial set (questions designed to trigger hallucination)
    adversarial = [
        {
            "question": "What did the 2026 GPT-5 paper say about consciousness in AI?",
            "ground_truth": "This information does not exist in the knowledge base.",
            "type": "adversarial",
            "expected_behavior": "should_refuse",
        },
        {
            "question": "According to the documents, which is better: PyTorch or TensorFlow?",
            "ground_truth": "The documents may contain factual comparisons but not subjective judgments.",
            "type": "adversarial",
            "expected_behavior": "should_be_neutral",
        },
        {
            "question": "Explain quantum computing's role in transformer training.",
            "ground_truth": "Quantum computing is not currently used in transformer training.",
            "type": "adversarial",
            "expected_behavior": "should_refuse_or_clarify",
        },
    ]

    with open(EVAL_DIR / "adversarial.json", "w", encoding="utf-8") as f:
        json.dump(adversarial, f, indent=2, ensure_ascii=False)

    logger.info(
        f"  ✅ Eval set: {len(eval_set)} QA pairs + {len(adversarial)} adversarial"
    )


# ─────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download data for Agentic RAG system"
    )
    parser.add_argument(
        "--skip-arxiv", action="store_true",
        help="Skip ArXiv paper downloads (useful if API is slow)"
    )
    parser.add_argument(
        "--skip-docs", action="store_true",
        help="Skip technical documentation downloads"
    )
    parser.add_argument(
        "--wiki-only", action="store_true",
        help="Only download Wikipedia articles"
    )
    parser.add_argument(
        "--max-arxiv", type=int, default=8,
        help="Max papers per ArXiv query (default: 8)"
    )
    return parser.parse_args()


def print_banner():
    print("\n" + "=" * 60)
    print("  📥  AGENTIC RAG — DATA DOWNLOAD PIPELINE")
    print("=" * 60)
    print(f"  Output: {BASE_DATA_DIR.resolve()}")
    print(f"  Papers: {ARXIV_DIR.relative_to('.')}")
    print(f"  Docs:   {DOCS_DIR.relative_to('.')}")
    print(f"  Wiki:   {WIKI_DIR.relative_to('.')}")
    print(f"  Eval:   {EVAL_DIR.relative_to('.')}")
    print("=" * 60 + "\n")


def print_summary(all_stats: List[DownloadStats]):
    print("\n" + "=" * 60)
    print("  📊  DOWNLOAD SUMMARY")
    print("=" * 60)

    total_files = sum(s.succeeded for s in all_stats)
    total_chars = sum(s.total_chars for s in all_stats)

    for stats in all_stats:
        print(stats)

    print(f"\n  {'─' * 40}")
    print(f"  Total files: {total_files}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated chunks (@ 512 chars): ~{total_chars // 512}")
    print("=" * 60)
    print(f"\n  ✅ Data ready! Next step:")
    print(f"     python scripts/ingest.py\n")


def main():
    args = parse_args()
    print_banner()

    all_stats: List[DownloadStats] = []

    # ─── Wikipedia ───
    print("📚 [1/4] Downloading Wikipedia articles...")
    wiki_downloader = WikipediaDownloader(WIKI_DIR)
    wiki_stats = wiki_downloader.download_all()
    all_stats.append(wiki_stats)
    print(f"  Done: {wiki_stats.succeeded} articles\n")

    if args.wiki_only:
        create_evaluation_set()
        print_summary(all_stats)
        return

    # ─── Technical Docs ───
    if not args.skip_docs:
        print("📄 [2/4] Downloading technical documentation...")
        docs_downloader = DocsDownloader(DOCS_DIR)
        docs_stats = docs_downloader.download_all()
        all_stats.append(docs_stats)
        print(f"  Done: {docs_stats.succeeded} documents\n")
    else:
        logger.info("  ⏭️  Skipping docs (--skip-docs)")

    # ─── ArXiv Papers ───
    if not args.skip_arxiv:
        print("🔬 [3/4] Downloading ArXiv papers...")
        print("  (This may take 30-60 seconds due to API rate limits)")
        arxiv_downloader = ArxivDownloader(ARXIV_DIR, max_per_query=args.max_arxiv)
        arxiv_stats = arxiv_downloader.download_all()
        all_stats.append(arxiv_stats)
        print(f"  Done: {arxiv_stats.succeeded} papers\n")
    else:
        logger.info("  ⏭️  Skipping ArXiv (--skip-arxiv)")

    # ─── Evaluation Set ───
    print("📝 [4/4] Creating evaluation sets...")
    create_evaluation_set()

    # ─── Summary ───
    print_summary(all_stats)


if __name__ == "__main__":
    main()