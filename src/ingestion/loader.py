"""
Document Loader — reads multiple file formats into standardized schema.
Handles: JSON (our data format), Markdown, TXT, PDF.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads raw files into a list of document dicts.
    
    Output schema:
        {"content": str, "metadata": {"source": str, "title": str, "type": str, ...}}
    """

    SUPPORTED = {".json", ".md", ".txt", ".pdf"}

    def load_directory(self, directory: str) -> List[Dict]:
        """Recursively load all supported documents from directory."""
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []

        documents = []
        for path in sorted(dir_path.rglob("*")):
            if path.suffix not in self.SUPPORTED:
                continue
            if path.stat().st_size == 0:
                continue

            try:
                docs = self._dispatch(path)
                documents.extend(docs)
            except Exception as e:
                logger.debug(f"Skip {path.name}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents

    def _dispatch(self, path: Path) -> List[Dict]:
        """Route to format-specific loader."""
        loaders = {
            ".json": self._load_json,
            ".md": self._load_text,
            ".txt": self._load_text,
            ".pdf": self._load_pdf,
        }
        loader = loaders.get(path.suffix, lambda p: [])
        return loader(path)

    def _load_json(self, path: Path) -> List[Dict]:
        """Load our standard JSON document format."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        content = data.get("content", "")
        if len(content.strip()) < 50:
            return []

        metadata = data.get("metadata", {})
        metadata.setdefault("source", path.stem)
        metadata.setdefault("title", data.get("title", path.stem))
        metadata.setdefault("file", path.name)

        return [{"content": content, "metadata": metadata}]

    def _load_text(self, path: Path) -> List[Dict]:
        """Load markdown or plaintext."""
        content = path.read_text(encoding="utf-8", errors="ignore")
        if len(content.strip()) < 50:
            return []

        return [{
            "content": content,
            "metadata": {
                "source": path.stem,
                "title": path.stem.replace("_", " ").replace("-", " ").title(),
                "type": "markdown" if path.suffix == ".md" else "text",
                "file": path.name,
            },
        }]

    def _load_pdf(self, path: Path) -> List[Dict]:
        """Load PDF via PyMuPDF (free, fast)."""
        try:
            import fitz
            doc = fitz.open(path)
            pages = [page.get_text() for page in doc]
            doc.close()

            text = "\n\n".join(pages)
            if len(text.strip()) < 50:
                return []

            return [{
                "content": text,
                "metadata": {
                    "source": path.stem,
                    "title": path.stem,
                    "type": "pdf",
                    "pages": len(pages),
                    "file": path.name,
                },
            }]
        except ImportError:
            return []
        except Exception as e:
            logger.debug(f"PDF failed ({path.name}): {e}")
            return []