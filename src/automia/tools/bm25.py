import bm25s
import Stemmer
from typing import List, Dict
from ..common import get_logger

class BM25Tool:
    def __init__(self, existing_experiments=[]):
        self._logger = get_logger("system")
        stem = Stemmer.Stemmer("english")
        self._tokenizer = bm25s.tokenization.Tokenizer(stemmer=stem)
        self._logger.info(f"BM25Tool: Indexing {len(existing_experiments)} existing experiments")
        self._index(existing_experiments)
        self._logger.info(f"BM25Tool: Indexing complete")

    def _format_experiment(self, experiment):
        exp_text = f"Idea: {experiment['idea']}\n\nDesign Justification: {experiment['design_justification']}\nImplementation: {experiment['implementation']}\nAnalysis Summary: {experiment['analysis_summary']}"

        return exp_text

    def _index(self, experiments: List[Dict]):
        if len(experiments) == 0:
            self._corpus_with_metadata = []
            self._corpus_text = []
            self._retriever = bm25s.BM25()
        else:
            self._corpus_with_metadata = experiments
            self._corpus_text = [self._format_experiment(exp) for exp in experiments]
            self._corpus_tokens = self._tokenizer.tokenize(self._corpus_text, return_as="tuple")
            self._retriever = bm25s.BM25(corpus=self._corpus_with_metadata)
            self._retriever.index(self._corpus_tokens)

    def update_index(self, experiment: Dict):
        self._logger.info(f"BM25Tool: Updating index with new experiment")
        self._corpus_with_metadata.append(experiment)
        self._corpus_text.append(self._format_experiment(experiment))
        self._corpus_tokens = self._tokenizer.tokenize(self._corpus_text, return_as="tuple")
        self._retriever = bm25s.BM25(corpus=self._corpus_with_metadata)
        self._retriever.index(self._corpus_tokens)
        self._logger.info(f"BM25Tool: Index updated successfully")

    def retrieve(self, query: str, k: int = 10):
        self._logger.info(f"BM25Tool: Retrieving top {k} experiments")
        if len(self._corpus_with_metadata) == 0:
            self._logger.warning(f"BM25Tool: No experiments to retrieve from")
            return []
        else:
            query_tokens = self._tokenizer.tokenize(query)
            min_k = min(k, len(self._corpus_with_metadata))
            results = self._retriever.retrieve(query_tokens, k=min_k)
            self._logger.info(f"BM25Tool: Retrieval complete")
            return [results.documents[0, i] for i in range(results.documents.shape[1])]
