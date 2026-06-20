import json
from pathlib import Path
import sys
import tempfile
import unittest


RETRIEVAL_DIR = Path(__file__).resolve().parents[1] / "source2/retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))

from index_builder import load_chunk_documents
from batch_search import search_all_papers, search_paper_equations
from query_builder import build_equation_meaning_query
from retrieval_models import ChunkDocument, SearchQuery
from retrieval_service import RetrievalService
from tokenizer import tokenize


def chunk(
    chunk_id,
    text,
    *,
    paper_id="paper",
    chunk_type="sentence",
    section_id="S1",
    equations=None,
    symbols=None,
):
    return ChunkDocument(
        chunk_id=chunk_id,
        paper_id=paper_id,
        chunk_type=chunk_type,
        text=text,
        section_id=section_id,
        section_title="Model",
        paragraph_ids=[f"{chunk_id}.p"],
        sentence_ids=[f"{chunk_id}.s"],
        nearby_equation_ids=equations or [],
        symbols=symbols or [],
        source="html",
    )


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            chunk(
                "meaning",
                r"Equation (3) defines the effective Hamiltonian H with \omega_c.",
                equations=["3"],
                symbols=["H", "omega_c"],
            ),
            chunk(
                "background",
                "The experiment measures temperature and pressure.",
            ),
            chunk(
                "other-paper",
                "Equation (3) defines another Hamiltonian.",
                paper_id="other",
                equations=["3"],
                symbols=["H"],
            ),
            chunk(
                "other-section",
                "The Hamiltonian describes free evolution.",
                section_id="S2",
                equations=["4"],
                symbols=["H"],
            ),
        ]
        self.service = RetrievalService(self.documents)

    def test_tokenizer_preserves_math_forms(self):
        tokens = tokenize(r"Eq. (3): \omega_{c} and ω_c with H_0")
        self.assertIn("3", tokens)
        self.assertIn("omega_c", tokens)
        self.assertIn("h_0", tokens)

    def test_bm25_ranks_evidence_and_returns_metadata(self):
        results = self.service.search(SearchQuery(
            text=r"Equation 3 defines Hamiltonian \omega_c",
            paper_id="paper",
            top_k=2,
        ))

        self.assertEqual(results[0].chunk_id, "meaning")
        self.assertEqual(results[0].method, "bm25")
        self.assertEqual(results[0].nearby_equation_ids, ["3"])
        self.assertGreater(results[0].score, 0)

    def test_metadata_filters_are_combined(self):
        results = self.service.search(SearchQuery(
            text="Hamiltonian",
            paper_id="paper",
            section_ids=["S1"],
            chunk_types=["sentence"],
            equation_ids=["3"],
            symbols=["omega_c"],
        ))

        self.assertEqual([result.chunk_id for result in results], ["meaning"])

    def test_tfidf_uses_the_same_interface(self):
        results = self.service.search(
            SearchQuery(text="effective Hamiltonian", paper_id="paper"),
            method="tfidf",
        )

        self.assertEqual(results[0].chunk_id, "meaning")
        self.assertEqual(results[0].method, "tfidf")

    def test_loads_only_requested_chunk_types(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "paper.json"
            values = [
                {**self.documents[0].__dict__},
                {**self.documents[1].__dict__, "chunk_type": "cross_reference"},
            ]
            path.write_text(json.dumps({
                "paper_id": "paper",
                "chunks": values,
            }), encoding="utf-8")

            documents = load_chunk_documents(Path(temporary_dir), ["sentence"])

            self.assertEqual([document.chunk_id for document in documents], ["meaning"])

    def test_builds_deterministic_equation_query(self):
        query = build_equation_meaning_query({
            "equation_id": "3",
            "symbols": [
                {"canonical": "H"},
                {"canonical": "omega_c"},
            ],
        })

        self.assertIn("Equation 3", query)
        self.assertIn("defines", query)
        self.assertIn("H", query)
        self.assertIn("omega_c", query)

    def test_batch_search_exports_one_file_per_paper(self):
        symbols = {
            "paper_id": "paper",
            "equations": [{
                "equation_id": "3",
                "latex": r"H=\omega_c a",
                "symbols": [
                    {"canonical": "H"},
                    {"canonical": "omega_c"},
                ],
            }],
        }
        payload = search_paper_equations(self.service, symbols, top_k=2)
        self.assertEqual(payload["queries"][0]["equation_id"], "3")
        self.assertEqual(payload["queries"][0]["results"][0]["chunk_id"], "meaning")

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            symbols_dir = root / "symbols"
            symbols_dir.mkdir()
            (symbols_dir / "paper.json").write_text(
                json.dumps(symbols), encoding="utf-8"
            )
            counts = search_all_papers(
                self.service, symbols_dir, root / "results", top_k=2
            )
            exported = json.loads((root / "results/paper.json").read_text())

            self.assertEqual(counts, (1, 1, 1))
            self.assertEqual(exported["paper_id"], "paper")
            self.assertEqual(exported["queries"][0]["task"], "equation_meaning_evidence")


if __name__ == "__main__":
    unittest.main()
