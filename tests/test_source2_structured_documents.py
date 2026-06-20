from pathlib import Path
import json
import unittest

from source2.document_builder import build_paper_document, write_papers
from source2.equations import EquationResolver
from source2.html_document import parse_html_document
from source2.references import iter_explicit_references
from source2.sentences import SpacySentenceSegmenter


FIXTURE_HTML = r"""
<html>
  <head><title>Fallback title</title></head>
  <body>
    <h1 class="ltx_title_document">A Structured Paper</h1>
    <div class="ltx_abstract">
      <p id="abs.p1" class="ltx_p">Abstract text about Eq. (9).</p>
    </div>
    <section id="S1" class="ltx_section">
      <h2>1 Model</h2>
      <p id="S1.p1" class="ltx_p">The state <math alttext="\\psi"></math> is defined. See Eq. (1).</p>
      <table id="S1.E1" class="ltx_equation ltx_eqn_table">
        <tr><td><math alttext="x=y"><semantics>
          <annotation id="ann-1" encoding="application/x-tex">\displaystyle x=y</annotation>
        </semantics></math></td><td class="ltx_eqn_eqno">(1)</td></tr>
      </table>
      <table id="S1.E2" class="ltx_equation ltx_eqn_table">
        <tr><td><math alttext="a=b"><semantics>
          <annotation encoding="application/x-tex">\displaystyle a=b</annotation>
        </semantics></math></td><td class="ltx_eqn_eqno">(2)</td></tr>
      </table>
      <table id="S1.EX" class="ltx_equation ltx_eqn_table">
        <tr><td><math alttext="q=r"><semantics>
          <annotation encoding="application/x-tex">\displaystyle q=r</annotation>
        </semantics></math></td></tr>
      </table>
      <section id="S1.SS1" class="ltx_subsection">
        <h3>1.1 Details</h3>
        <p id="S1.SS1.p1" class="ltx_p">Using Eqs. (1, 2) gives the result.</p>
      </section>
    </section>
    <section id="bib" class="ltx_bibliography">
      <h2>References</h2><p id="bib.p1" class="ltx_p">Excluded bibliography text.</p>
    </section>
    <section id="A1" class="ltx_appendix">
      <h2>Appendix A</h2><p id="A1.p1" class="ltx_p">Appendix text.</p>
    </section>
  </body>
</html>
"""


def equation_entry(latex):
    return {
        "equation": latex,
        "surrounding_text": {"before": "before", "after": "after"},
        "audit-trail": [],
    }


class StructuredDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.segmenter = SpacySentenceSegmenter()

    def test_dom_structure_and_sentence_offsets(self):
        parsed = parse_html_document(FIXTURE_HTML, self.segmenter)
        sections = parsed.ordered_sections()
        section_by_id = {section.section_id: section for section in sections}

        self.assertEqual(parsed.title, "A Structured Paper")
        self.assertIn("__abstract__", section_by_id)
        self.assertIn("S1", section_by_id)
        self.assertIn("S1.SS1", section_by_id)
        self.assertIn("A1", section_by_id)
        self.assertNotIn("bib", section_by_id)
        self.assertEqual(section_by_id["S1.SS1"].parent_section_id, "S1")
        self.assertIn(r"\psi", parsed.paragraphs_by_id["S1.p1"].text)
        self.assertNotIn("bib.p1", parsed.paragraphs_by_id)
        for paragraph in parsed.paragraphs_by_id.values():
            for sentence in paragraph.sentences:
                self.assertEqual(paragraph.text[sentence.start:sentence.end], sentence.text)

    def test_reference_parser_handles_singular_lists_and_ranges(self):
        text = "See Eq. (2), Eqs. (3, 4), and Equations (5)-(7), but not bare (8)."
        references = [
            (kind, labels, match.group(0))
            for match, kind, labels in iter_explicit_references(text)
        ]

        self.assertEqual(references[0][:2], ("singular", ["2"]))
        self.assertEqual(references[1][:2], ("list", ["3", "4"]))
        self.assertEqual(references[2][:2], ("range", ["5", "6", "7"]))
        self.assertEqual(len(references), 3)

    def test_equation_resolution_fallbacks_and_unresolved(self):
        parsed = parse_html_document(FIXTURE_HTML, self.segmenter)
        resolver = EquationResolver(parsed)

        by_annotation = resolver.resolve(
            "1", equation_entry("x=y"),
            ['<annotation id="ann-1" encoding="application/x-tex">x=y</annotation>'],
        )
        by_label = resolver.resolve("2", equation_entry("a=b"), [])
        by_latex = resolver.resolve("3", equation_entry("q=r"), [])
        unresolved = resolver.resolve("4", equation_entry("does-not-exist"), [])

        self.assertEqual(by_annotation.record.match_method, "annotation_id")
        self.assertEqual(by_label.record.match_method, "visible_label")
        self.assertEqual(by_latex.record.match_method, "exact_latex")
        self.assertEqual(unresolved.record.match_method, "unresolved")

    def test_real_paper_integration(self):
        project = Path(__file__).resolve().parents[1]
        equations = json.loads((project / "data/3_equations.json").read_text())
        annotations = json.loads((project / "data/2_annotations.json").read_text())
        document = build_paper_document(
            "2403.03204",
            equations["2403.03204"],
            annotations["2403.03204"],
            project / "data/html/2403.03204.html",
            self.segmenter,
        )

        self.assertEqual(len(document.equations), 7)
        self.assertTrue(all(equation.section_id for equation in document.equations))
        self.assertTrue(document.sections)
        self.assertTrue(document.cross_references)

    def test_write_papers_creates_one_json_file_per_paper(self):
        import tempfile

        papers = [
            {"paper_id": "2403.03204", "title": "First"},
            {"paper_id": "2502.10129", "title": "Second"},
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "structured_papers"
            write_papers({"papers": papers}, output_dir)

            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["2403.03204.json", "2502.10129.json"],
            )
            self.assertEqual(
                json.loads((output_dir / "2403.03204.json").read_text()),
                papers[0],
            )


if __name__ == "__main__":
    unittest.main()
