"""Entry point: run the full pipeline (stages 1–7)."""

import json
import time

import stage01_build_documents
import stage02_build_chunks
import stage03_extract_symbols
import stage04_retrieve
import stage05_extract_meanings
import stage06_postprocess
import stage07_extract_symbol_meanings

from config import BUILD_REPORT, OUTPUT_BASE


def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    build = {}
    wall_start = time.time()

    print("=== Stage 1: Build Documents ===")
    t0 = time.time()
    r1 = stage01_build_documents.run()
    build["stage01_documents"] = {**r1, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r1['paper_count']}  equations={r1['total_equations']}"
        f"  sentences={r1['total_sentences']}"
        f"  ({build['stage01_documents']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 2: Build Chunks ===")
    t0 = time.time()
    r2 = stage02_build_chunks.run()
    build["stage02_chunks"] = {**r2, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r2['paper_count']}  chunks={r2['total_chunks']}"
        f"  ({build['stage02_chunks']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 3: Extract Symbols ===")
    t0 = time.time()
    r3 = stage03_extract_symbols.run()
    build["stage03_symbols"] = {**r3, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r3['paper_count']}  symbols={r3['total_symbols']}"
        f"  enriched_chunks={r3['total_enriched_chunks']}"
        f"  ({build['stage03_symbols']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 4: BM25 Retrieval ===")
    t0 = time.time()
    r4 = stage04_retrieve.run()
    build["stage04_retrieval"] = {**r4, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r4['paper_count']}  queries={r4['total_queries']}"
        f"  results={r4['total_results']}"
        f"  ({build['stage04_retrieval']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 5: Extract Equation Meanings ===")
    t0 = time.time()
    r5 = stage05_extract_meanings.run()
    build["stage05_meanings"] = {**r5, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r5['paper_count']}  found={r5['total_meanings_found']}"
        f"  empty={r5['total_meanings_empty']}"
        f"  ({build['stage05_meanings']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 6: Postprocess Equation Meanings ===")
    t0 = time.time()
    r6 = stage06_postprocess.run()
    build["stage06_postprocess"] = {**r6, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r6['paper_count']}  records={r6['total_records']}"
        f"  changed={r6['total_changed']}  nonempty={r6['total_nonempty']}"
        f"  empty={r6['total_empty']}  flagged={r6['total_flagged']}"
        f"  ({build['stage06_postprocess']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 7: Extract Symbol Meanings ===")
    t0 = time.time()
    r7 = stage07_extract_symbol_meanings.run()
    build["stage07_symbol_meanings"] = {**r7, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r7['paper_count']}  symbols={r7['total_symbols']}"
        f"  definitions={r7['total_definitions_found']}"
        f"  ({build['stage07_symbol_meanings']['elapsed_seconds']:.1f}s)"
    )

    build["total_elapsed_seconds"] = round(time.time() - wall_start, 2)

    tmp = BUILD_REPORT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(build, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(BUILD_REPORT)

    print(f"\nSymbol meanings -> {r7['paper_count']} papers")
    print(f"Build report -> {BUILD_REPORT}")
    print(f"Total: {build['total_elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
