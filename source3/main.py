"""Entry point: run the full pipeline (stages 1–10)."""

import json
import time

import stage01_download_html
import stage02_build_documents
import stage03_build_chunks
import stage04_extract_equations
import stage05_build_embeddings
import stage06_extract_meanings
import stage07_extract_symbols
import stage08_extract_symbol_meanings
import stage09_build_relations
import stage10_export_final

from config import BUILD_REPORT, OUTPUT_DIR


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    build = {}
    wall_start = time.time()

    print("=== Stage 1: Download HTML ===")
    t0 = time.time()
    r1 = stage01_download_html.run()
    build["stage01_download"] = {**r1, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  cached={r1['cached']}  downloaded={r1['downloaded']}  failed={r1['failed']}"
        f"  ({build['stage01_download']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 2: Build Documents ===")
    t0 = time.time()
    r2 = stage02_build_documents.run()
    build["stage02_documents"] = {**r2, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r2['paper_count']}  raw_equations={r2['total_raw_equations']}"
        f"  sentences={r2['total_sentences']}"
        f"  ({build['stage02_documents']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 3: Build Chunks ===")
    t0 = time.time()
    r3 = stage03_build_chunks.run()
    build["stage03_chunks"] = {**r3, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r3['paper_count']}  chunks={r3['total_chunks']}"
        f"  ({build['stage03_chunks']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 4: Extract Reviewed Equations ===")
    t0 = time.time()
    r4 = stage04_extract_equations.run()
    build["stage04_equations"] = {**r4, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  equations={r4['total_equations']}  resolved={r4['total_resolved']}"
        f"  unresolved={r4['total_unresolved']}"
        f"  ({build['stage04_equations']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 5: Build Embeddings ===")
    t0 = time.time()
    r5 = stage05_build_embeddings.run()
    build["stage05_embeddings"] = {**r5, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r5['paper_count']}  vectors={r5['total_vectors']}"
        f"  truncations={r5['total_truncations']}  device={r5['device']}"
        f"  ({build['stage05_embeddings']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 6: Extract Equation Meanings ===")
    t0 = time.time()
    r6 = stage06_extract_meanings.run()
    build["stage06_meanings"] = {**r6, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r6['paper_count']}  found={r6['total_meanings_found']}"
        f"  empty={r6['total_meanings_empty']}"
        f"  ({build['stage06_meanings']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 7: Extract Symbols ===")
    t0 = time.time()
    r7 = stage07_extract_symbols.run()
    build["stage07_symbols"] = {**r7, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r7['paper_count']}  symbols={r7['total_symbols']}"
        f"  ({build['stage07_symbols']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 8: Extract Symbol Meanings ===")
    t0 = time.time()
    r8 = stage08_extract_symbol_meanings.run()
    build["stage08_symbol_meanings"] = {**r8, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r8['paper_count']}  definitions={r8['total_definitions_found']}"
        f"  ({build['stage08_symbol_meanings']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 9: Build Relations ===")
    t0 = time.time()
    r9 = stage09_build_relations.run()
    build["stage09_relations"] = {**r9, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r9['paper_count']}  pairs={r9['total_pairs']}"
        f"  strong={r9['total_strong']}  potential={r9['total_potential']}"
        f"  ({build['stage09_relations']['elapsed_seconds']:.1f}s)"
    )

    print("=== Stage 10: Export Final Data ===")
    t0 = time.time()
    r10 = stage10_export_final.run()
    build["stage10_export"] = {**r10, "elapsed_seconds": round(time.time() - t0, 2)}
    print(
        f"  papers={r10['paper_count']}  equations={r10['equation_count']}"
        f"  meanings={r10['meanings_found']}"
        f"  ({build['stage10_export']['elapsed_seconds']:.1f}s)"
    )

    build["total_elapsed_seconds"] = round(time.time() - wall_start, 2)

    tmp = BUILD_REPORT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(build, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(BUILD_REPORT)

    print(f"\nFinal data -> {r10['output']}")
    print(f"Build report -> {BUILD_REPORT}")
    print(f"Total: {build['total_elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
