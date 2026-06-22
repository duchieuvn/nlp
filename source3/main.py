"""Entry point: run the full embedding pipeline (stages 1–5)."""


import json
import time

import stage01_download_html
import stage02_build_documents
import stage03_build_chunks
import stage04_extract_equations
import stage05_build_embeddings

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

    build["total_elapsed_seconds"] = round(time.time() - wall_start, 2)

    tmp = BUILD_REPORT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(build, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(BUILD_REPORT)

    print(f"\nBuild report -> {BUILD_REPORT}")
    print(f"Total: {build['total_elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
