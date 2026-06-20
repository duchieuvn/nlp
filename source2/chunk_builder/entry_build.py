from config import CHUNKS_DIR, STRUCTURED_PAPERS_DIR
from chunk_io import build_all_chunk_files


def main() -> None:
	paper_count, chunk_count = build_all_chunk_files(
		STRUCTURED_PAPERS_DIR,
		CHUNKS_DIR,
	)
	print(f"Wrote {chunk_count} chunks for {paper_count} papers to {CHUNKS_DIR}")


if __name__ == "__main__":
	main()
