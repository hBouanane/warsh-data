"""Pull the surah-2 rows (no audio) from Haitam03/warsh-segments.

Column projection only: the audio column is what makes these shards large, and
there is no room on disk for it.
"""

import json
import time

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

COLUMNS = [
    "segment_id", "reciter_slug", "surah_number", "segment_index", "asr",
    "text_warsh", "alignment_score", "word_start", "word_end", "ayah_start",
    "ayah_end", "flagged", "duration_seconds",
]

fs = HfFileSystem()
files = sorted(fs.glob("datasets/Haitam03/warsh-segments/data/*.parquet"))
print(f"{len(files)} shards", flush=True)

rows = []
for index, path in enumerate(files):
    started = time.time()
    with fs.open(path, "rb") as handle:
        table = pq.ParquetFile(handle).read(columns=COLUMNS)
    found = [r for r in table.to_pylist() if r["surah_number"] == 2]
    rows.extend(found)
    print(f"shard {index}: {table.num_rows} rows, {len(found)} in surah 2, "
          f"{time.time() - started:.0f}s, running total {len(rows)}", flush=True)
    if len(rows) >= 600:
        print("enough collected, stopping early", flush=True)
        break

rows.sort(key=lambda r: (r["reciter_slug"], r["segment_index"]))
with open("baqara_rows.json", "w", encoding="utf-8") as fh:
    json.dump(rows, fh, ensure_ascii=False)
print(f"saved {len(rows)} rows", flush=True)
