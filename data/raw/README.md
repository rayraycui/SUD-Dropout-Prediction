# data/raw/

TEDS-D public-use source files land here (see `src/00_download_teds.py`).
They are **not** committed: SAMHSA distributes them and this repository does not
redistribute them.

`MANIFEST.csv` pins the identity of each source file as used in this analysis:

| column | meaning |
|---|---|
| `year` | discharge year (2021 is excluded by design) |
| `bundle`, `url` | the SAMHSA ZIP downloaded, and where from |
| `bundle_bytes`, `bundle_sha256` | identity of the ZIP |
| `data_file` | the CSV extracted from that ZIP |
| `data_bytes`, `data_sha256` | identity of the extracted file |
| `delimiter`, `n_columns`, `n_data_rows` | parsed shape (rows exclude the header) |
| `verified_utc` | date the URL and bytes were last confirmed against the live site |

After running `python src/00_download_teds.py`, compare your downloads against this
file. SAMHSA can reissue a public-use file without changing its URL, so matching
SHA-256 is the only way to confirm you have the vintage the published numbers used.

Two traps this manifest records:

1. **Bundle names do not indicate format.** The 2015-2019 bundles are named
   `...-tsv...zip`, but every extracted file is comma-separated. Each header row
   carries 75-77 commas and zero tabs.
2. **Filenames are inconsistent across years** (`tedsd_2015_puf.csv` but
   `tedsd_puf_2017.csv`), which is why `src/01_build_analysis_table.py` maps them
   explicitly rather than globbing.

The seven files hold 10,713,968 data rows; after restricting to a resolvable outcome and a mappable primary substance, and excluding SUB1 code 19 (other drugs, 37,578 episodes), the analytic sample is 6,679,648 episodes across 17 substances.
