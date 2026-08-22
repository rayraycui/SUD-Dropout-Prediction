#!/usr/bin/env python3
"""
Download the TEDS-D public-use files (2015-2020, 2022) from SAMHSA and extract
the CSVs into data/raw/. 2021 is intentionally skipped (its public-use file ships
text labels rather than the numeric codes used in every other year).

Source: SAMHSA Data Files (https://www.samhsa.gov/data/data-we-collect/teds/datafiles)
The TEDS-D discharge public-use files are distributed as ZIP bundles. SAMHSA
occasionally changes the exact download URLs; if a link 404s, browse the TEDS-D
page above and update the URL for that year.

After extraction, data/raw/ should contain (filenames vary slightly by year):
    tedsd_2015_puf.csv, tedsd_2016_puf.csv, tedsd_puf_2017.csv,
    tedsd_puf_2018.csv, tedsd_puf_2019.csv, tedsd_puf_2020.csv,
    tedsd_puf_2022.csv
"""
import os, sys, zipfile, urllib.request, glob

RAW = "data/raw"
os.makedirs(RAW, exist_ok=True)

# SAMHSA TEDS-D discharge public-use ZIP bundles (CSV format).
# NOTE: verify/refresh these URLs at the TEDS-D data page if any fail.
URLS = {
    2015: "https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/TEDS-D-2015/TEDS-D-2015-datasets/TEDS-D-2015-DS0001/TEDS-D-2015-DS0001-bundles-with-study-info/TEDS-D-2015-DS0001-bndl-data-csv_v1.zip",
    2016: "https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/TEDS-D-2016/TEDS-D-2016-datasets/TEDS-D-2016-DS0001/TEDS-D-2016-DS0001-bundles-with-study-info/TEDS-D-2016-DS0001-bndl-data-csv_v1.zip",
    2017: "https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/TEDS-D-2017/TEDS-D-2017-datasets/TEDS-D-2017-DS0001/TEDS-D-2017-DS0001-bundles-with-study-info/TEDS-D-2017-DS0001-bndl-data-csv_v1.zip",
    2018: "https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/TEDS-D-2018/TEDS-D-2018-datasets/TEDS-D-2018-DS0001/TEDS-D-2018-DS0001-bundles-with-study-info/TEDS-D-2018-DS0001-bndl-data-csv_v1.zip",
    2019: "https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/TEDS-D-2019/TEDS-D-2019-datasets/TEDS-D-2019-DS0001/TEDS-D-2019-DS0001-bundles-with-study-info/TEDS-D-2019-DS0001-bndl-data-tsv_V1.zip",
    2020: "https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/TEDS-D-2020/TEDS-D-2020-datasets/TEDS-D-2020-DS0001/TEDS-D-2020-DS0001-bundles-with-study-info/TEDS-D-2020-DS0001-bndl-data-csv_v1.zip",
    2022: "https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/TEDS-D-2022/TEDS-D-2022-datasets/TEDS-D-2022-DS0001/TEDS-D-2022-DS0001-bundles-with-study-info/TEDS-D-2022-DS0001-bndl-data-csv_v1.zip",
}

def fetch(year, url):
    zpath = os.path.join(RAW, f"tedsd_{year}.zip")
    if not os.path.exists(zpath):
        print(f"[{year}] downloading ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=120) as r, open(zpath, "wb") as f:
            f.write(r.read())
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.lower().endswith((".csv", ".tsv")):
                z.extract(name, RAW)
                print(f"[{year}] extracted {name}", flush=True)

if __name__ == "__main__":
    for y, u in URLS.items():
        try:
            fetch(y, u)
        except Exception as e:
            print(f"[{y}] FAILED: {e}\n  -> refresh the URL from the TEDS-D data page.", file=sys.stderr)
    print("\nData files present in data/raw/:")
    for p in sorted(glob.glob(os.path.join(RAW, "*.csv")) + glob.glob(os.path.join(RAW, "*.tsv"))):
        print(" ", p, f"({os.path.getsize(p)//1_000_000} MB)")
