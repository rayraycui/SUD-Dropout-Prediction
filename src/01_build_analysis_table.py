import pandas as pd
import numpy as np
import os

# These are the files src/00_download_teds.py extracts (it writes to data/raw/).
# 2019 is distributed as a TAB-separated file: SAMHSA ships that year as
# "...DS0001-bndl-data-tsv_V1.zip" while every other year is "-csv_v1.zip",
# so read_year() below picks the separator from the extension.
paths_all = {
    2015: 'data/raw/tedsd_2015_puf.csv',
    2016: 'data/raw/tedsd_2016_puf.csv',
    2017: 'data/raw/tedsd_puf_2017.csv',
    2018: 'data/raw/tedsd_puf_2018.csv',
    2019: 'data/raw/tedsd_puf_2019.tsv',
    2020: 'data/raw/tedsd_puf_2020.csv',
    2022: 'data/raw/tedsd_puf_2022.csv',
}

need = ["REASON","SUB1","SUB2","SUB3","ROUTE1","FREQ1","FRSTUSE1","SERVICES","DSMCRIT",
        "NOPRIOR","PSOURCE","AGE","GENDER","RACE","EDUC","EMPLOY","STFIPS"]
feat15 = need[2:]

sub1_all = {
    2:"Alcohol", 3:"Cocaine", 4:"Cannabis", 5:"Heroin", 6:"Non_rx_methadone", 7:"Other_opioids",
    8:"PCP", 9:"Other_hallucinogens", 10:"Methamphetamine", 11:"Other_amphetamines",
    12:"Other_stimulants", 13:"Benzodiazepines", 14:"Other_sedatives", 15:"Other_tranquilizers",
    16:"Barbiturates", 17:"Other_sedatives_hypnotics", 18:"Inhalants", 19:"OTC", 20:"Other_drugs"
}
big6 = {"Alcohol","Cocaine","Cannabis","Heroin","Other_opioids","Methamphetamine"}

def read_year(p):
    sep = "\t" if p.endswith(".tsv") else ","
    try:
        return pd.read_csv(p, usecols=need, sep=sep)
    except UnicodeDecodeError:
        return pd.read_csv(p, usecols=need, sep=sep, encoding="latin-1")

d = pd.concat([read_year(p).assign(YEAR=y) for y, p in sorted(paths_all.items())], ignore_index=True)
d = d[d["REASON"].isin([1,2])].copy()
d["y"] = (d["REASON"]==2).astype(int)
d["subname"] = d["SUB1"].map(sub1_all)
d = d[d["subname"].notna()].copy()
d["is_big6"] = d["subname"].isin(big6)
for c in feat15:
    d[c] = pd.to_numeric(d[c], errors="coerce").replace(-9, np.nan)
d = d[["YEAR","y","SUB1","subname","is_big6"]+feat15].reset_index(drop=True)
d.to_parquet("data/teds_d_analysis_2015_2022.parquet", index=False)
print("N=", f"{len(d):,}", "dropout=", round(d.y.mean(),3), "MB=", os.path.getsize("data/teds_d_analysis_2015_2022.parquet")//1_000_000)

# Class-size summary table (per substance: n, dropout rate, big-6 membership)
cs = (d.groupby("subname")
        .agg(n=("y","size"), dropout=("y","mean"), is_big6=("is_big6","first"))
        .reset_index().sort_values("n", ascending=False))
cs["dropout"] = cs["dropout"].round(3)
cs["group"] = np.where(cs["is_big6"], "BIG6", "small")
cs.to_csv("data/teds_d_class_sizes_2015_2022.csv", index=False)
print("class-size summary rows:", len(cs))