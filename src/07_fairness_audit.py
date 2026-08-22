META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def goodness_arrow(ax, text="higher = better", loc="upper left", axis="y", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams["legend.fontsize"]
    pos = {"upper left": (0.02, 0.98), "upper right": (0.98, 0.98),
           "lower left": (0.02, 0.02), "lower right": (0.98, 0.02)}[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    arrow = "↑ " if axis == "y" else "→ "
    ax.text(pos[0], pos[1], arrow + text, transform=ax.transAxes,
            fontsize=fontsize, color=META_GREY, ha=ha, va=va)


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

fp = pd.read_parquet("data/fairness_preds.parquet")

RACE_MAP={1:"Alaska Native",2:"Am Indian",3:"Asian/PI",4:"Black",5:"White",
          6:"Asian",7:"Other single",8:"Two+ races",9:"NHOPI",13:"multi",20:"Hispanic-only(NR)",21:"Other"}
GENDER_MAP={1:"Male",2:"Female"}
AGE_MAP={1:"12-14",2:"15-17",3:"18-20",4:"21-24",5:"25-29",6:"30-34",7:"35-39",
         8:"40-44",9:"45-49",10:"50-54",11:"55-64",12:"65+"}

def boot_auc_ci(y,p,n_boot=300,seed=0,cap=100000):
    r=np.random.default_rng(seed)
    if len(y)>cap:
        idx=r.choice(len(y),cap,replace=False); y,p=y[idx],p[idx]
    n=len(y); a=[]
    for _ in range(n_boot):
        b=r.integers(0,n,n); yb=y[b]
        if yb.min()!=yb.max(): a.append(roc_auc_score(yb,p[b]))
    return round(np.percentile(a,2.5),4),round(np.percentile(a,97.5),4)

def subgroup_ci(df,col,mapping,min_n=1000):
    out=[]
    for v,g in df.groupby(col):
        if len(g)<min_n or g.y.nunique()<2: continue
        auc=roc_auc_score(g.y,g.p); lo,hi=boot_auc_ci(g.y.values,g.p.values)
        out.append({"dim":col,"group":mapping.get(int(v),f"code{int(v)}"),"n":len(g),
                    "dropout":round(g.y.mean(),3),"auc":round(auc,4),"ci_lo":lo,"ci_hi":hi,
                    "calib_gap":round(g.p.mean()-g.y.mean(),4)})
    return pd.DataFrame(out)

race=subgroup_ci(fp[fp.RACE>=0],"RACE",RACE_MAP)
sex=subgroup_ci(fp[fp.GENDER>=0],"GENDER",GENDER_MAP)
age=subgroup_ci(fp[fp.AGE>=0],"AGE",AGE_MAP)

st=[]
for v,g in fp.groupby("STFIPS"):
    if len(g)<2000 or g.y.nunique()<2: continue
    st.append({"state":int(v),"n":len(g),"auc":round(roc_auc_score(g.y,g.p),4),
               "calib_gap":round(g.p.mean()-g.y.mean(),4)})
state=pd.DataFrame(st)
pd.concat([race,sex,age],ignore_index=True).to_csv("data/p1_fairness_subgroups.csv",index=False)
state.to_csv("data/p1_fairness_states.csv",index=False)

apply_figure_style()
overall_auc=0.7913
fig, axes = plt.subplots(2,2,figsize=(11,9))

def plot_dim(ax, df_, title, sort=True):
    df2=df_.sort_values("auc") if sort else df_
    y=np.arange(len(df2))
    xerr=np.clip(np.vstack([df2.auc-df2.ci_lo, df2.ci_hi-df2.auc]),0,None)
    ax.errorbar(df2.auc,y,xerr=xerr,fmt="o",color="#1f4e79",capsize=2,ms=5)
    ax.axvline(overall_auc,color="#c0504d",ls="--",lw=1,label=f"overall {overall_auc:.3f}")
    ax.set_yticks(y); ax.set_yticklabels([f"{g} (n={n:,})" for g,n in zip(df2.group,df2.n)],fontsize=7.5)
    ax.set_xlabel("AUROC (95% CI)"); ax.set_title(title,fontsize=9.5)
    ax.legend(fontsize=7,loc="lower right")

plot_dim(axes[0,0], race, "By race/ethnicity")
plot_dim(axes[0,1], age, "By age band", sort=False)
age_ord=age.copy(); age_ord["o"]=age_ord.group.map({v:k for k,v in AGE_MAP.items()}); age_ord=age_ord.sort_values("o")
axes[0,1].clear(); plot_dim(axes[0,1], age_ord, "By age band", sort=False)

plot_dim(axes[1,0], sex, "By sex", sort=False)

ax=axes[1,1]
ax.hist(state.auc,bins=20,color="#9ab8d6",edgecolor="#1f4e79")
ax.axvline(overall_auc,color="#c0504d",ls="--",lw=1,label=f"overall {overall_auc:.3f}")
ax.axvline(state.auc.median(),color="#1f4e79",ls=":",lw=1,label=f"state median {state.auc.median():.3f}")
ax.set_xlabel("Per-state AUROC"); ax.set_ylabel("# states"); ax.set_title(f"By state (n={len(state)} states)",fontsize=9.5)
ax.legend(fontsize=7)

fig.suptitle("Subgroup fairness: discrimination by race, age, sex, state (pooled GBT)",fontsize=11,y=1.00)
fig.tight_layout()

for ax in [axes[1,0],axes[1,1]]:
    ax.xaxis.set_label_coords(0.5,-0.09)
fig.savefig("figures/p1_fairness.png",dpi=300,bbox_inches="tight")