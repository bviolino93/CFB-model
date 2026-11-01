"""Does a run of games going over/under our pre-game totals predict the next
game against the closing total? (2026-11-01; result: fading it won ~55% in
2017-25 but not 2014-16; following it lost. Spread form predicted nothing.)"""
import pandas as pd, numpy as np, cfb_model
g=cfb_model.load_features()
g=g[g.margin.notna()].copy()
g["when"]=pd.to_datetime(g.start_date,utc=True)
rows=[]
for side,opp in (("home","away"),("away","home")):
    rows.append(pd.DataFrame({"game_id":g.game_id,"season":g.season,"when":g.when,"team":g[f"{side}_team"],
      "tot":g.points-g.pred_total,"side":side}))
L=pd.concat(rows).sort_values("when")
L["pre"]=L.groupby(["season","team"]).tot.transform(lambda s:s.shift().rolling(3,min_periods=2).mean())
L["n"]=L.groupby(["season","team"]).cumcount()
h=L[L.side=="home"].set_index("game_id"); a=L[L.side=="away"].set_index("game_id")
d=g.set_index("game_id").join(h[["pre","n"]]).join(a[["pre","n"]],rsuffix="_a")
d=d[(d.n>=3)&(d.n_a>=3)].dropna(subset=["pre","pre_a","close_total"])
d["sig"]=(d.pre+d.pre_a)/2      # avg pts per game their last ~3 games beat our total by
d["ou"]=d.points-d.close_total
d=d[d.ou!=0]
for thr in (6,8,10,12):
    for era,(lo,hi) in {"2014-16":(2014,2016),"2017-21":(2017,2021),"2022-25":(2022,2025)}.items():
        e=d[(d.season>=lo)&(d.season<=hi)]
        hot=e[e.sig>=thr]; cold=e[e.sig<=-thr]
        lt60=hot[hot.close_total<60]
        print(f"thr {thr} {era}: hot->under {(hot.ou<0).mean():.1%} n={len(hot)} (total<60: {(lt60.ou<0).mean():.1%} n={len(lt60)}); cold->over {(cold.ou>0).mean():.1%} n={len(cold)}")
print(d.loc[d.season==2025,"sig"].describe())
