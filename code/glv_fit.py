# -*- coding: utf-8 -*-
"""
gLV 模型拟合（CFU 尺度，ln N 形式求解）
=========================================

复现方法（在项目根目录运行）:
    pip install -r code/requirements.txt
    python code/glv_fit.py              # 完整拟合（6 进程并行），输出到 results/
    python code/glv_fit.py --serial     # 串行运行（内存紧张时使用，结果相同）
    python code/glv_fit.py --figures-only # 只根据已有结果重画图
    python code/glv_fit.py --selftest   # 只跑正确性自检（解析解对比、α=0 退化、合成数据回收）

流程:
  1. 对复孔的 log10 值求平均 -> N = 10**mean (CFU)；6h 记为 t=0，6h 实测值作为 N0
  2. 单培养: Logistic 拟合 r, K
  3. 固定 r, K，拟合 gLV 相互作用系数 α:
        d ln N_i / dt = r_i * (1 - (N_i + sum_j α_ij N_j) / K_i)
     方法 A: 只用 D1-D6-L 三菌数据
     方法 B: D1-L、D6-L、D1-D6(补充数据)、D1-D6-L 联合拟合
  4. 预测值取 log10，与实测 log10 均值比较求残差（检出限 1.0 单独处理）
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # 每个进程单线程 BLAS，降低内存占用
import csv
import functools
import glob
import json
import os
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ----------------------------------------------------------------------------- 配置
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "results")

TIMES_H = np.array([6.0, 24.0, 48.0, 72.0])
T_MODEL = TIMES_H - 6.0          # 6h -> t = 0
LOD = 1.0                        # 检出限 (log10 CFU)
CENSOR_LOD = True                # True: 实测在检出限且预测 <= LOD -> 残差 0
ALPHA_BOUND = 10000.0            # α 搜索范围 ±10000
BOUND_FLAG = 0.95 * ALPHA_BOUND  # |α| 超过该值视为"顶到边界"
N_STARTS = 30                    # 多初值次数
SEED = 20261104
Y_CAP = np.log(1e15)             # ln N 上限截断（防止溢出）
Y_LOW = np.log(1e-6)             # ln N 下限（低于 1e-6 CFU 视为已灭绝）
RATE_CAP = 20.0                  # |d lnN/dt| 平滑上限 (h^-1)，只在 |α| 极大的非生理区域生效，避免刚性/溢出
PENALTY = 50.0                   # 积分失败时的残差
MAX_RHS = 4000                   # 单次 ODE 求解允许的最大右端函数调用次数（正常约 100–400 次）；
                                 # 超过说明处于极端刚性的非生理参数区，直接判为失败，防止卡死

SPECIES_ALPHA = ["D1<-D6", "D1<-L", "D6<-D1", "D6<-L", "L<-D1", "L<-D6"]
# 每个 α 在 3x3 矩阵 (顺序 D1, D6, L) 中的位置 (i 被 j 影响)
ALPHA_POS = {"D1<-D6": (0, 1), "D1<-L": (0, 2), "D6<-D1": (1, 0),
             "D6<-L": (1, 2), "L<-D1": (2, 0), "L<-D6": (2, 1)}


# ----------------------------------------------------------------------------- 数据
def _stats(rows):
    """rows: list(时间) of list(复孔 log 值) -> 均值、标准差、是否删失、是否含部分检出限值"""
    a = np.array(rows, dtype=float)
    mean = a.mean(axis=1)
    sd = a.std(axis=1, ddof=1)
    cens = mean <= LOD + 1e-9
    partial = (np.isclose(a, LOD).any(axis=1)) & ~cens
    return dict(mean=mean, sd=sd, cens=cens, partial=partial, n=a.shape[1])


def load_main_file(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = rows[0][1:]
    body = {r[0].strip(): [float(x) for x in r[1:]] for r in rows[1:] if r and r[0].strip()}
    tkeys = ["6h", "24h", "48h", "72h"]
    mat = np.array([body[k] for k in tkeys])  # 4 x 60
    # 按表头分组（同名列连续出现）
    groups, order = {}, []
    for idx, h in enumerate(header):
        h = h.strip()
        if h not in groups:
            groups[h] = []
            order.append(h)
        groups[h].append(idx)
    name = os.path.basename(path)
    m = re.search(r"1-6-\s*(\d+)\s+(\d+)\.csv", name)
    strain, grp = m.group(1), m.group(2)
    Lname = f"L{strain}"

    def sl(h, k, n=6):
        cols = groups[h][k * n:(k + 1) * n]
        return _stats(mat[:, cols].tolist())

    ds = dict(file=name, strain=Lname, group=grp, key=f"{Lname}_{grp}")
    ds["mono"] = {"D1": sl("D1", 0), "D6": sl("D6", 0), "L": sl(Lname, 0)}
    ds["D1L"] = {"D1": sl(f"D1-{Lname}", 0), "L": sl(f"D1-{Lname}", 1)}
    ds["D6L"] = {"D6": sl(f"D6-{Lname}", 0), "L": sl(f"D6-{Lname}", 1)}
    ds["D1D6L"] = {"D1": sl(f"D1-D6-{Lname}", 0), "D6": sl(f"D1-D6-{Lname}", 1),
                   "L": sl(f"D1-D6-{Lname}", 2)}
    return ds


def load_d1d6(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for sp in ["D1", "D6"]:
        rr = [r for r in rows if r["菌种"].strip() == sp]
        rr.sort(key=lambda r: float(r["时间"].replace("h", "")))
        out[sp] = _stats([[float(r[f"log10值{k}"]) for k in (1, 2, 3)] for r in rr])
    return out


def load_all():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "paper 7 fig 5*.csv")))
    dsets = [load_main_file(p) for p in files]
    d1d6 = load_d1d6(os.path.join(DATA_DIR, "3_sepcies.csv"))
    return dsets, d1d6


# ----------------------------------------------------------------------------- 模型
class _TooStiff(Exception):
    pass


def simulate(N0, r, K, A, t_eval=T_MODEL):
    """在 y = ln N 上求解 gLV。A: 交互矩阵（对角线=1）。返回 log10 N (n_sp x n_t)，失败返回 None"""
    r = np.asarray(r, float); K = np.asarray(K, float); A = np.asarray(A, float)
    y0 = np.log(np.asarray(N0, float))

    ncall = [0]

    def rhs(t, y):
        ncall[0] += 1
        if ncall[0] > MAX_RHS:
            raise _TooStiff()
        N = np.exp(np.clip(y, Y_LOW, Y_CAP))
        g = r * (1.0 - (A @ N) / K)
        u = g / RATE_CAP
        return g / (1.0 + u ** 4) ** 0.25          # 平滑饱和：|g|<<RATE_CAP 时与 g 相同（相对误差 <1e-7）

    def jac(t, y):
        yc = np.clip(y, Y_LOW, Y_CAP)
        N = np.exp(yc)
        dN = np.where(yc == y, N, 0.0)
        g = r * (1.0 - (A @ N) / K)
        u = g / RATE_CAP
        ds = (1.0 + u ** 4) ** -1.25
        return (ds * -(r / K))[:, None] * A * dN[None, :]

    import warnings
    for method in ("LSODA", "Radau"):            # LSODA 失败时改用 Radau 重算
        ncall[0] = 0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method=method,
                                jac=jac, rtol=1e-8, atol=1e-10)
        except Exception:
            continue
        if sol.success and sol.y.shape[1] == len(t_eval) and np.all(np.isfinite(sol.y)):
            # 预测下限 1e-6 CFU (log10=-6)：低于此值视为灭绝，避免残差被极端负值主导
            return np.maximum(sol.y / np.log(10.0), Y_LOW / np.log(10.0))
    return None


def logistic_analytic(N0, r, K, t):
    return K / (1.0 + (K - N0) / N0 * np.exp(-r * t))


def residuals_log(pred_log, stats_list, skip_t0=True):
    """pred_log: n_sp x n_t; stats_list: 各物种实测统计。返回残差向量（跳过 t0，因为 t0 即初值）"""
    res = []
    i0 = 1 if skip_t0 else 0
    for s, st in enumerate(stats_list):
        for k in range(i0, len(T_MODEL)):
            p, o = pred_log[s, k], st["mean"][k]
            if st["cens"][k] and CENSOR_LOD:
                res.append(max(p - LOD, 0.0))
            elif st["cens"][k]:
                res.append(p - LOD)
            else:
                res.append(p - o)
    return np.array(res)


def n0_of(st):
    return 10.0 ** st["mean"][0]


# ----------------------------------------------------------------------------- 第一步：单培养
def fit_logistic(st):
    N0 = n0_of(st)
    best = None
    for r0 in (0.05, 0.2, 0.6):
        for lk0 in (np.max(st["mean"]), np.max(st["mean"]) + 1.0):
            def f(p):
                pred = simulate([N0], [p[0]], [10 ** p[1]], [[1.0]])
                if pred is None:
                    return np.full(3, PENALTY)
                return residuals_log(pred, [st])
            lk0c = float(np.clip(lk0, 1.01, 11.9))
            res = least_squares(f, [r0, lk0c], bounds=([1e-4, 1.0], [5.0, 12.0]), method="trf")
            if best is None or res.cost < best.cost:
                best = res
    r, lk = best.x
    return dict(r=float(r), log10K=float(lk), K=float(10 ** lk), N0=float(N0),
                log10N0=float(st["mean"][0]), ssr=float(2 * best.cost))


# ----------------------------------------------------------------------------- 第二步：α
def build_A(alpha_vec, names=SPECIES_ALPHA):
    A = np.eye(3)
    for v, nm in zip(alpha_vec, names):
        A[ALPHA_POS[nm]] = v
    return A


def systems_for(ds, d1d6, method):
    """返回参与拟合的培养体系列表: (标签, 物种下标, 实测统计列表)"""
    tri = ("D1-D6-L", [0, 1, 2], [ds["D1D6L"]["D1"], ds["D1D6L"]["D6"], ds["D1D6L"]["L"]])
    if method == "A":
        return [tri]
    return [("D1-L", [0, 2], [ds["D1L"]["D1"], ds["D1L"]["L"]]),
            ("D6-L", [1, 2], [ds["D6L"]["D6"], ds["D6L"]["L"]]),
            ("D1-D6", [0, 1], [d1d6["D1"], d1d6["D6"]]),
            tri]


def predict_system(alpha_vec, rK, idx, stats_list):
    A = build_A(alpha_vec)[np.ix_(idx, idx)]
    r = rK["r"][idx]; K = rK["K"][idx]
    N0 = [n0_of(s) for s in stats_list]
    return simulate(N0, r, K, A)


def alpha_objective(alpha_vec, rK, systems):
    out = []
    for _, idx, sts in systems:
        pred = predict_system(alpha_vec, rK, idx, sts)
        n = len(idx) * (len(T_MODEL) - 1)
        out.append(np.full(n, PENALTY) if pred is None else residuals_log(pred, sts))
    return np.concatenate(out)


def start_points(n, dim, rng, extra=()):
    pts = [np.zeros(dim)] + [np.asarray(e, float) for e in extra]
    while len(pts) < n:
        mag = 10 ** rng.uniform(-2, 4, dim)
        sign = rng.choice([-1.0, 1.0], dim)
        pts.append(np.clip(mag * sign, -0.99 * ALPHA_BOUND, 0.99 * ALPHA_BOUND))
    return pts


def multistart_fit(rK, systems, names, fixed=None, seed=SEED, n_starts=N_STARTS, extra=()):
    """拟合 names 中的 α（其余在 fixed 中给定为常数）。返回最优解及多初值统计"""
    fixed = fixed or {}
    rng = np.random.default_rng(seed)

    def full(x):
        d = dict(fixed); d.update(zip(names, x))
        return [d.get(nm, 0.0) for nm in SPECIES_ALPHA]

    f = lambda x: alpha_objective(full(x), rK, systems)
    runs = []
    for x0 in start_points(n_starts, len(names), rng, extra):
        try:
            res = least_squares(f, x0, bounds=(-ALPHA_BOUND, ALPHA_BOUND), method="trf",
                                x_scale="jac", max_nfev=150)
            runs.append((2 * res.cost, res.x.copy()))
        except Exception:
            continue
    runs.sort(key=lambda t: t[0])
    best_ssr, best_x = runs[0]
    near = [x for s, x in runs if s <= best_ssr * 1.01 + 1e-10]
    near = np.array(near)
    resid = f(best_x)
    return dict(names=names, alpha=dict(zip(names, best_x.tolist())), ssr=float(best_ssr),
                n_points=int(len(resid)), rmse=float(np.sqrt(np.mean(resid ** 2))),
                n_near=int(len(near)), n_runs=len(runs),
                near_min=dict(zip(names, near.min(0).tolist())),
                near_max=dict(zip(names, near.max(0).tolist())),
                at_bound=[nm for nm, v in zip(names, best_x) if abs(v) >= BOUND_FLAG],
                full=full(best_x))


def _fit_task(args):
    ds, d1d6, rK_d, d1d6_alpha, m = args
    import time
    t0 = time.time()
    rK = dict(r=np.array([rK_d["D1"]["r"], rK_d["D6"]["r"], rK_d[ds["key"]]["r"]]),
              K=np.array([rK_d["D1"]["K"], rK_d["D6"]["K"], rK_d[ds["key"]]["K"]]))
    extra = [[d1d6_alpha["D1<-D6"], 0, d1d6_alpha["D6<-D1"], 0, 0, 0]]
    fr = multistart_fit(rK, systems_for(ds, d1d6, m), SPECIES_ALPHA,
                        seed=SEED + zlib.crc32((ds["key"] + m).encode()) % 1000, extra=extra)
    fr["elapsed_s"] = round(time.time() - t0, 1)
    return ds["key"], m, fr


# ----------------------------------------------------------------------------- 输出
def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dsets, d1d6 = load_all()
    print(f"读取 {len(dsets)} 个数据集 + D1-D6 补充数据")

    # ---- 第一步：r, K
    rK = {"D1": fit_logistic(dsets[0]["mono"]["D1"]), "D6": fit_logistic(dsets[0]["mono"]["D6"])}
    for ds in dsets:
        rK[ds["key"]] = fit_logistic(ds["mono"]["L"])
    for k, v in rK.items():
        print(f"  {k:10s} r={v['r']:.4f}  log10K={v['log10K']:.3f}  SSR={v['ssr']:.4f}")

    # ---- D1-D6 双菌基线
    rK2 = dict(r=np.array([rK["D1"]["r"], rK["D6"]["r"], 1.0]),
               K=np.array([rK["D1"]["K"], rK["D6"]["K"], 1.0]))
    sys_d1d6 = [("D1-D6", [0, 1], [d1d6["D1"], d1d6["D6"]])]
    base = multistart_fit(rK2, sys_d1d6, ["D1<-D6", "D6<-D1"], seed=SEED)
    print(f"  D1-D6 基线: {base['alpha']}  SSR={base['ssr']:.4f}")

    # ---- 第二步：α（8 个数据集并行）
    jobs = [(ds, d1d6, rK, base["alpha"], m) for m in ("B", "A") for ds in dsets]
    fits = {ds["key"]: {} for ds in dsets}

    def _done(res, i):
        k, m, fr = res
        fits[k][m] = fr
        print(f"  [{i}/{len(jobs)}] {k} 方法{m}: SSR={fr['ssr']:.4f}  RMSE={fr['rmse']:.3f}  "
              f"顶边界={fr['at_bound'] or '无'}  用时 {fr['elapsed_s']}s")

    if "--serial" in sys.argv:
        for i, j in enumerate(jobs, 1):
            _done(_fit_task(j), i)
    else:
        from concurrent.futures import as_completed
        with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as ex:
            futs = [ex.submit(_fit_task, j) for j in jobs]
            for i, fu in enumerate(as_completed(futs), 1):
                _done(fu.result(), i)

    # ---- 预测与残差
    pred_rows = []
    curves = {}
    tt = np.linspace(0, 66, 133)

    def add_rows(dkey, method, culture, idx, sts, alpha_full, rKv):
        A = build_A(alpha_full)[np.ix_(idx, idx)]
        N0 = [n0_of(s) for s in sts]
        p = simulate(N0, rKv["r"][idx], rKv["K"][idx], A)
        pc = simulate(N0, rKv["r"][idx], rKv["K"][idx], A, t_eval=tt)
        names = [["D1", "D6", "L"][i] for i in idx]
        curves[(dkey, method, culture)] = dict(species=names, t=tt.tolist(),
                                              log10N=None if pc is None else pc.tolist())
        for s, st in enumerate(sts):
            for k in range(len(T_MODEL)):
                obs, pr = st["mean"][k], p[s, k]
                if k == 0:
                    res = 0.0
                elif st["cens"][k] and CENSOR_LOD:
                    res = max(pr - LOD, 0.0)
                elif st["cens"][k]:
                    res = pr - LOD
                else:
                    res = pr - obs
                pred_rows.append([dkey, method, culture, names[s], int(TIMES_H[k]),
                                  round(obs, 4), round(st["sd"][k], 4), int(st["cens"][k]),
                                  int(st["partial"][k]), round(10 ** obs, 2),
                                  round(float(pr), 4), round(float(10 ** pr), 2),
                                  round(float(res), 4), "N0(初值)" if k == 0 else ""])

    # 单培养
    for key, st, name in [("D1", dsets[0]["mono"]["D1"], "D1"), ("D6", dsets[0]["mono"]["D6"], "D6")] + \
            [(ds["key"], ds["mono"]["L"], "L") for ds in dsets]:
        v = rK[key]
        rKv = dict(r=np.array([v["r"]] * 3), K=np.array([v["K"]] * 3))
        sp_i = {"D1": 0, "D6": 1, "L": 2}[name]
        add_rows(key, "mono", f"{name} 单培养", [sp_i], [st], [0] * 6, rKv)
    add_rows("D1-D6补充", "baseline", "D1-D6", [0, 1], [d1d6["D1"], d1d6["D6"]], base["full"], rK2)

    for ds in dsets:
        key = ds["key"]
        rKv = dict(r=np.array([rK["D1"]["r"], rK["D6"]["r"], rK[key]["r"]]),
                   K=np.array([rK["D1"]["K"], rK["D6"]["K"], rK[key]["K"]]))
        for m in ("A", "B"):
            al = fits[key][m]["full"]
            for label, idx, sts in systems_for(ds, d1d6, "B"):
                add_rows(key, m, label, idx, sts, al, rKv)

    write_csv(os.path.join(OUT_DIR, "predictions_residuals.csv"),
              ["dataset", "method", "culture", "species", "time_h", "obs_log10_mean", "obs_log10_sd",
               "censored_LOD", "partial_LOD_reps", "obs_CFU", "pred_log10", "pred_CFU",
               "residual_log10", "note"], pred_rows)

    write_csv(os.path.join(OUT_DIR, "params_rK.csv"),
              ["species_or_dataset", "r_per_h", "K_CFU", "log10K", "N0_CFU", "log10N0", "SSR_log10"],
              [[k, round(v["r"], 5), f"{v['K']:.4e}", round(v["log10K"], 4), f"{v['N0']:.4e}",
                round(v["log10N0"], 4), round(v["ssr"], 5)] for k, v in rK.items()])

    for m in ("A", "B"):
        rows = []
        for ds in dsets:
            fr = fits[ds["key"]][m]
            rows.append([ds["key"], ds["strain"], ds["group"]] +
                        [round(fr["alpha"][nm], 4) for nm in SPECIES_ALPHA] +
                        [round(fr["ssr"], 5), fr["n_points"], round(fr["rmse"], 4),
                         f"{fr['n_near']}/{fr['n_runs']}", ";".join(fr["at_bound"])] +
                        [f"[{fr['near_min'][nm]:.3g}, {fr['near_max'][nm]:.3g}]" for nm in SPECIES_ALPHA])
        write_csv(os.path.join(OUT_DIR, f"alpha_{m}.csv"),
                  ["dataset", "L_strain", "group"] + SPECIES_ALPHA +
                  ["SSR", "n_points", "RMSE_log10", "n_starts_within_1pct_of_best", "alpha_at_bound"] +
                  [f"range_{nm}" for nm in SPECIES_ALPHA], rows)

    summary = dict(config=dict(LOD=LOD, CENSOR_LOD=CENSOR_LOD, ALPHA_BOUND=ALPHA_BOUND,
                               N_STARTS=N_STARTS, SEED=SEED),
                   rK=rK, baseline_D1D6=base, fits=fits,
                   partial_lod=[[ds["key"], c, sp, int(TIMES_H[k])]
                                for ds in dsets for c in ("mono", "D1L", "D6L", "D1D6L")
                                for sp, st in ds[c].items() for k in range(4) if st["partial"][k]])
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT_DIR, "curves.json"), "w", encoding="utf-8") as f:
        json.dump({"|".join(k): v for k, v in curves.items()}, f)

    worked_example(dsets[0], rK, fits[dsets[0]["key"]]["B"])
    make_figures(dsets, d1d6, rK, base, fits, curves)
    print("完成，结果已写入", OUT_DIR)


def worked_example(ds, rK, fit):
    """把第一个数据集 t=0 时刻的数值代入方程，输出完整计算过程"""
    key = ds["key"]
    sts = [ds["D1D6L"]["D1"], ds["D1D6L"]["D6"], ds["D1D6L"]["L"]]
    N = np.array([n0_of(s) for s in sts])
    r = np.array([rK["D1"]["r"], rK["D6"]["r"], rK[key]["r"]])
    K = np.array([rK["D1"]["K"], rK["D6"]["K"], rK[key]["K"]])
    A = build_A(fit["full"])
    lines = [f"# 公式代入示例（数据集 {key}，方法 B，三菌 D1-D6-{ds['strain']}，t=0 即 6h）", ""]
    nm = ["D1", "D6", "L"]
    for i in range(3):
        terms = " + ".join([f"{N[i]:.4g}"] + [f"({A[i, j]:.4g})×{N[j]:.4g}" for j in range(3) if j != i])
        tot = A[i] @ N
        g = r[i] * (1 - tot / K[i])
        lines.append(f"- {nm[i]}: d lnN/dt = {r[i]:.4f} × (1 − [{terms}] / {K[i]:.4g})"
                     f" = {r[i]:.4f} × (1 − {tot:.4g}/{K[i]:.4g}) = {g:.4f} h⁻¹")
        lines.append(f"  - 对应 dN/dt = N × d lnN/dt = {N[i]:.4g} × {g:.4f} = {N[i] * g:.4g} CFU/h")
    with open(os.path.join(OUT_DIR, "worked_example.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- 图
COL = {"D1": "#2a78d6", "D6": "#eb6834", "L": "#1baf7a"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def make_figures(dsets, d1d6, rK, base, fits, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figdir = os.path.join(OUT_DIR, "figures"); os.makedirs(figdir, exist_ok=True)

    def panel(ax, title, items, ckeys):
        _style(ax)
        for sp, st in items:
            ax.errorbar(TIMES_H, st["mean"], yerr=st["sd"], fmt="o", ms=5, color=COL[sp],
                        mec=SURF, mew=1, capsize=2, lw=1, zorder=3)
        for ck, ls in ckeys:
            c = curves.get(ck)
            if not c or c["log10N"] is None:
                continue
            for s, sp in enumerate(c["species"]):
                ax.plot(np.array(c["t"]) + 6, c["log10N"][s], ls, color=COL[sp], lw=2, zorder=2)
        ax.axhline(LOD, color=MUTED, lw=1, ls=":", zorder=1)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xticks(TIMES_H); ax.set_ylim(0, 11)

    for ds in dsets:
        key = ds["key"]; L = ds["strain"]
        fig, axs = plt.subplots(1, 5, figsize=(17, 3.6), sharey=True)
        fig.patch.set_facecolor(SURF)
        panel(axs[0], "单培养 (Logistic)",
              [("D1", ds["mono"]["D1"]), ("D6", ds["mono"]["D6"]), ("L", ds["mono"]["L"])],
              [(("D1", "mono", "D1 单培养"), "-"), (("D6", "mono", "D6 单培养"), "-"),
               ((key, "mono", "L 单培养"), "-")])
        panel(axs[1], f"D1 + {L}", [("D1", ds["D1L"]["D1"]), ("L", ds["D1L"]["L"])],
              [((key, "B", "D1-L"), "-"), ((key, "A", "D1-L"), "--")])
        panel(axs[2], f"D6 + {L}", [("D6", ds["D6L"]["D6"]), ("L", ds["D6L"]["L"])],
              [((key, "B", "D6-L"), "-"), ((key, "A", "D6-L"), "--")])
        panel(axs[3], "D1 + D6（补充数据）", [("D1", d1d6["D1"]), ("D6", d1d6["D6"])],
              [((key, "B", "D1-D6"), "-"), ((key, "A", "D1-D6"), "--")])
        panel(axs[4], f"D1 + D6 + {L}",
              [("D1", ds["D1D6L"]["D1"]), ("D6", ds["D1D6L"]["D6"]), ("L", ds["D1D6L"]["L"])],
              [((key, "B", "D1-D6-L"), "-"), ((key, "A", "D1-D6-L"), "--")])
        axs[0].set_ylabel("log10 CFU", color=INK2)
        for a in axs:
            a.set_xlabel("时间 (h)", color=INK2, fontsize=8)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], color=COL[s], marker="o", lw=2, label=s if s != "L" else L) for s in COL]
        h += [Line2D([], [], color=INK2, lw=2, label="方法 B（联合拟合）"),
              Line2D([], [], color=INK2, lw=2, ls="--", label="方法 A（仅三菌）"),
              Line2D([], [], color=MUTED, lw=1, ls=":", label="检出限 1.0")]
        fig.legend(handles=h, loc="upper center", ncol=6, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"{key}：实测（点 ± SD）与 gLV 模型（线）", y=1.09, fontsize=11, color=INK)
        fig.savefig(os.path.join(figdir, f"fit_{key}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # α 热图：红=抑制(α>0)，蓝=促进(α<0)，灰色中点
    cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#8f1f1f"])
    norm = SymLogNorm(linthresh=1.0, vmin=-ALPHA_BOUND, vmax=ALPHA_BOUND, base=10)
    for m in ("A", "B"):
        fig, axs = plt.subplots(2, 4, figsize=(15, 7.4))
        fig.patch.set_facecolor(SURF)
        for ax, ds in zip(axs.ravel(), dsets):
            A = build_A(fits[ds["key"]][m]["full"]); M = A.copy(); np.fill_diagonal(M, np.nan)
            ax.imshow(M, cmap=cmap, norm=norm)
            labs = ["D1", "D6", ds["strain"]]
            ax.set_xticks(range(3)); ax.set_yticks(range(3))
            ax.set_xticklabels(labs, fontsize=8, color=INK2); ax.set_yticklabels(labs, fontsize=8, color=INK2)
            ax.set_xticks(np.arange(-.5, 3), minor=True); ax.set_yticks(np.arange(-.5, 3), minor=True)
            ax.grid(which="minor", color=SURF, lw=2); ax.tick_params(which="minor", length=0)
            for i in range(3):
                for j in range(3):
                    if i == j:
                        ax.text(j, i, "—", ha="center", va="center", color=MUTED); continue
                    v = A[i, j]
                    star = "*" if abs(v) >= BOUND_FLAG else ""
                    dark = abs(v) > 300
                    ax.text(j, i, f"{v:.3g}{star}", ha="center", va="center", fontsize=8,
                            color="#ffffff" if dark else INK)
            ax.set_title(f"{ds['key']}  (RMSE={fits[ds['key']][m]['rmse']:.2f})", fontsize=9, color=INK)
            ax.set_xlabel("j（施加影响）", fontsize=7, color=MUTED)
            if ax in axs[:, 0]:
                ax.set_ylabel("i（受影响）", fontsize=7, color=MUTED)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axs, shrink=0.8, pad=0.02)
        cb.set_label("α_ij（symlog 色标；>0 抑制，<0 促进）", color=INK2)
        cb.set_ticks([-1e4, -1e2, -1, 0, 1, 1e2, 1e4])
        fig.suptitle(f"相互作用系数 α_ij — 方法 {m}（{'仅三菌数据' if m == 'A' else '双菌+三菌联合'}）"
                     f"；* = 顶到 ±{ALPHA_BOUND:g} 边界", fontsize=11, color=INK)
        fig.savefig(os.path.join(figdir, f"alpha_heatmap_{m}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ----------------------------------------------------------------------------- 自检
def selftest():
    t = T_MODEL
    # 1) 单物种 ODE vs 解析解
    N0, r, K = 1e2, 0.3, 1e9
    num = 10 ** simulate([N0], [r], [K], [[1.0]])[0]
    ana = logistic_analytic(N0, r, K, t)
    err = np.max(np.abs(np.log10(num) - np.log10(ana)))
    print(f"[1] Logistic 解析解对比: 最大 log10 误差 = {err:.2e}", "OK" if err < 1e-5 else "FAIL")
    # 2) α = 0 退化为独立 Logistic
    N0s, rs, Ks = [1e5, 1e4, 10], [0.25, 0.3, 0.1], [1e8, 1e9, 1e3]
    num = 10 ** simulate(N0s, rs, Ks, np.eye(3))
    ana = np.array([logistic_analytic(a, b, c, t) for a, b, c in zip(N0s, rs, Ks)])
    err = np.max(np.abs(np.log10(num) - np.log10(ana)))
    print(f"[2] α=0 退化检查: 最大 log10 误差 = {err:.2e}", "OK" if err < 1e-5 else "FAIL")
    # 3) 合成数据回收 α（联合体系，信息量充足）
    true = np.array([0.005, 5.0, 20.0, -300.0, 0.05, 0.02])
    rK = dict(r=np.array([0.25, 0.3, 0.2]), K=np.array([1e7, 1e9, 1e6]))
    def mk(idx, N0):
        p = simulate(N0, rK["r"][idx], rK["K"][idx], build_A(true)[np.ix_(idx, idx)])
        return [dict(mean=p[s], sd=np.zeros(4), cens=np.zeros(4, bool), partial=np.zeros(4, bool))
                for s in range(len(idx))]
    systems = [("D1-L", [0, 2], mk([0, 2], [1e5, 1e3])), ("D6-L", [1, 2], mk([1, 2], [1e5, 1e3])),
               ("D1-D6", [0, 1], mk([0, 1], [1e4, 1e4])), ("tri", [0, 1, 2], mk([0, 1, 2], [1e5, 1e5, 1e3]))]
    fr = multistart_fit(rK, systems, SPECIES_ALPHA, n_starts=15)
    est = np.array([fr["alpha"][n] for n in SPECIES_ALPHA])
    print(f"[3] 合成数据回收: SSR={fr['ssr']:.2e}")
    for n, a, b in zip(SPECIES_ALPHA, true, est):
        print(f"     {n:7s} 真值={a:10.4g}  估计={b:10.4g}")


print = functools.partial(print, flush=True)

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--figures-only" in sys.argv:          # 不重新拟合，只根据 results/*.json 重画图
        dsets, d1d6 = load_all()
        with open(os.path.join(OUT_DIR, "summary.json"), encoding="utf-8") as f:
            sm = json.load(f)
        with open(os.path.join(OUT_DIR, "curves.json"), encoding="utf-8") as f:
            cv = {tuple(k.split("|")): v for k, v in json.load(f).items()}
        make_figures(dsets, d1d6, sm["rK"], sm["baseline_D1D6"], sm["fits"], cv)
    else:
        main()
