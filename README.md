# Virtual Element Methods for Biofilm Mechanics

> **IKM Hannover** — Nishioka Keisuke (2026)
>
> VEM を口腔バイオフィルムの計算力学に初適用。Wriggers グループの VEM 固体力学 + Aldakheel の Phase-field 破壊を、DI-dependent 構成則と結合。

## Overview

FEM の一般化として **任意多角形/多面体要素** 上で弾性・破壊問題を解く。
形状関数を明示構成しない ("virtual") 代わりに射影演算子 Π^∇ で剛性行列を近似する。

**バイオフィルムへの利点**:
- Confocal 画像の菌コロニーがそのまま VEM 要素 → **2-step pipeline** (FEM の 5-step から短縮)
- 成長でトポロジーが変わっても任意形状許容 → **remesh 不要**
- Phase-field 破壊でメッシュバイアスなし → **剥離パスに物理的妥当性**

## Architecture

```
VirtualElementMethods/
├── vem.py                      # Scalar Poisson (Sutton 2017 base)
├── vem_elasticity.py           # 2D plane-stress elasticity (P₁)
├── vem_3d.py                   # 3D elasticity on polyhedra
├── vem_3d_advanced.py          # 3D Voronoi + sparse assembly + VTK
│
├── vem_nonlinear.py            # ★ Neo-Hookean hyperelasticity + Newton-Raphson
├── vem_phase_field.py          # ★ Phase-field fracture (Aldakheel 2018)
├── vem_spacetime.py            # Space-Time VEM (SLS viscoelastic)
├── vem_growth_coupled.py       # Growth-Coupled VEM (Hamilton ODE + VEM)
├── vem_confocal_pipeline.py    # Confocal → VEM pipeline
├── vem_error_estimator.py      # A posteriori error + adaptive refinement
│
├── vem_convergence_study.py    # h-convergence: VEM vs FEM
├── vem_benchmark.py            # Performance benchmarks
├── vem_apple.py                # Apple-shaped 3D demo
├── vem_3d_confocal.py          # 3D confocal → VEM
├── process_heine_fish.py       # Heine 2025 FISH → VEM (real data)
│
├── tests/                      # 70+ pytest tests
│   ├── test_vem_poisson.py         (6 tests)
│   ├── test_vem_elasticity.py      (9 tests)
│   ├── test_vem_3d.py              (15 tests)
│   ├── test_vem_spacetime.py       (4 tests)
│   ├── test_vem_growth.py          (12 tests)
│   └── test_vem_error_estimator.py (12 tests)
├── heine_extracted/            # FISH images from Heine 2025 Fig 3B
└── results/                    # Pipeline outputs + demo figures
```

## Development Roadmap

### Track A: 固体力学・非線形 VEM → バイオフィルム

**現状**: P₁ 線形弾性 VEM (小変形) → 現実のバイオフィルムは **大変形 + 非線形材料**

| Step | 内容 | 難易度 | Impact | Status |
|------|------|--------|--------|--------|
| **A1** | **Neo-Hookean VEM** — 大変形超弾性 (biofilm は軟体, ε > 10%) | 中 | 高 | **Done** ✓ |
| **A2** | **P₂ VEM** — 応力精度向上 (BC角 30%→5%), 辺中点 DOF | 中 | 中 | Planned |
| **A3** | **VE-VEM** — SLS 粘弾性 + VEM (space-time prototype の実用化) | 高 | 高 | Prototype |

### Track B: 破壊・Phase-field VEM → バイオフィルム剥離

**着眼点**: バイオフィルム剥離 (detachment) = **界面破壊問題** そのもの

| Step | 内容 | 難易度 | Impact | Status |
|------|------|--------|--------|--------|
| **B1** | **Phase-field detachment** — DI→破壊靭性 G_c(DI) マッピング | 中 | 極高 | **Done** ✓ |
| **B2** | **CZM (Cohesive Zone) on VEM** — 歯-バイオフィルム界面の剥離 | 中 | 高 | Planned |
| **B3** | **Adaptive VEM + crack** — 亀裂先端に自動細分化 (error estimator 済) | 高 | 高 | Planned |

### 優先順位

1. **B1 (Phase-field)** — 世界初: VEM × phase-field × バイオフィルム剥離、論文価値最大
2. **A1 (Neo-Hookean)** — 大変形時の線形 vs 非線形比較で物理的妥当性を示す
3. **B3 (Adaptive + crack)** — error estimator 既に実装済み、亀裂先端自動細分化は即座に拡張可能
4. **A3 (VE-VEM)** — space-time prototype から実用化、粘弾性バイオフィルムの時間応答
5. **A2 (P₂)** — 応力精度改善、Artioli et al. (2017) 参照
6. **B2 (CZM)** — 歯-バイオフィルム界面、臨床的意義大

## Module Details

### Core Solvers

| Module | Description | Key Features |
|--------|-------------|--------------|
| `vem_elasticity.py` | 2D P₁ linear elasticity | 6 poly basis, patch test 1e-18, sparse assembly |
| `vem_nonlinear.py` | **Neo-Hookean hyperelasticity** | Newton-Raphson, load stepping, line search, DI→μ,λ |
| `vem_phase_field.py` | **Phase-field fracture** | Aldakheel 2018, staggered solve, DI→G_c, spectral decomp |
| `vem_3d.py` | 3D elasticity on polyhedra | 12 poly basis, patch test 1e-19 |
| `vem_3d_advanced.py` | 3D Voronoi + VTK | Sparse solver, convergence rate 1.80 |

### Growth-Coupled

| Module | Description | Key Features |
|--------|-------------|--------------|
| `vem_growth_coupled.py` | Hamilton ODE + VEM | 5-species replicator, cell division, two-way coupling |
| `vem_spacetime.py` | Space-Time VEM | Anisotropic (x,t) Voronoi, SLS viscoelastic |
| `vem_confocal_pipeline.py` | Confocal → VEM | 5ch fluorescence → colony detection → Voronoi → VEM |

### Analysis Tools

| Module | Description | Key Features |
|--------|-------------|--------------|
| `vem_error_estimator.py` | A posteriori error | ZZ-type, Dörfler marking, h-adaptive refinement |
| `vem_convergence_study.py` | h-convergence | Manufactured solution, L²/H¹ rates |
| `vem_benchmark.py` | Performance | Timing, scaling, memory |

## Key Results

### A1: Neo-Hookean VEM
- Linear vs NL comparison: strain ~1% → ~43% displacement difference
- Newton-Raphson converges in 10 load steps, ~25 NR iterations
- Nonlinearity significant at ε > 5-10% (typical for soft biofilm under GCF)

### B1: Phase-field Detachment
- **Step 18: 急激な破壊** (d: 0.33 → 1.0)
- Dysbiotic center cracks first (low G_c = 0.01 J/m²)
- Commensal periphery remains intact (high G_c = 0.5 J/m²)
- 82/82 nodes in crack zone reach d=1.0 (full failure)
- Clear load-displacement curve with distinct failure point

### h-Convergence (Linear VEM)
| Method | L² rate | H¹ rate |
|--------|---------|---------|
| VEM (Voronoi) | **2.14** | **1.29** |
| VEM (Quad) | **2.03** | **1.99** |
| FEM (Triangle) | **1.88** | **0.99** |

### Heine 2025 FISH → VEM
- 10 FISH images processed (Commensal/Dysbiotic HOBIC, Days 1-21)
- 200-325 Voronoi cells per image
- Hybrid DI pipeline: spatial info from images + DI from TMCMC-calibrated ODE

## Constitutive Laws

### E(DI) — Stiffness
```
E(DI) = E_min + (E_max - E_min) · (1 - DI)^n
E_max = 1000 Pa (commensal), E_min = 30 Pa (dysbiotic), n = 2
```

### G_c(DI) — Fracture Toughness (NEW)
```
G_c(DI) = G_c_min + (G_c_max - G_c_min) · (1 - DI)^n
G_c_max = 0.5 J/m² (commensal, tough), G_c_min = 0.01 J/m² (dysbiotic, fragile)
```

### Neo-Hookean — Large Deformation (NEW)
```
W(F) = μ/2·(I₁ - 2) - μ·ln(J) + λ/2·(ln J)²
μ, λ derived from E(DI), ν
```

## Quick Start

```bash
# Run tests
python -m pytest tests/ -v

# Demos
python vem_elasticity.py                # Linear elasticity
python vem_nonlinear.py                 # Neo-Hookean large deformation
python vem_phase_field.py               # Phase-field detachment
python vem_growth_coupled.py            # Growth-coupled simulation
python vem_confocal_pipeline.py         # Confocal → VEM pipeline
```

## References

### Foundational
- Beirão da Veiga et al. (2013) "Basic principles of VEM" — M3AS
- Beirão da Veiga et al. (2014) "Hitchhiker's Guide to VEM" — M3AS
- Sutton (2017) "VEM in 50 lines of MATLAB" — Numer. Algorithms

### Nonlinear / Fracture (IKM Hannover)
- Wriggers, Hudobivnik (2019) "Low order 3D VEM for finite elasto-plastic" — Comput. Mech.
- Aldakheel et al. (2018) "Phase-field brittle fracture using VEM" — CMAME
- Nguyen-Thanh et al. (2018) "VEM for 2D fracture analysis" — CMAME
- Wriggers, Aldakheel, Hudobivnik (2024) *VEM in Engineering Sciences* — Springer

### Biofilm Mechanics
- Klempt et al. (2024) — Staggered coupling FEM + growth
- Heine et al. (2025) — 5-species in vitro FISH data
- Pattem et al. (2018, 2021) — Oral biofilm AFM + 16S rRNA

## PDFs & External Repos

論文 PDF: `~/IKM_Hiwi/external_repos/VEM_papers/` (11 papers, indexed in README.md)

Related repositories in `~/IKM_Hiwi/external_repos/`:
mVEM, vemlab, Veamy, polyfem, scikit-fem, VEM3D, vem (IKM Hannover)

---

> **世界初**: VEM × バイオフィルム力学。IKM の VEM 固体力学の強みと、5-species TMCMC calibration を組み合わせた、実験データ駆動型の計算バイオメカニクス。
