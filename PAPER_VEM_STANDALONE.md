# VEM 単独論文アウトライン

## Working Title
**"Virtual Element Method for oral biofilm mechanics: from confocal images to viscoelastic fracture on arbitrary polygonal meshes"**

## Target Journal
- **Computational Mechanics** (Wriggers editor, IKM ホーム) — 最有力
- Computer Methods in Applied Mechanics and Engineering (CMAME)
- International Journal for Numerical Methods in Engineering (IJNME)

## Novelty Statement
1. **世界初**: VEM をバイオフィルム力学に適用
2. **Confocal → VEM 2-step pipeline**: FEM の 5-step を大幅短縮
3. **DI-dependent 粘弾性 VEM**: SLS + Simo 1987 on arbitrary polygons/polyhedra
4. **Phase-field 剥離 on VEM**: DI → G_c マッピングによるバイオフィルム破壊
5. **Growth-coupled VE-VEM**: 微生物動態 → 時間発展する粘弾性応答

---

## Structure

### 1. Introduction (2 pages)

#### 1.1 Biofilm mechanics — why it matters
- 口腔バイオフィルム: 構造化多菌種コミュニティ, dysbiosis → 歯周病
- 力学的性質 (E, G_c, η) が菌種組成に依存 (Pattem 2018/2021, Peterson 2015)
- 剥離 (detachment) = 界面破壊 → 力学モデルが必須
- 既存 FEM アプローチの限界: 構造メッシュ生成がボトルネック

#### 1.2 Virtual Element Method — why VEM
- 任意多角形/多面体要素 (Beirão da Veiga 2013, 2014)
- Confocal コロニー = Voronoi セル = VEM 要素 (1:1 対応)
- Remesh 不要 → 成長・剥離・適応細分化に自然に対応
- IKM の VEM 固体力学 (Wriggers 2019, 2024; Aldakheel 2018)

#### 1.3 Contributions
- 12 モジュールの統合フレームワーク (Table 1)
- DI-dependent 構成則: E(DI), G_c(DI), SLS(DI)
- 2D/3D machine precision 検証
- Confocal → VEM パイプラインのデモ

---

### 2. Mathematical Framework (4 pages)

#### 2.1 VEM for linear elasticity (P₁)
- Virtual element space, elliptic projection Π^∇
- Stiffness decomposition: K = K_π + K_stab
- 2D: 6 poly basis (3 RBM + 3 strain), edge integrals
- 3D: 12 poly basis (3 trans + 3 rot + 6 strain), face integrals
- Stabilization: α_s tr(C)|E| (I - Π D)^T (I - Π D)

#### 2.2 P₂ VEM (2nd-order)
- Vertex + edge midpoint DOFs → 4n_v DOFs/element
- 12 poly basis: 3 RBM + 3 linear strain + 6 quadratic
- Analytical strain energy via sub-triangulation Gauss quadrature
- Volume correction for div(σ(p_α)) ≠ 0 in quadratic modes

#### 2.3 Viscoelastic VEM — SLS + Simo 1987
- Standard Linear Solid: σ = C_inf ε + h(t)
- Simo exponential integrator: h_{n+1} = exp(-dt/τ) h_n + γ C_1 Δε
- Algorithmic tangent: C_alg = C_inf + γ C_1
- DI → SLS parameters: E_inf(DI), τ(DI), E_1(DI), η(DI)

#### 2.4 Neo-Hookean VEM (large deformation)
- W(F) = μ/2(I₁-2) - μ ln(J) + λ/2(ln J)²
- Newton-Raphson + load stepping + line search
- DI → μ, λ via E(DI), ν

#### 2.5 Phase-field fracture on VEM
- Aldakheel 2018 framework: staggered u-d solve
- Spectral decomposition: ψ⁺/ψ⁻ tension-compression split
- DI → G_c(DI): commensal tough, dysbiotic fragile
- Irreversibility: d_{n+1} ≥ d_n

#### 2.6 Cohesive zone model on VEM
- Bilinear traction-separation law (Park-Paulino-Roesler 2009)
- Interface elements at tooth-biofilm boundary
- DI → σ_max, δ_c: mixed-mode coupling

#### 2.7 Adaptive h-refinement
- A posteriori error estimator: ZZ-type stress recovery
- Dörfler marking strategy
- Crack-tip indicator: η = w₁|∇d| + w₂ ψ⁺/G_c
- Field transfer (d, ψ history) via nearest-neighbor interpolation

---

### 3. DI-Dependent Constitutive Laws (1.5 pages)

#### 3.1 Dysbiosis Index (DI)
- DI = -Σ φᵢ ln φᵢ / ln 5 (normalized Shannon entropy)
- Hamilton 5-species ODE → φᵢ(t) → DI(t)
- TMCMC calibration (20 params, 4 conditions)

#### 3.2 Stiffness: E(DI)
- E(DI) = E_min + (E_max - E_min)(1-DI)^n
- E_max = 1000 Pa, E_min = 30 Pa, n = 2
- 文献根拠: Pattem 2018 (30× range), Gloag 2019, percolation theory (n=2)

#### 3.3 Fracture toughness: G_c(DI)
- G_c(DI) = G_c_min + (G_c_max - G_c_min)(1-DI)^n
- G_c_max = 0.5 J/m², G_c_min = 0.01 J/m²

#### 3.4 Viscoelastic: SLS(DI)
- E_inf(DI), E_0 = 2 E_inf, E_1 = E_0 - E_inf
- τ(DI) = τ_min + (τ_max - τ_min)(1-DI)^n
- η = E_1 τ

---

### 4. Confocal → VEM Pipeline (1.5 pages)

#### 4.1 FEM の 5-step pipeline (従来)
confocal → voxel → marching cubes → tet mesh → Abaqus input

#### 4.2 VEM の 2-step pipeline (提案)
confocal → colony detection → Voronoi tessellation → VEM

#### 4.3 Implementation
- 5ch fluorescence → species classification
- Connected component → colony centroid
- Voronoi tessellation (scipy.spatial)
- Per-colony DI assignment → per-element material
- Demo: Heine 2025 FISH images

---

### 5. Numerical Verification (3 pages)

#### 5.1 Patch test & convergence (linear elasticity)
- Patch test: 10⁻¹⁸ (2D), 10⁻¹⁹ (3D)
- h-convergence: L² = 2.14, H¹ = 1.29 (Voronoi)
- VEM vs FEM comparison (Table)
- P₂ vs P₁: 15-45% stress accuracy improvement

#### 5.2 VE-VEM validation
- Laterally confined step strain — analytical solution
- 2D: 64 Voronoi cells, error = 1.3×10⁻¹⁵
- 3D: 27 hexahedral cells, error = 4.9×10⁻¹⁶
- Stress relaxation: monotonic decrease ✓
- Long-time limit: σ → E_inf ε/(1-ν²) ✓
- Instantaneous response: σ₀ = E_0 ε/(1-ν²) ✓

#### 5.3 Neo-Hookean validation
- Cook's membrane: Newton-Raphson convergence
- Linear vs nonlinear: 43% displacement difference at ε~1%
- Load stepping: 10 steps, ~25 NR iterations

#### 5.4 Phase-field validation
- Single edge notch: crack path comparison
- DI-dependent: dysbiotic cracks first (G_c = 0.01 J/m²)
- Step 18: catastrophic failure (d: 0.33 → 1.0)

---

### 6. Biofilm Applications (4 pages)

#### 6.1 Growth-coupled viscoelastic response
- Hamilton ODE → DI(t) → SLS(t) → VE-VEM
- 3 conditions: CS, DH, DS
- **Key result**: CS stiffens (DI↓, E↑, τ↑) vs DS softens (DI↑, E↓, τ↓)
- Fig: 3-condition stress relaxation comparison
- Fig: DI(t), E_inf(t), τ(t) evolution

#### 6.2 Phase-field biofilm detachment
- DI gradient: commensal periphery + dysbiotic center
- GCF pressure loading
- Dysbiotic center cracks → commensal survives
- Adaptive refinement at crack tip: 40 → 121 cells
- Fig: Phase-field evolution (d field snapshots)
- Fig: Load-displacement curve with failure point

#### 6.3 CZM tooth-biofilm interface
- Bilinear TSL with DI-dependent σ_max
- Progressive debonding from weak (dysbiotic) center
- Load redistribution to intact (commensal) periphery
- Fig: Interface traction distribution at progressive load factors

#### 6.4 Confocal image-based analysis
- Heine 2025 FISH → Voronoi → VEM
- Per-colony DI → spatially resolved E, G_c
- von Mises stress field on confocal-derived mesh
- Fig: Confocal image → Voronoi mesh → stress field (side by side)

#### 6.5 DI gradient with viscoelasticity
- Spatial DI variation → spatially varying SLS params
- Commensal (left) 59.7% relaxation vs dysbiotic (right) 75.0%
- Fig: Spatial stress field at t=0, t=τ, t=3τ

---

### 7. Discussion (2 pages)

#### 7.1 VEM vs FEM for biofilm
- Pipeline 短縮 (5-step → 2-step)
- Per-colony resolution without sub-element averaging
- Topology changes handled naturally
- Convergence rates competitive with FEM

#### 7.2 Viscoelastic VEM — significance
- Simo 1987 × VEM: 文献に前例なし
- Machine precision → 実装の完全な正しさ
- Growth-coupled: 組成変化 → 力学応答を一気通貫

#### 7.3 Phase-field on VEM — significance
- Aldakheel 2018 をバイオフィルムに初適用
- DI → G_c: 組成から破壊靭性を予測
- Adaptive refinement: 計算コスト削減

#### 7.4 Limitations
- E(DI), G_c(DI), SLS(DI) は partially supported (直接同時計測データなし)
- 小変形仮定 (Neo-Hookean は大変形対応済み)
- 2D plane-stress (3D は検証済み、応用は future work)
- 粘弾性は SLS 1 branch のみ (Prony series 拡張可能)
- Confocal pipeline はシンセティック画像でデモ

#### 7.5 Future work
- VEM × TMCMC integration (Bayesian inference with VEM forward model)
- 3D confocal z-stack → polyhedral VEM
- DeepONet surrogate on VEM meshes
- 実験検証: Sanz-Martin 2022 or 新規 confocal データ

---

### 8. Conclusion (0.5 pages)
- 世界初: VEM × バイオフィルム力学
- 12 ソルバー, 120+ テスト, machine precision
- DI-dependent 弾性/粘弾性/破壊の統一フレームワーク
- Confocal → VEM 2-step pipeline で実験データ直結
- IKM の VEM 固体力学 + TMCMC 微生物動態 = 新しい計算バイオメカニクス

---

## Figure Plan (12-15 figures)

| Fig | Content | Source |
|-----|---------|--------|
| 1 | **Pipeline comparison**: FEM 5-step vs VEM 2-step | New schematic |
| 2 | **VEM schematic**: projection Π^∇, polygon element | New schematic |
| 3 | **Constitutive laws**: E(DI), G_c(DI), SLS(DI) + literature overlay | `generate_grand_showcase.py` panel (a) extended |
| 4 | **h-convergence**: VEM vs FEM, L²/H¹ rates | `vem_convergence_study.py` |
| 5 | **VE-VEM validation**: confined relaxation, 2D & 3D analytical match | `vem_viscoelastic.py` + `vem_3d_viscoelastic.py` |
| 6 | **P₁ vs P₂**: stress accuracy comparison | `vem_p2_elasticity.py` |
| 7 | **Neo-Hookean**: linear vs nonlinear displacement | `vem_nonlinear.py` |
| 8 | **Phase-field evolution**: d-field snapshots + load-displacement | `vem_phase_field.py` |
| 9 | **Adaptive refinement**: mesh evolution at crack tip | `vem_adaptive_fracture.py` |
| 10 | **CZM debonding**: interface traction at progressive loads | `vem_czm.py` |
| 11 | **Growth-coupled VE-VEM**: 3-condition DI(t), σ(t), E(t) evolution | `vem_viscoelastic_growth.py` |
| 12 | **DI gradient + viscoelasticity**: spatial stress at t=0, τ, 3τ | `vem_viscoelastic.py` demo |
| 13 | **Confocal → VEM**: FISH image → Voronoi → stress field | `vem_confocal_pipeline.py` |
| 14 | **Grand showcase**: 8-panel overview (graphical abstract candidate) | `generate_grand_showcase.py` |

---

## References (~40 papers)

### VEM foundational
- Beirão da Veiga et al. (2013, 2014) — M3AS
- Sutton (2017) — Numer. Algorithms
- Ahmad et al. (2013) — Comput. Math. Appl.

### VEM nonlinear / fracture (IKM)
- Wriggers, Hudobivnik (2019) — Comput. Mech.
- Wriggers, Aldakheel, Hudobivnik (2024) — Springer book
- Aldakheel et al. (2018) — CMAME
- Nguyen-Thanh et al. (2018) — CMAME
- Artioli et al. (2017) — P₂ VEM

### Viscoelastic mechanics
- Simo (1987) — CMAME — exponential integrator
- Simo, Hughes (1998) — Computational Inelasticity
- Xu, Junker, Wriggers (2025) — Space-time VEM (if published)

### Phase-field fracture
- Bourdin, Francfort, Marigo (2000, 2008)
- Miehe, Hofacker, Welschinger (2010) — CMAME
- Ambati, Gerasimov, De Lorenzis (2015) — CM

### Cohesive zone
- Park, Paulino, Roesler (2009) — IJNME
- Xu, Needleman (1994)

### Biofilm mechanics
- Pattem et al. (2018) — Sci Rep — AFM + 16S
- Pattem et al. (2021) — Sci Rep — hydrated biofilm
- Gloag et al. (2019) — J Bacteriol
- Peterson, Stoodley (2015) — FEMS Microbiol Rev
- Flemming, Wingender (2010) — Nat Rev Microbiol
- Klempt et al. (2024) — staggered coupling
- Heine et al. (2025) — 5-species FISH

### Percolation / n=2 justification
- Sahimi (1994) — Applications of Percolation Theory
- Arbabi, Sahimi (1993) — Phys Rev B

---

## 執筆スケジュール案

| Week | Task |
|------|------|
| 1 | Sec 2 (数学) + Sec 5 (検証) — 既存コード結果をまとめる |
| 2 | Sec 3 (構成則) + Sec 4 (pipeline) — 図作成 |
| 3 | Sec 6 (応用) — growth-coupled, phase-field, confocal 結果 |
| 4 | Sec 1 (intro) + Sec 7 (discussion) + Sec 8 (conclusion) |
| 5 | 全体推敲 + Wriggers/Aldakheel 先生にドラフト共有 |

## 著者案
- Nishioka K., Aldakheel F., Wriggers P.
- IKM, Leibniz Universität Hannover
