# Tomada de decisão: conectividade e critério SICLE (Oral Epithelium)

Benchmark em 4 ROIs (`healthy-18-roi2`, `healthy-19-roi2`, `healthy-17-roi2`, `severe-03-roi2`), mesmo Cellpose, mesmo pós-processo (blur σ=0.5, AND condicional, etc.).

## 1. Conectividade (`--conn-opt`)

| Opção | Fórmula (intuição) | Papel |
|--------|-------------------|--------|
| **`fmax`** | `max(caminho, ‖f_root−f_j‖^(1+α\|O(R)−O(j)\|))` | SICLE-IRREG; depende de saliência **ao longo do caminho** |
| **`fsum`** | `(irreg + α\|Δsal\|)·‖f_root−f_j‖` acumulado | SICLE-COMP; células compactas |
| **`gradvmaxmul`** | `(irreg + α\|∇sal_j−∇sal_i\|)·‖f_root−f_j‖` com max no caminho | Usa **gradiente** da saliência na borda |

**Decisão:** `gradvmaxmul` — com saliência sigmoide (sem Otsu) o mapa ainda é forte na borda; `fmax` depende de contraste ao longo do caminho; `gradvmaxmul` usa |∇sal| no anel.

### BR médio (pares literatura + nossa escolha)

| ROI | fmax+minsc | fsum+maxsc | gradvmaxmul+minsc |
|-----|----------:|-----------:|------------------:|
| healthy-18-roi2 | 0.348 | 0.331 | **0.456** |
| healthy-19-roi2 | 0.337 | 0.323 | **0.386** |
| healthy-17-roi2 | 0.356 | 0.308 | **0.395** |
| severe-03-roi2 | 0.268 | 0.248 | **0.307** |

---

## 2. Critério (`--crit-opt`) — com `gradvmaxmul` fixo

O critério define a **prioridade de remoção de seeds** no IFT (superpixels iniciais):

| Critério | Prioridade (SICLE) | Interpretação |
|----------|-------------------|---------------|
| **`minsc`** | `size_perc × min_color_grad` | Remove seeds **pequenos e pouco contrastados** (default IRREG) |
| **`maxsc`** | `size_perc × max_color_grad` | Remove seeds **grandes e contrastados** (default COMP) |
| **`size`** | `size_perc` | Só área do superpixel |
| **`spread`** | `size_perc × min_dist` | Favorece remover seeds **perto do centro** |

### BR por critério (`gradvmaxmul`, α=2.0)

| ROI | minsc | maxsc | size | spread | Melhor |
|-----|------:|------:|-----:|-------:|--------|
| healthy-18-roi2 | **0.456** | 0.393 | 0.370 | 0.373 | minsc |
| healthy-19-roi2 | 0.386 | **0.411** | 0.358 | 0.337 | maxsc |
| healthy-17-roi2 | 0.395 | 0.391 | **0.446** | 0.355 | size |
| severe-03-roi2 | **0.307** | 0.284 | 0.271 | 0.284 | minsc |

### BR macro (média das 4 ROIs)

| Critério | BR médio |
|----------|--------:|
| **minsc** | **0.386** |
| maxsc | 0.370 |
| size | 0.361 |
| spread | 0.337 |

**Decisão:** manter **`minsc`** — melhor BR médio; alinhado ao preset irregular e estável em casos severe e na ROI exemplar `healthy-18-roi2`.

- **`maxsc`**: ganha só em `healthy-19-roi2` (campo muito denso, 50 células); é o par natural de `fsum` no SICLE-COMP, não do nosso `gradvmaxmul`.
- **`size`**: ganha BR em `healthy-17-roi2` (poucas células), mas perde nos outros cenários → pouco robusto.
- **`spread`**: pior macro; supõe objeto mais “convexo”, ruim para contatos laterais.

---

## 3. Configuração final do pipeline (2026)

```
--no-saliency-linearize          # sigmoid apenas (sem compressão Otsu)
--sicle-conn-opt gradvmaxmul
--sicle-crit-opt minsc
--sicle-alpha 2.0
--saliency-blur-sigma 0.5
--disable-and-merge              # SICLE cru no bbox (sem morfologia / sem AUR)
--closing-radius 0
```

Arquivo: `configs/sicle_raw_nolin_blur05.args`

Sweep **Nf**: `oral/benchmark_nf_sweep.py --full` → `outputs/runs/nf_sweep_full/`

Validação **Nf** e **multiscale** (BR antes/depois do AND): `oral/benchmark_nf_multiscale_validation.py` → `outputs/runs/nf_multiscale_validation/metrics_nf_multiscale.csv`

---

## 4. Figuras sugeridas

| Figura | Arquivo |
|--------|---------|
| **Painel comparativo completo (rotulado)** | `outputs/runs/path_cost_benchmark/panels/exemplar_comparison_labeled.png` |
| Conectividades (1 ROI) | `outputs/runs/path_cost_benchmark/panels/healthy_healthy-18-roi2_path_costs.png` |
| Critérios (1 ROI) | `outputs/runs/path_cost_benchmark/panels/healthy_healthy-18-roi2_criteria_gradvmaxmul.png` |

Gerar o painel completo:

```bash
python3 oral/build_exemplar_comparison_panel.py
```

---

## 5. Reproduzir

```bash
cd new_pipeline
export PYTHONPATH="$(pwd):$(pwd)/pipeline:$(pwd)/cellpose:$(pwd)/oral"
export SICLE_BIN=../SICLE/bin/RunSICLE

python3 oral/benchmark_conn_cost_exemplars.py --skip-cellpose
python3 oral/benchmark_sicle_criteria.py --skip-cellpose
python3 oral/build_criteria_exemplar_panel.py --category healthy --roi healthy-18-roi2
```

CSV: `outputs/runs/path_cost_benchmark/metrics_criteria.csv`, `metrics_by_roi.csv`
