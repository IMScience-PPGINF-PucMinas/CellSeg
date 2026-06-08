# Validação Nf e multiscale (sem Otsu)

Método base: `--no-saliency-linearize`, `gradvmaxmul` + `minsc`, blur σ=0.5.

Script: `oral/benchmark_nf_multiscale_validation.py`  
CSV: `outputs/runs/nf_multiscale_validation/metrics_nf_multiscale.csv`

## BR macro — escala única, pós-processo completo (`post=full`)

| Nf | BR médio (4 ROIs) |
|----|------------------:|
| **2** | **0.387** |
| 3 | 0.234 |
| 4 | 0.163 |
| 5 | 0.127 |

**Conclusão:** aumentar `Nf` **não** recupera BR; piora de forma consistente. O dano ocorre **dentro do SICLE** (hierarquia mais fina → mais remoção de seeds pelo `minsc`), não no AND com Cellpose.

## `full` vs `sicle_raw` (só `disable-and-merge`, sem AND)

| ROI | Nf=2 full | Nf=2 raw | Δ (raw − full) |
|-----|----------:|---------:|---------------:|
| healthy-18-roi2 | 0.447 | 0.444 | −0.003 |
| healthy-19-roi2 | 0.401 | 0.413 | **+0.012** |
| healthy-17-roi2 | 0.405 | 0.432 | **+0.026** |
| severe-03-roi2 | 0.295 | 0.302 | +0.007 |

O pós-processo (AND condicional) tira um pouco de BR em alguns ROIs, mas a queda de Nf≥3 é muito maior (ex.: healthy-18: 0.447 → 0.218 com Nf=3).

## Multiscale

Com `Nf=2`, `--sicle-multiscale` + `last` ou `veta_composite` ≈ escala única (mesmos BR/Dice).  
Com `Nf≥3`, multiscale não compensa a perda — escolha de escala (`last` vs `veta_composite`) muda pouco.

Para análise intermediária, inspecionar `percell_cell_outputs/cell_*/` em pastas `nf3_*` vs `nf2_*` (máscaras SICLE por célula e `percell_sicle_log.txt`).

## Configuração recomendada

Manter **`Nf=2`** (default). Não subir Nf para “recuperar” borda — o critério `minsc` remove mais seeds em níveis mais finos.

Arquivo de args: `configs/sicle_nolin_blur05.args`
