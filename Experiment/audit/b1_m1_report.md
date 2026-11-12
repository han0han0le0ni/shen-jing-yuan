# Benchmark-1 sweep report

| D | k1_200 | k3 | notrace |
|---|---|---|---|
| 10 | 100.0% | 98.7% | 46.7% |
| 50 | 100.0% | 100.0% | 46.7% |
| 100 | 100.0% | 100.0% | 46.7% |
| 200 | 100.0% | 100.0% | 46.7% |
| 500 | 46.7% | 67.3% | 46.7% |
| 1000 | 46.7% | 46.7% | 46.7% |

| condition | D1/2 | acc@200 | acc@500 |
|---|---|---|---|
| k1_200 | 481 | 100.0% | 46.7% |
| k3 | 919 | 100.0% | 67.3% |
| notrace | 0 | 46.7% | 46.7% |

Source: `data/b1_m1/sweep_*.jsonl` (summary records). Each row is one independent process/brain instance.
