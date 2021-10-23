# Replicate-lce
This replication for lce-loss is based on [Capreolus](https://github.com/capreolus-ir/capreolus) toolkit.


## Quick Start
1. Prerequisites: Python 3.7+ and Java 11. See the [installation instructions](https://capreolus.ai/en/latest/installation.html)
2. Clone capreolus toolkit: git clone https://github.com/crystina-z/capreolus.git -b feature/eval+ptmaxp
3. Run an experiment: `
python run.py --config_path yaml_config_neg/yaml_filename` 

<br />


## Tables
Here are the three tables we stated and the corresponding yaml file for each row 

<br />


### Table1
| Baseline - HN+FirstStage - Loss - n    | MRR@10 |yaml_file_name | 
|-------------------------------|----------|---------------|
| (1) monoELECTRABase-BM25-CE-1       |  0.378  | test_line4_nneg_1_bm25_ce.yaml
| (2) monoELECTRABase-BM25-HG-1       |  0.379  | test_line5_nneg_1_bm25_hg.yaml
| (3) monoELECTRABase-BM25-LCE-1       |  0.378  | test_line24_nneg_1_bm25_lce.yaml
| (4) monoELECTRABase-BM25-LCE-7       |  0.391  | test_line24_nneg_1_bm25_lce.yaml
| (5) monoELECTRABase-TCT-ColBERTv2-CE-1             |  0.365     | test_line26_nneg_1_tct_ce.yaml
| (6)  monoELECTRABase-TCT-ColBERTv2-HG-1       |  0.375   | test_line27_nneg_1_tct_hg.yaml
| (7)  monoELECTRABase-TCT-ColBERTv2-LCE-1           |  0.394   | test_line16_nneg_1_tct_lce.yaml
| (8)  monoELECTRABase-TCT-ColBERTv2-LCE-7             | 0.401     | test_line11_nneg_7_tct_lce.yaml


<br />


Please note that in table 2 both a and c, b and d are trained with the same yaml file but inference on different ones.

### Table2
| HN-FirstStage    | MRR@10 |yaml_file_name | 
|-------------------------------|----------|---------------|
| (a) BM25-BM25       |  0.391  | test_line10_nneg_7_bm25_lce.yaml
| (b) TCT-ColBERTv2-BM25      |  0.389 | test_line11_nneg_7_tct_lce.yaml
| (c) BM25-TCT-ColBERTv2       |  0.402  | test_line10_nneg_7_bm25_lce.yaml
| (d) TCT-ColBERTv2-TCT-ColBERTv2      |  0.401  | test_line11_nneg_7_tct_lce.yaml


<br />



### Table3
| Group-size    | MRR@10 |yaml_file_name | 
|-------------------------------|----------|---------------|
| (1) 2      |  0.393  | test_line26_nneg_1_tct_ce.yaml
| (2) 4      |  0.400 | test_line17_nneg_3_tct_lce.yaml
| (3) 8       |  0.401  | test_line11_nneg_7_tct_lce.yaml
| (4) 16      |  0.408  | test_line19_nneg_15_tct_lce.yaml
| (5) 32      |  0.414  | test_line20_nneg_31_tct_lce.yaml
