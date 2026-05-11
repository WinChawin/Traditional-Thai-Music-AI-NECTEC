# 01_FindingTori-Reproduction

Reproduce paper [Finding Tori: Self-Supervised Learning for Analyzing Korean Folk Song](https://arxiv.org/abs/2308.02249) (Han et al., ISMIR 2023) บน Google Colab

ทำเป็นจุดเริ่มของ Traditional-Thai-Music-AI เพราะ paper นี้แก้ปัญหาใกล้กับเป้าหมายของเรา — ดนตรีถ่ายทอดแบบมุขปาฐะ, ไม่มี notation มาตรฐาน, มี classification system (tori) ที่ใช้เป็น proxy ของ "ทาง" ใน Thai music ได้

## ผลที่ได้

ทั้ง 3 phases reproduce paper สำเร็จบน Colab L4 ภายใน ~50 นาที

| Phase | Goal | Status |
|---|---|---|
| 1 | Reproduce Table 1 — 4 baselines NDCG + RF accuracy | ✓ NDCG ±0.01, RF 3/4 within 1σ |
| 2 | Train CNN 2 ตัวจาก scratch | ✓ SSL 0.8533 ตรง paper 0.853 |
| 3 | Reproduce Figure 4 — 4-panel UMAP | ✓ ทั้ง 4 panels ตรง paper |

ดูตาราง + รูปทั้งหมดใน `results/`

## Bugs ที่เจอใน upstream repo

5 จุดที่ทำให้ "out-of-the-box reproduction" ไม่ work — patch ทั้งหมดอยู่ใน notebook section 0.5 + Phase 2 Hydra overrides

1. `get_eval_result.py`: `hasattr` check พลาดบน wandb-style config
2. `get_eval_result.py`: ชี้ `metadata_sed.csv` ที่ไม่ได้ ship
3. `train.py`: `wandb.init()` ใช้ placeholder project/entity
4. `train.py`: `meta_csv_path` default เดียวกับ bug 2
5. Hydra 1.2+ `chdir` default ทำให้ relative paths พัง

ที่ไม่ใช่ bug แต่เป็น README typo: README บอก `--model=region-trained` แต่ code จริงคือ `region-supervised`

## ทำไมถึง reproduce paper นี้

ตามที่ project guidelines ระบุ "บิดเบือน ทาง / ทางเสียง / ลีลาครู" ของ Thai music เป็น probabilistic, lineage-aware concepts — ไม่ใช่ classification ตายตัว

Finding Tori แสดงว่า:
- SSL ที่ไม่ใช้ label > supervised ที่ใช้ label สำหรับ task ประเภทนี้
- UMAP overlay + metadata filter ใช้ probe ได้ว่า embedding cluster ตามที่ผู้เชี่ยวชาญคาดหรือไม่
- pipeline ทำงานกับ field recording ที่ noisy ของนักร้อง non-expert ได้

3 ข้อนี้ map ตรงกับสิ่งที่ Thai project ต้องการ

## Reference

- Paper: Han, D., Caro Repetto, R., & Jeong, D. (2023). *Finding Tori: Self-Supervised Learning for Analyzing Korean Folk Song.* ISMIR 2023.
- Upstream repo: https://github.com/danbinaerinHan/finding-tori
- Dataset: Anthology of Korean Traditional Folksongs (MBC, 1989-1996) — paper ไม่ redistribute เสียงต้นฉบับ มีแต่ pre-extracted F0 contour CSV