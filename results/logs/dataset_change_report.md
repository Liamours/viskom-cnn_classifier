# Dataset Change Report — ChestMNIST
Generated: 2026-04-12

---

## 1. Source Dataset

| Property | Value |
|---|---|
| Dataset | ChestMNIST (MedMNIST v2) |
| Modality | Chest X-Ray (grayscale) |
| Task | Multi-Label Binary Classification (14 labels) |
| Image Size | 224 × 224 px |
| License | CC BY 4.0 |

---

## 2. Official Split Counts

| Split | Official Count |
|---|---|
| train | 78,468 |
| val | 11,219 |
| test | 22,433 |
| **Total** | **112,120** |

---

## 3. Downloaded Count (Pre-Cleaning)

| Split | Downloaded | Match Official |
|---|---|---|
| train | 78,468 | Yes |
| val | 11,219 | Yes |
| test | 22,433 | Yes |
| **Total** | **112,120** | Yes |

---

## 4. Duplicate Detection

- Method: MD5 hash of raw image bytes
- Scope: all splits cross-checked
- Duplicate groups found: **25**
- Total files to remove: **26**

### Duplicate Groups

| Group | Hash (truncated) | Files | Action |
|---|---|---|---|
| 1 | ... | `train_03930.png` *(keep)*, `train_35093.png` *(delete)* | Keep earliest by train→val→test priority |
| 2 | ... | `train_06003.png` *(keep)*, `train_18301.png` *(delete)* | Keep earliest by train→val→test priority |
| 3 | ... | `train_06164.png` *(keep)*, `train_74404.png` *(delete)* | Keep earliest by train→val→test priority |
| 4 | ... | `train_08935.png` *(keep)*, `train_64077.png` *(delete)* | Keep earliest by train→val→test priority |
| 5 | ... | `train_09134.png` *(keep)*, `train_74178.png` *(delete)* | Keep earliest by train→val→test priority |
| 6 | ... | `train_11181.png` *(keep)*, `train_17715.png` *(delete)* | Keep earliest by train→val→test priority |
| 7 | ... | `train_11800.png` *(keep)*, `train_43643.png` *(delete)* | Keep earliest by train→val→test priority |
| 8 | ... | `train_18920.png` *(keep)*, `train_63791.png` *(delete)* | Keep earliest by train→val→test priority |
| 9 | ... | `train_21954.png` *(keep)*, `train_29343.png` *(delete)* | Keep earliest by train→val→test priority |
| 10 | ... | `train_31395.png` *(keep)*, `train_60807.png` *(delete)* | Keep earliest by train→val→test priority |
| 11 | ... | `train_35331.png` *(keep)*, `train_40086.png` *(delete)*, `train_75334.png` *(delete)* | Keep earliest by train→val→test priority |
| 12 | ... | `train_38395.png` *(keep)*, `train_61195.png` *(delete)* | Keep earliest by train→val→test priority |
| 13 | ... | `train_39438.png` *(keep)*, `train_53058.png` *(delete)* | Keep earliest by train→val→test priority |
| 14 | ... | `train_42998.png` *(keep)*, `train_52519.png` *(delete)* | Keep earliest by train→val→test priority |
| 15 | ... | `train_60444.png` *(keep)*, `train_74858.png` *(delete)* | Keep earliest by train→val→test priority |
| 16 | ... | `train_65650.png` *(keep)*, `val_01517.png` *(delete)* | Keep earliest by train→val→test priority |
| 17 | ... | `train_76551.png` *(keep)*, `val_05194.png` *(delete)* | Keep earliest by train→val→test priority |
| 18 | ... | `val_02373.png` *(keep)*, `val_09387.png` *(delete)* | Keep earliest by train→val→test priority |
| 19 | ... | `val_03418.png` *(keep)*, `val_11214.png` *(delete)* | Keep earliest by train→val→test priority |
| 20 | ... | `val_03477.png` *(keep)*, `val_10307.png` *(delete)* | Keep earliest by train→val→test priority |
| 21 | ... | `val_04957.png` *(keep)*, `val_05832.png` *(delete)* | Keep earliest by train→val→test priority |
| 22 | ... | `val_08332.png` *(keep)*, `test_05546.png` *(delete)* | Keep earliest by train→val→test priority |
| 23 | ... | `val_08331.png` *(keep)*, `val_10388.png` *(delete)* | Keep earliest by train→val→test priority |
| 24 | ... | `test_03242.png` *(keep)*, `test_17956.png` *(delete)* | Keep earliest by train→val→test priority |
| 25 | ... | `test_05411.png` *(keep)*, `test_14567.png` *(delete)* | Keep earliest by train→val→test priority |

### Deduplication Strategy

Priority rule: **train > val > test**. Within same split: keep lower filename index.
Cross-split duplicates: image is retained in the higher-priority split only.

### Files Deleted

| # | Deleted File | Split | Kept File |
|---|---|---|---|
| 1 | `train_35093.png` | train | `train_03930.png` |
| 2 | `train_18301.png` | train | `train_06003.png` |
| 3 | `train_74404.png` | train | `train_06164.png` |
| 4 | `train_64077.png` | train | `train_08935.png` |
| 5 | `train_74178.png` | train | `train_09134.png` |
| 6 | `train_17715.png` | train | `train_11181.png` |
| 7 | `train_43643.png` | train | `train_11800.png` |
| 8 | `train_63791.png` | train | `train_18920.png` |
| 9 | `train_29343.png` | train | `train_21954.png` |
| 10 | `train_60807.png` | train | `train_31395.png` |
| 11 | `train_40086.png` | train | `train_35331.png` |
| 11 | `train_75334.png` | train | `train_35331.png` |
| 12 | `train_61195.png` | train | `train_38395.png` |
| 13 | `train_53058.png` | train | `train_39438.png` |
| 14 | `train_52519.png` | train | `train_42998.png` |
| 15 | `train_74858.png` | train | `train_60444.png` |
| 16 | `val_01517.png` | val | `train_65650.png` |
| 17 | `val_05194.png` | val | `train_76551.png` |
| 18 | `val_09387.png` | val | `val_02373.png` |
| 19 | `val_11214.png` | val | `val_03418.png` |
| 20 | `val_10307.png` | val | `val_03477.png` |
| 21 | `val_05832.png` | val | `val_04957.png` |
| 22 | `test_05546.png` | test | `val_08332.png` |
| 23 | `val_10388.png` | val | `val_08331.png` |
| 24 | `test_17956.png` | test | `test_03242.png` |
| 25 | `test_14567.png` | test | `test_05411.png` |

---

## 5. Dataset After Cleaning

| Split | Before | Removed | After |
|---|---|---|---|
| train | 78,468 | 16 | 78,452 |
| val | 11,219 | 7 | 11,212 |
| test | 22,433 | 3 | 22,430 |
| **Total** | **112,120** | **26** | **112,094** |

---

## 6. Notes

- Duplicates originate from the NIH ChestX-ray14 source dataset.
- Removed images are exact pixel-level duplicates (identical MD5 hash).
- Label metadata CSVs (train.csv, val.csv, test.csv) updated to reflect deletions.
- No label re-assignment performed; only redundant entries removed.