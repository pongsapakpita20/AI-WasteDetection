# AI Waste Sorter - DVC Pipeline Guide

## วิธีใช้งาน DVC Pipeline

### วิธีที่ 1: ใช้ Batch Script (ง่ายที่สุด)

```bash
# รัน pipeline ทั้งหมด
run_pipeline.bat
```

### วิธีที่ 2: ใช้ DVC Command โดยตรง

#### รัน Pipeline ทั้งหมด:

```bash
dvc repro evaluate
```

จะทำ:
1. `train` - เทรนโมเดล
2. `promote_best` - promote best.pt
3. `evaluate` - ประเมินโมเดล

#### รันเฉพาะ Stage:

```bash
# เทรนโมเดลเท่านั้น
dvc repro train

# ประเมินโมเดลเท่านั้น (ต้องมีโมเดลแล้ว)
dvc repro evaluate
```

#### Push/Pull Artifacts:

```bash
# อัปโหลด artifacts ไป DVC storage
dvc push

# ดึง artifacts จาก DVC storage
dvc pull waste-detection.dvc
```

---

## Pipeline Stages

### Stage 1: `train`
- รัน `python train.py` - เทรนโมเดล YOLO
- รัน `python promote_best.py` - promote best.pt ไป `artifacts/models/`
- Output: `artifacts/models/waste-sorter-best.pt`

### Stage 2: `evaluate`
- รัน `python evaluate.py` - ประเมินโมเดล
- Output: `artifacts/eval/metrics.json`

---

## DVC Storage

- **Remote Type:** Local storage
- **Location:** `D:\DVC-Storage`
- **Config:** `.dvc/config`

**หมายเหตุ:** ไม่จำเป็นต้องมี DVC-Storage เพราะโมเดลถูกเก็บใน Git repository แล้ว

---

## Parameters

แก้ไขพารามิเตอร์ได้ใน `params.yaml`:
- `train.*` - พารามิเตอร์การเทรน
- `evaluate.*` - พารามิเตอร์การประเมิน

---

## Troubleshooting

### DVC ไม่พบคำสั่ง
```bash
pip install dvc
```

### Pipeline ไม่รัน (ทุกอย่าง up-to-date)
```bash
# Force rerun
dvc repro -f evaluate
```

### ดู Pipeline Graph
```bash
dvc dag
```

### ดู Pipeline Status
```bash
dvc status
```

---

## หมายเหตุ

- Pipeline จะรันเฉพาะเมื่อ dependencies หรือ parameters เปลี่ยน
- ถ้าต้องการรันซ้ำโดยไม่เปลี่ยนอะไร ให้ใช้ `dvc repro -f`
- Artifacts จะถูกเก็บใน `D:\DVC-Storage` (ตาม config) - แต่ไม่จำเป็นต้องมีเพราะโมเดลอยู่ใน Git แล้ว
