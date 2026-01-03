# Project Evaluation Report - Violations Only

**Date:** 03.01.2026  
**Project:** Scam Detection  
**Evaluated Against:** Project.md requirements

---

## CRITICAL VIOLATIONS (-3 points each)

### 1. Model Checkpoint File in Git (-3 points)
**Violation:** Data/model files in Git  
**Requirement:** "No data/model files in Git (.json, .csv, .h5, .cbm, .pth, .onnx, etc.)"

**Evidence:**
```bash
$ git ls-files | grep -E "\.(ckpt|pth|onnx|pkl|h5)$"
models/checkpoints/transformer-epoch=00-val_f1=0.75.ckpt

$ ls -lh models/checkpoints/transformer-epoch=00-val_f1=0.75.ckpt
-rw-r--r-- 83M models/checkpoints/transformer-epoch=00-val_f1=0.75.ckpt
```

**Impact:** 83MB model checkpoint tracked in git repository

**Fix:** Remove from git and add to .gitignore
```bash
git rm --cached models/checkpoints/*.ckpt
echo "*.ckpt" >> .gitignore
```

---

## MAJOR DEFICIENCIES (0 points earned)

### 2. TensorRT Conversion Not Implemented (0/5 points)
**Requirement:** "TensorRT Conversion (5 points): Convert from ONNX or directly from model using shell/Python CLI"

**Evidence:**
- No TensorRT conversion script found
- No `.trt` files in project
- No TensorRT code in `scam_detection/` directory
- Search for `tensorrt|TensorRT` in Python files: 0 results in project code (only in dependencies)

**Status:** Not implemented

---

### 3. ONNX Export Broken (0/5 points instead of 5/5)
**Requirement:** "ONNX Conversion (5 points): Convert model to ONNX format"

**Evidence:**
```bash
$ uv run python scripts/export_onnx.py <model> <output> <tokenizer> <type>
ModuleNotFoundError: No module named 'onnxscript'
```

**Problem:** Missing dependency `onnxscript` required for ONNX export

**Built-in command also fails:**
```bash
$ python -m scam_detection.commands export ...
hydra.errors.OverrideParseException: mismatched input '=' expecting <EOF>
```

**Status:** Implementation exists but non-functional

---

### 4. MLflow Serving Returns Mock Data (2/10 points instead of 5/10)
**Requirement:** "Inference Server: MLflow Serving (5 points max)"

**Evidence from `scam_detection/serving/mlflow_model.py`:**
```python
def predict(self, context, model_input):
    # ... extract texts
    
    # Implement prediction logic
    # For simplicity, return mock predictions
    return [0] * len(texts)  # ❌ ALWAYS RETURNS ZEROS
```

**Status:** Wrapper exists but not functional (returns mock predictions)

---

### 5. Triton Inference Server Not Implemented (0/10 points)
**Requirement:** "Triton Inference Server (10 points max)"

**Evidence:**
- Search for `triton|Triton` in Python files: 0 results
- No Triton configuration files
- No model repository setup

**Status:** Not implemented

---

## MINOR VIOLATIONS (-1 point)

### 6. Incorrect Terminology: "email" instead of "message"
**Requirement:** Checklist Flag #1: Use "messages" terminology, not "emails"

**Evidence from `scripts/export_onnx.py` line 32:**
```python
dummy_input = tokenizer(
    "This is a test email.",  # ❌ Should be "message"
    truncation=True,
    ...
)
```

**Impact:** Minor - only in test data

---

## DOCUMENTATION ISSUE

### 7. README Documents Unimplemented Feature
**Finding:** README.md mentions "TensorRT" in features but TensorRT is not implemented

**Evidence:**
- README line 7: "Production-ready model export to ONNX and TensorRT formats"
- README Production Preparation section documents TensorRT
- No actual TensorRT implementation exists

**Impact:** Misleading documentation

---

## SUMMARY OF VIOLATIONS

| Violation | Type | Points Lost |
|-----------|------|-------------|
| Model checkpoint in git | Critical | -3 |
| TensorRT not implemented | Missing feature | -5 |
| ONNX export broken | Broken feature | -5 |
| MLflow returns mock data | Incomplete feature | -3 |
| Triton not implemented | Missing feature | -10 |
| "email" terminology | Minor | -1 |
| **TOTAL DEDUCTIONS** | | **-27 points** |

**Note:** ONNX was originally working but is currently broken due to missing `onnxscript` dependency.

---

## IMMEDIATE FIXES REQUIRED

### Priority 1: Critical (Blocker)
1. **Remove model checkpoint from git** (-3 points)
   ```bash
   git rm --cached models/checkpoints/transformer-epoch=00-val_f1=0.75.ckpt
   echo "*.ckpt" >> .gitignore
   git commit -m "Remove model checkpoint from git"
   ```

### Priority 2: Broken Features
2. **Fix ONNX export** (-5 points)
   - Add `onnxscript` to pyproject.toml dependencies
   - Test export works
   
3. **Fix MLflow serving** (-3 points)
   - Implement real predictions in `mlflow_model.py`
   - Remove mock `return [0] * len(texts)` line

### Priority 3: Missing Features
4. **Implement TensorRT conversion** (-5 points)
   - Create conversion script
   - Document usage

5. **Fix terminology** (-1 point)
   - Change "email" to "message" in export_onnx.py

---

## WHAT'S WORKING

✅ Pre-commit hooks pass  
✅ PyTorch Lightning integration  
✅ DVC integration  
✅ Hydra configuration  
✅ MLflow logging  
✅ Training pipeline  
✅ CLI structure (4 commands)  
✅ No argparse usage  
✅ No warnings.filterwarnings  
✅ Proper naming conventions  
✅ CSV data not in git (properly managed by DVC)

---

**Current Estimated Score:** 48-52/75 (64-69%) - Grade D/C  
**With Critical Fix:** 51-55/75 (68-73%) - Grade C  
**With All Fixes:** 75+/75 (>90%) - Grade A

**Report Generated:** 03.01.2026, 23:01  
**Method:** Automated testing + code scanning
