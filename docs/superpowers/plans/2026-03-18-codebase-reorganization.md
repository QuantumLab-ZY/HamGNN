# HamGNN Codebase Reorganization Plan

> **Goal:** Organize the HamGNN codebase for better maintainability, clarity, and ease of use.

**Architecture Summary:** This is a scientific Python project for Hamiltonian Graph Neural Networks. The codebase consists of:
- Core ML framework (`HamGNN_v_2_1/`)
- DFT software interfaces (`utils_openmx/`, `utils_siesta/`, `utils_abacus/`)
- Utility tools (`band_cal_parallel/`, `npz_to_lmdb.py`)
- Documentation (`docs/`, `config_examples/`)

---

## Current Issues Identified

### 1. Directory Structure Issues
- `HamGNN_v_2_1/` - Package name contains version number (non-standard)
- `npz_to_lmdb.py` - Utility script in root directory
- `utils_openmx/`, `utils_siesta/`, `utils_abacus/` - DFT interfaces scattered in root
- `openmx_postprocess/` contains both source code and compiled binaries

### 2. Large Binary Files (Git Inefficiency)
| File | Size | Type |
|------|------|------|
| `utils_siesta/honpas_1.2_H0.zip` | 52MB | Modified DFT source |
| `utils_abacus/abacus_H0_export/abacus-postprocess-v353_source.tar.gz` | 18MB | Modified DFT source |

### 3. Deprecated Files
- `utils_abacus/utils_abacus_deprecated.tar.gz` - Explicitly marked deprecated

### 4. Missing Documentation
- No `__init__.py` docstrings explaining modules
- No module-level README files

### 5. Incomplete .gitignore
- Only covers basic Python build artifacts

---

## Proposed Reorganization

### New Directory Structure
```
HamGNN/
├── hamgnn/                    # Renamed: Core package (no version in name)
│   ├── __init__.py
│   ├── main.py
│   ├── version.py
│   ├── models/
│   ├── nn/
│   ├── physics/
│   ├── toolbox/
│   ├── utils/
│   └── config/
├── interfaces/                # New: Consolidated DFT interfaces
│   ├── openmx/
│   │   ├── postprocess/        # Source + binaries
│   │   ├── graph_data_gen.py
│   │   ├── poscar2openmx.py
│   │   └── utils.py
│   ├── siesta/
│   │   ├── graph_data_gen_siesta.py
│   │   ├── poscar2siesta.py
│   │   ├── read_siesta.py
│   │   ├── hsx_tools/         # hsxdump source
│   │   └── honpas_1.2_H0.zip # Consider moving to releases
│   └── abacus/
│       ├── graph_data_gen_abacus.py
│       ├── poscar2abacus.py
│       ├── read_abacus.py
│       └── abacus_H0_export/  # abacus-postprocess source
├── tools/                     # New: Consolidated tools
│   ├── band_cal_parallel/
│   ├── npz_to_lmdb.py
│   └── (other utilities)
├── examples/                  # New: Configuration examples
│   ├── V1.0/
│   └── V2.x/
├── docs/
├── Uni-HamGNN/
├── LICENSE
├── README.md
├── setup.py
├── HamGNN.yaml               # Conda environment
└── .gitignore
```

### Proposed Changes Summary

| Action | From | To | Notes |
|--------|------|-----|-------|
| Rename | `HamGNN_v_2_1/` | `hamgnn/` | Python convention: lowercase |
| Move | `npz_to_lmdb.py` | `tools/` | Consolidate utilities |
| Move | `utils_openmx/` | `interfaces/openmx/` | Group by function |
| Move | `utils_siesta/` | `interfaces/siesta/` | Group by function |
| Move | `utils_abacus/` | `interfaces/abacus/` | Group by function |
| Move | `config_examples/` | `examples/` | Shorter name |
| Move | `band_cal_parallel/` | `tools/` | Consolidate tools |
| Delete | `utils_abacus_deprecated.tar.gz` | - | Explicitly deprecated |
| Create | Module `__init__.py` files | - | Add docstrings |
| Update | `setup.py` | - | Update package references |
| Update | `entry_points` | - | Update console script paths |
| Update | `.gitignore` | - | Add more patterns |
| Update | README.md | - | Update paths to reflect new structure |

---

## Task Breakdown

### Phase 1: Cleanup (Low Risk)

- [ ] **Task 1.1:** Delete deprecated file `utils_abacus/utils_abacus_deprecated.tar.gz`

### Phase 2: Directory Reorganization (Medium Risk)

- [ ] **Task 2.1:** Create new directory structure
  ```bash
  mkdir -p interfaces/openmx interfaces/siesta interfaces/abacus tools
  ```

- [ ] **Task 2.2:** Move and rename `HamGNN_v_2_1/` → `hamgnn/`
  ```bash
  mv HamGNN_v_2_1 hamgnn
  ```

- [ ] **Task 2.3:** Move DFT interface directories
  ```bash
  mv utils_openmx/* interfaces/openmx/
  mv utils_siesta/* interfaces/siesta/
  mv utils_abacus/* interfaces/abacus/
  ```

- [ ] **Task 2.4:** Move utilities
  ```bash
  mv npz_to_lmdb.py tools/
  mv band_cal_parallel tools/
  mv config_examples examples/
  ```

- [ ] **Task 2.5:** Clean up empty directories
  ```bash
  rmdir utils_openmx utils_siesta utils_abacus
  ```

### Phase 3: Configuration Updates (Critical)

- [ ] **Task 3.1:** Update `setup.py` package paths
  - Change `HamGNN_v_2_1` → `hamgnn`
  - Update entry_points paths

- [ ] **Task 3.2:** Update `setup.py` console script references
  ```python
  entry_points={
      "console_scripts": [
          "HamGNN2.0 = hamgnn.main:HamGNN",
          "band_cal = interfaces.openmx.band_cal:main",
          "graph_data_gen = interfaces.openmx.graph_data_gen:main",
          "poscar2openmx = interfaces.openmx.poscar2openmx:main"
      ]
  },
  ```

- [ ] **Task 3.3:** Update `HamGNN.yaml` reference paths

- [ ] **Task 3.4:** Update all internal imports within `hamgnn/` package

- [ ] **Task 3.5:** Update all imports in `interfaces/` modules

### Phase 4: Documentation

- [ ] **Task 4.1:** Add docstrings to `hamgnn/__init__.py`
- [ ] **Task 4.2:** Add docstrings to `interfaces/openmx/__init__.py`
- [ ] **Task 4.3:** Add docstrings to `interfaces/siesta/__init__.py`
- [ ] **Task 4.4:** Add docstrings to `interfaces/abacus/__init__.py`
- [ ] **Task 4.5:** Add docstrings to `tools/__init__.py`

### Phase 5: .gitignore Enhancement

- [ ] **Task 5.1:** Expand `.gitignore` with comprehensive patterns

### Phase 6: README Updates

- [ ] **Task 6.1:** Update README.md to reflect new directory structure
- [ ] **Task 6.2:** Update all path references in README

---

## Files to Modify

### Modify (Configuration)
- `setup.py` - Package name, entry_points, install_requires
- `HamGNN.yaml` - Path references (if any)
- `.gitignore` - Add more patterns
- `README.md` - Update paths

### Modify (Internal Imports)
- `hamgnn/main.py`
- `hamgnn/models/*.py`
- `hamgnn/nn/*.py`
- `hamgnn/physics/*.py`
- `hamgnn/toolbox/**/*.py`
- `hamgnn/utils/*.py`
- `interfaces/openmx/*.py`
- `interfaces/siesta/*.py`
- `interfaces/abacus/*.py`
- `tools/npz_to_lmdb.py`
- `Uni-HamGNN/Uni-HamiltonianPredictor.py`

### Create
- `hamgnn/__init__.py`
- `interfaces/__init__.py`
- `interfaces/openmx/__init__.py`
- `interfaces/siesta/__init__.py`
- `interfaces/abacus/__init__.py`
- `tools/__init__.py`
- `tools/band_cal_parallel/__init__.py` (if needed)

### Delete
- `utils_abacus/utils_abacus_deprecated.tar.gz`

---

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| Phase 1 | Low | Just deleting deprecated file |
| Phase 2 | Medium | Need to verify all moves work |
| Phase 3 | High | Many imports need updating |
| Phase 4 | Low | Documentation only |
| Phase 5 | Low | .gitignore changes |
| Phase 6 | Medium | README must stay accurate |

**Critical Path:** Phase 3 (Configuration Updates) is the most critical and must be done carefully with testing after each sub-task.

---

## Testing After Changes

After Phase 2-3, must verify:
1. `python setup.py install` works
2. `HamGNN --help` runs
3. `band_cal --help` runs
4. `graph_data_gen --help` runs
5. `import hamgnn` succeeds in Python
6. All tests pass (if any exist)

---

## Alternative Approach (Less Disruptive)

If full reorganization is too risky, consider a **lighter touch**:

1. Keep directory structure as-is
2. Just add `__init__.py` docstrings to existing modules
3. Improve `.gitignore`
4. Delete only clearly deprecated files
5. Rename `HamGNN_v_2_1` → `hamgnn` only (most important change)

**This reduces risk but provides 60% of the benefit.**
