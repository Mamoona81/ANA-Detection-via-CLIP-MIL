# Contributing Guidelines

## Overview

This repository welcomes contributions, feedback, and improvements! Whether you're fixing bugs, improving documentation, or proposing new features, please follow the guidelines below.

## How to Contribute

### 1. Report Issues

**Found a bug?** Open a GitHub issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Python/PyTorch version info

**Example:**
```
Title: Model inference crashes on GPU with batch_size > 64

Description:
When using batch_size=128 on GPU, inference fails with RuntimeError.
- Steps: Run `python inference.py --batch_size 128 --device cuda`
- Expected: Process 128 samples
- Actual: CUDA out of memory error
- Environment: Python 3.9, PyTorch 2.0.1, CUDA 11.8
```

### 2. Suggest Improvements

**Ideas for enhancements?** Open a discussion or issue with:
- Clear title
- Motivation/use case
- Proposed solution (if applicable)
- Links to relevant papers/references

**Examples:**
- Focal loss to handle class imbalance
- Attention mechanisms for interpretability
- End-to-end fine-tuning support
- Performance optimizations

### 3. Submit Code Changes

**For bug fixes or features:**

1. **Fork the repository**
   ```bash
   git clone https://github.com/Mamoona81/ANA-Detection-via-CLIP-MIL.git
   cd ANA-Detection-via-CLIP-MIL
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

3. **Make changes**
   - Follow code style (see below)
   - Add docstrings to new functions
   - Test your changes locally

4. **Commit with clear messages**
   ```bash
   git commit -m "Fix: Address class imbalance with focal loss"
   git commit -m "Feature: Add per-class threshold optimization"
   ```

5. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a PR on GitHub with:
   - Clear title and description
   - Reference to any related issues
   - Summary of changes

## Code Style Guidelines

### Python Style

- **PEP 8:** Follow Python Enhancement Proposal 8
- **Line Length:** Max 100 characters
- **Naming:**
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`

### Documentation

- **Docstrings:** Google-style format
  ```python
  def process_embedding(embedding: np.ndarray, normalize: bool = True) -> torch.Tensor:
      """
      Process CLIP embedding for MIL model.
      
      Args:
          embedding: Input array of shape (257, 768)
          normalize: Whether to apply L2-normalization
      
      Returns:
          PyTorch tensor of shape (257, 768)
      
      Raises:
          ValueError: If embedding shape is invalid
      """
  ```

- **Comments:** Explain WHY, not WHAT
  ```python
  # Good: Explains reasoning
  # Use max pooling instead of mean to capture presence/absence of patterns (MIL principle)
  x = torch.max(x, dim=1).values
  
  # Bad: States obvious
  # Set x to maximum value across dimension 1
  x = torch.max(x, dim=1).values
  ```

### File Organization

```
src/
├── mil_pipeline.py          # Main pipeline
├── data_loader.py           # Data utilities
├── mil_evaluate.py          # Evaluation
├── classification.py        # Classification helpers
└── evaluate.py              # General helpers
```

New files should follow this structure and import pattern.

## Testing

### Before Submitting

1. **Test locally**
   ```bash
   # Run with sample data
   python src/mil_pipeline.py
   
   # Test inference
   python -c "from src.mil_pipeline import ANAMILModel; model = ANAMILModel(); print('✓ Model loads')"
   ```

2. **Check imports**
   ```bash
   python -c "import src.mil_pipeline; import src.data_loader; print('✓ Imports OK')"
   ```

3. **Validate data loading**
   ```bash
   python -c "from src.data_loader import ANAFeatureDataset; ds = ANAFeatureDataset('data/features_index_english.csv', 'data', 'train'); print(f'✓ Loaded {len(ds)} samples')"
   ```

## Documentation Updates

When modifying code or adding features:

1. **Update docstrings** in the function/class
2. **Update TECHNICAL.md** if architecture changes
3. **Update README.md** if usage changes
4. **Update INFERENCE.md** if inference API changes
5. **Update DATA.md** if data format changes

## Reporting Security Issues

⚠️ **Do NOT open public issues for security vulnerabilities.**

Instead, email Mamoona Nisar with:
- Description of vulnerability
- Impact assessment
- Proposed fix (if applicable)

## Areas for Contribution

### High Priority

- [ ] **Focal loss implementation** for class imbalance
- [ ] **Per-class threshold optimization** on validation set
- [ ] **Calibration methods** (temperature scaling)
- [ ] **Batch inference optimization** (GPU utilization)

### Medium Priority

- [ ] **Attention mechanisms** for interpretability
- [ ] **Self-supervised pre-training** module
- [ ] **Unit tests** for core functions
- [ ] **Type hints** throughout codebase

### Lower Priority

- [ ] **Additional visualization utilities**
- [ ] **Config file support** (YAML/JSON)
- [ ] **Docker setup** for reproducibility
- [ ] **Comprehensive logging**

## Code Review Process

1. **Automated checks:** Code is tested for imports and basic syntax
2. **Manual review:** Code style, logic, and alignment with project goals
3. **Testing:** Changes are validated on sample data
4. **Feedback:** Reviewers provide constructive suggestions
5. **Approval:** Once approved, PR can be merged

## Communication

- **Issues:** Ask questions directly in GitHub issues
- **Discussions:** Use GitHub Discussions for open-ended topics
- **Email:** For sensitive/urgent matters

## Attribution

Contributors will be acknowledged in:
- GitHub "Contributors" tab
- Project CONTRIBUTORS.md file (if created)
- Release notes for significant contributions

## License

By contributing, you agree that your contributions are licensed under the same [MIT License](LICENSE) as the project.

## Examples of Good Contributions

### Example 1: Bug Fix
```python
# Before: Hard-coded path
DATA_DIR = "d:/path/to/data"

# After: Relative path
DATA_DIR = Path(__file__).parent.parent / "data"
```
**PR Title:** "Fix: Use relative paths for cross-platform compatibility"

### Example 2: Feature Addition
```python
# New module: focal_loss.py
def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """
    Focal loss for addressing class imbalance.
    Based on Lin et al. (2017), Focal Loss for Dense Object Detection.
    """
    ...
```
**PR Title:** "Feature: Add focal loss for class imbalance handling"

### Example 3: Documentation
- Added comprehensive examples in `docs/INFERENCE.md`
- Fixed typo in `docs/TECHNICAL.md`
- Clarified data loading in README

---

**Questions?** Open an issue or contact Mamoona Nisar.

**Thank you for contributing!** 🎉
