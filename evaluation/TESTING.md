# Metric Testing Documentation

Comprehensive unit tests for all evaluation metrics to ensure correctness, numerical stability, and expected behavior.

## Quick Start

```bash
# Install pytest (if not already installed)
pip install pytest

# Run all tests (excluding slow feature-based tests)
cd evaluation
pytest metric_tests.py -v

# Run all tests including slow ones
pytest metric_tests.py -v -m ""

# Run only fast unit tests
pytest metric_tests.py -v -m "not slow"

# Run specific test class
pytest metric_tests.py::TestKLDivergence -v

# Run specific test
pytest metric_tests.py::TestKLDivergence::test_self_divergence_is_zero -v
```

## Test Organization

### Test Categories

1. **Label Distribution Metrics** (Fast, ~5 seconds)
   - `TestKLDivergence` - 7 tests
   - `TestKSLabelTest` - 6 tests
   - `TestMMDLabelDistributions` - 5 tests
   - `TestMahalanobisDistanceLabels` - 4 tests

2. **MMD RBF Kernel** (Fast, ~2 seconds)
   - `TestMMDRBFKernel` - 5 tests

3. **Feature Distribution Metrics** (Slow, ~60 seconds)
   - `TestFIDDistance` - 5 tests
   - `TestMMDFeatures` - 2 tests
   - `TestWassersteinDistanceFeatures` - 2 tests
   - `TestCramerVonMisesFeatures` - 2 tests
   - `TestKSMultivariateTest` - 5 tests

4. **Edge Cases** (Fast, ~2 seconds)
   - `TestEdgeCases` - 6 tests

5. **Integration Tests** (Fast, ~3 seconds)
   - `TestMetricConsistency` - 3 tests
   - `TestNumericalStability` - 3 tests

**Total**: 55 tests

## What Each Test Validates

### Mathematical Properties

#### Self-Distance Tests
Tests that a metric gives 0 (or near-0) when comparing a distribution with itself.

```python
def test_self_divergence_is_zero():
    """KL(P||P) = 0"""
    assert KL(dataset, dataset) < 1e-6
```

**Why**: Fundamental property of distance/divergence measures.

#### Non-Negativity Tests
Tests that metrics always return non-negative values.

```python
def test_non_negative():
    """metric(P, Q) >= 0 for all P, Q"""
    assert metric(dataset1, dataset2) >= 0
```

**Why**: Required for proper distance/divergence measures.

#### Symmetry Tests
Tests if metric(P, Q) = metric(Q, P).

```python
def test_symmetric():
    """metric(P, Q) = metric(Q, P)"""
    assert abs(metric(P, Q) - metric(Q, P)) < 1e-10
```

**Why**: Some metrics should be symmetric (MMD, FID), others shouldn't (KL).

#### Triangle Inequality
Tests if metric(P, R) ≤ metric(P, Q) + metric(Q, R).

**Why**: Required for proper metrics (Wasserstein satisfies this).

### Expected Behavior Tests

#### Identical Distributions
```python
def test_identical_distributions_near_zero():
    """Similar distributions → low metric value"""
```
**Expected**: Very low values (~0)

#### Complementary Distributions
```python
def test_complementary_distributions_high():
    """Completely different distributions → high metric value"""
```
**Expected**: High values (>0.5 for normalized metrics)

#### Ordering Test
```python
def test_metrics_order_distributions_consistently():
    """identical < slight_diff < very_diff"""
```
**Expected**: Metrics should order distributions by heterogeneity.

### Edge Cases

1. **Empty Datasets** - Should raise error or handle gracefully
2. **Single Sample** - Should work or fail informatively
3. **Single Class** - All samples from one class
4. **Large Classes** - 1000+ classes
5. **Extreme Skew** - 99.9% in one class
6. **Zero Counts** - Some classes have 0 samples

### Known Values

Some tests verify against mathematically computed values:

```python
def test_known_value():
    """
    P = [0.5, 0.5], Q = [0.8, 0.2]
    KL(P||Q) = 0.5*log(0.5/0.8) + 0.5*log(0.5/0.2) ≈ 0.223
    """
    assert abs(computed - 0.223) < 0.05
```

## Test Fixtures

### Synthetic Data Fixtures

1. **`identical_distributions`**
   ```python
   # 100 samples, 10 classes, uniform distribution
   # Split into two equal clients
   dataset1: 50 samples
   dataset2: 50 samples
   ```
   **Use**: Test that metrics give ~0 for identical distributions

2. **`complementary_distributions`**
   ```python
   dataset1: classes [0, 1, 2, 3, 4]
   dataset2: classes [5, 6, 7, 8, 9]
   # No overlap!
   ```
   **Use**: Test that metrics give high values for different distributions

3. **`skewed_distributions`**
   ```python
   dataset1: uniform over 10 classes
   dataset2: 90% class 0, 10% others
   ```
   **Use**: Test behavior with skewed distributions

4. **`single_class_distributions`**
   ```python
   dataset1: all class 0
   dataset2: all class 1
   ```
   **Use**: Test extreme cases

5. **`temp_image_datasets`**
   ```python
   # Creates synthetic 64x64 RGB images
   dataset1: 20 images (solid colors)
   dataset2: 20 images (similar colors, small shift)
   dataset3: 20 images (very different colors, large shift)
   ```
   **Use**: Test feature-based metrics without real datasets

## Running Specific Test Sets

### Fast Tests Only (Label Metrics)
```bash
pytest metric_tests.py -v -m "not slow"
```
**Time**: ~15 seconds
**Tests**: 40 tests (all label metrics + edge cases)

### Slow Tests Only (Feature Metrics)
```bash
pytest metric_tests.py -v -m "slow"
```
**Time**: ~60 seconds
**Tests**: 15 tests (all feature metrics)

### Specific Metric
```bash
# Test only KL Divergence
pytest metric_tests.py::TestKLDivergence -v

# Test only FID
pytest metric_tests.py::TestFIDDistance -v
```

### Failed Tests Only
```bash
# Re-run only failed tests from last run
pytest metric_tests.py --lf -v
```

### With Coverage
```bash
# Install coverage plugin
pip install pytest-cov

# Run with coverage report
pytest metric_tests.py --cov=metrics --cov-report=html
# Open htmlcov/index.html to view coverage
```

## Interpreting Test Results

### Success Example
```
============================= test session starts ==============================
collected 55 items

metric_tests.py::TestKLDivergence::test_self_divergence_is_zero PASSED   [  1%]
metric_tests.py::TestKLDivergence::test_non_negative PASSED              [  3%]
...
========================= 55 passed in 18.42s ==================================
```
✅ All metrics working correctly

### Failure Example
```
FAILED metric_tests.py::TestKLDivergence::test_self_divergence_is_zero - AssertionError: Self-divergence should be ~0, got 0.234

=========================== short test summary info ============================
FAILED metric_tests.py::TestKLDivergence::test_self_divergence_is_zero
```
❌ Bug in KL Divergence implementation

### Common Failures

1. **Numerical Instability**
   ```
   AssertionError: KL should be finite even with zero counts
   ```
   **Fix**: Add epsilon smoothing to avoid log(0)

2. **Wrong Property**
   ```
   AssertionError: MMD should be symmetric, got 0.45 vs 0.32
   ```
   **Fix**: Check MMD implementation for asymmetry bugs

3. **Ordering Failure**
   ```
   AssertionError: KL should order distributions: 0.12 < 0.34 < 0.23
   ```
   **Fix**: Metric not discriminative enough or implementation error

## Adding New Tests

### For a New Metric

1. **Create test class**:
   ```python
   class TestNewMetric:
       """Test NewMetric."""
   ```

2. **Add property tests**:
   ```python
   def test_self_distance_zero(self, identical_distributions):
       """NewMetric(P, P) = 0"""

   def test_non_negative(self, skewed_distributions):
       """NewMetric(P, Q) >= 0"""

   def test_symmetric(self, skewed_distributions):
       """NewMetric(P, Q) = NewMetric(Q, P)"""
   ```

3. **Add behavior tests**:
   ```python
   def test_identical_distributions_near_zero(self, identical_distributions):
       """Similar → low value"""

   def test_complementary_distributions_high(self, complementary_distributions):
       """Different → high value"""
   ```

4. **Add edge cases**:
   ```python
   def test_handles_single_class(self, single_class_distributions):
       """Should work with single class"""
   ```

### For a New Property

```python
def test_new_property(self):
    """Test that metric satisfies new property."""
    # Setup
    dataset1 = ...
    dataset2 = ...

    # Compute
    result = metrics.new_metric(dataset1, dataset2)

    # Assert
    assert property_holds(result), f"Property failed: {result}"
```

## Test Markers

Tests can be marked for selective execution:

```python
@pytest.mark.slow
def test_expensive_computation():
    """This test takes a long time."""

@pytest.mark.unit
def test_pure_function():
    """This is a fast unit test."""

@pytest.mark.integration
def test_full_pipeline():
    """This tests multiple components."""
```

Run marked tests:
```bash
pytest metric_tests.py -v -m "slow"        # Only slow tests
pytest metric_tests.py -v -m "not slow"    # Exclude slow tests
pytest metric_tests.py -v -m "unit"        # Only unit tests
```

## Continuous Integration

Add to `.github/workflows/test.yml`:

```yaml
name: Test Metrics

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install pytest numpy scipy torch torchvision pillow

    - name: Run fast tests
      run: |
        cd evaluation
        pytest metric_tests.py -v -m "not slow"

    - name: Run slow tests (on main branch only)
      if: github.ref == 'refs/heads/main'
      run: |
        cd evaluation
        pytest metric_tests.py -v -m "slow"
```

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'evaluation'
```
**Solution**: Run from evaluation directory or set PYTHONPATH

### CUDA Errors
```
RuntimeError: CUDA out of memory
```
**Solution**: Tests use small datasets, but if OOM occurs:
```python
device = torch.device('cpu')  # Force CPU in tests
```

### Slow Test Timeout
```
ERROR: test session failed after 300s timeout
```
**Solution**: Run slow tests separately or increase timeout

### Fixture Not Found
```
fixture 'temp_image_datasets' not found
```
**Solution**: Check fixture is defined in same file or conftest.py

## Performance Benchmarks

Expected runtimes (on typical laptop, CPU only):

| Test Suite | Tests | Time |
|------------|-------|------|
| Label Metrics | 22 | ~10s |
| MMD Kernel | 5 | ~2s |
| Feature Metrics | 15 | ~60s |
| Edge Cases | 6 | ~2s |
| Integration | 6 | ~3s |
| **Total** | **55** | **~80s** |

Fast tests only (no feature metrics): **~15s**

## Best Practices

1. **Run fast tests frequently** during development
2. **Run slow tests** before committing
3. **Run all tests** before pull requests
4. **Add tests** when fixing bugs
5. **Update tests** when changing metric implementations
6. **Check coverage** to find untested code paths

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [NumPy Testing](https://numpy.org/doc/stable/reference/routines.testing.html)
- [Statistical Tests in SciPy](https://docs.scipy.org/doc/scipy/reference/stats.html)
