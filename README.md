# FP8 Auto-Converter

A Work in Progress PyTorch implementation for converting between bfloat16 and FP8 formats, using **only native PyTorch and NumPy**.

## Project Goals

This project aims to:

1. **Convert Tensors**: 
   - bfloat16 -> FP8 (stored as uint8) -> back to bfloat16
   - Using pure bit manipulation (no .to(torch.uint8))
   - N-bit mantissa support (configurable)

2. **Implement Stochastic Rounding**:
   - Following [Higham's approach](https://nhigham.com/2020/07/07/what-is-stochastic-rounding/)
   - Preserve statistical properties
   - Mean preservation over multiple iterations

3. **Handle Edge Cases**:
   - NaN, Infinity
   - Denormal numbers
   - Zero values
   - Sign preservation

4. **Comprehensive Testing**:
   - Statistical property validation
   - Edge case verification
   - Large-scale iteration tests


> 🚀 **Good Start, But Beware**: While the basic implementation is in place, there are critical issues with statistical properties and sign bit handling. Perfect for learning about FP8 and numerical methods, but not yet production-ready!

## Current Status: Help Needed! 🚨

This implementation currently fails critical statistical tests. We're looking for help from the community to fix core issues:

```python
# Current behavior (problematic):
tensor([  2.3594,  20.7500, -12.5000]) ->
tensor([ -2.3594, -20.7500, -12.5000])  # Complete sign flips!
```

### Key Issues to Solve

1. **Sign Bit Flipping**:
   - Values are getting their signs reversed during conversion
   - Needs urgent fix in bit manipulation logic

2. **Statistical Test Failure**:
   ```python
   # Test that fails:
   assert torch.allclose(original, recovered_avg, rtol=1e-2)
   # Running tensor through 100k iterations should preserve mean
   ```

3. **Stochastic Rounding Issues**:
   - Current implementation doesn't preserve statistical properties
   - Mean values drift significantly over multiple iterations

## How You Can Help

1. **Core Areas Needing Review**:
   - Sign bit handling in `float_to_fp8` function
   - Stochastic rounding implementation
   - Statistical properties preservation

2. **Testing**:
   ```python
   # Help us fix this test:
   def test_statistical_properties(tensor_size=1000, num_runs=100000):
       x = torch.randn(tensor_size, dtype=torch.bfloat16)
       output_avg = average_over_runs(x, num_runs)
       assert torch.allclose(x, output_avg, rtol=1e-2)  # Currently failing
   ```

3. **Contributions Welcome**:
   - Bug fixes
   - Code reviews
   - Test cases
   - Documentation improvements

## Getting Started

```python
# Current implementation (needs fixing):
from fp8_auto import round_to_fp8_represented_as_int8, undo_int8_fp8

# Help us fix these functions
fp8_tensor = round_to_fp8_represented_as_int8(tensor, n_mantissa=3)
recovered = undo_int8_fp8(fp8_tensor, n_mantissa=3, target_dt=torch.bfloat16)
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Run the test suite
4. Submit a pull request

We especially need help with:
- [ ] Fixing sign bit preservation
- [ ] Improving stochastic rounding accuracy
- [ ] Adding comprehensive statistical tests
- [ ] Optimizing performance

## Community

- Open an issue for discussion
- Join [Discord](https://discord.gg/gpumode) 
- Share your expertise

Let's make this work together! 🚀

