import torch
from typing import Tuple, Optional
from rich.console import Console
from rich.table import Table


def set_random_seed(seed: int = 42):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)


def stochastic_rounding(value: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    """Perform stochastic rounding on the given value."""
    floor_value = torch.floor(value / epsilon) * epsilon
    ceil_value = floor_value + epsilon
    probability = (value - floor_value) / epsilon
    rand_value = torch.rand_like(value)
    rounded_value = torch.where(rand_value < probability, ceil_value, floor_value)
    return rounded_value


def handle_special_cases(value: float) -> Optional[int]:
    """Handle special cases like zero, NaN, and infinity."""
    if abs(value) < 1e-7:
        return 0
    if torch.isnan(torch.tensor(value)):
        return 0xFF  # NaN
    if torch.isinf(torch.tensor(value)):
        return 0x80 if value < 0 else 0x7F  # -inf or inf
    return None


def scale_and_extract_sign(value: float, scaling_factor: float) -> Tuple[float, int]:
    """Scale the value and extract the sign bit."""
    value_scaled = value * scaling_factor
    sign = 0 if value_scaled >= 0 else 1
    value_scaled = abs(value_scaled)
    return value_scaled, sign


def calculate_exponent_mantissa(
    value_scaled: float, n_mantissa: int, bias: int
) -> Tuple[int, float]:
    """Calculate the exponent and mantissa for the scaled value."""
    if value_scaled == 0:
        return 0, 0.0
    exponent = int(torch.floor(torch.log2(torch.tensor(value_scaled))).item()) + bias
    mantissa = value_scaled / (2 ** (exponent - bias)) - 1
    return exponent, mantissa


def apply_stochastic_rounding(mantissa: float, n_mantissa: int) -> float:
    """Apply stochastic rounding to the mantissa."""
    mantissa_bits = 2**n_mantissa
    epsilon = 1.0 / mantissa_bits
    rounded_mantissa = stochastic_rounding(
        torch.tensor(mantissa), torch.tensor(epsilon)
    ).item()
    return rounded_mantissa


def combine_components(
    sign: int,
    exponent: int,
    mantissa_quantized: float,
    n_mantissa: int,
    exponent_bits: int,
) -> int:
    """Combine sign, exponent, and mantissa into a single FP8 value."""
    mantissa_bits = 2**n_mantissa
    fp8_value = (
        (sign << (exponent_bits + n_mantissa))
        | (exponent << n_mantissa)
        | int(mantissa_quantized * mantissa_bits)
    )
    return fp8_value


def float_to_fp8(value: float, n_mantissa: int, scaling_factor: float) -> int:
    """Convert a float value to FP8 format."""
    special_case = handle_special_cases(value)
    if special_case is not None:
        return special_case

    value_scaled, sign = scale_and_extract_sign(value, scaling_factor)

    bias = 15 if n_mantissa == 2 else 7
    exponent_bits = 5 if n_mantissa == 2 else 4
    exponent, mantissa = calculate_exponent_mantissa(value_scaled, n_mantissa, bias)

    # Handle subnormal numbers
    if exponent <= 0:
        mantissa = value_scaled / (2 ** (1 - bias))
        exponent = 0
    # Handle overflow to infinity
    elif exponent >= (1 << exponent_bits) - 1:
        return 0x80 if sign == 1 else 0x7F

    mantissa_quantized = apply_stochastic_rounding(mantissa, n_mantissa)

    return combine_components(
        sign, exponent, mantissa_quantized, n_mantissa, exponent_bits
    )


def fp8_to_float(
    fp8_value: int, n_mantissa: int, scaling_factor: float
) -> torch.Tensor:
    """Convert an FP8 value to float."""
    if fp8_value == 0:
        return torch.tensor(0.0, dtype=torch.bfloat16)
    if fp8_value == 0xFF:
        return torch.tensor(float("nan"), dtype=torch.bfloat16)
    if fp8_value == 0x80:
        return torch.tensor(float("-inf"), dtype=torch.bfloat16)
    if fp8_value == 0x7F:
        return torch.tensor(float("inf"), dtype=torch.bfloat16)

    sign = (fp8_value >> (4 + n_mantissa)) & 1
    exponent_bits = 4 if n_mantissa == 3 else 5
    bias = 7 if n_mantissa == 3 else 15

    exponent = (fp8_value >> n_mantissa) & ((1 << exponent_bits) - 1)
    mantissa = (fp8_value & ((1 << n_mantissa) - 1)) / (2**n_mantissa)

    if exponent == 0:
        value = mantissa * 2 ** (1 - bias) * scaling_factor
    else:
        value = (mantissa + 1) * 2 ** (exponent - bias) * scaling_factor

    if sign == 1:
        value = -value

    return torch.tensor(value, dtype=torch.bfloat16)


@torch.jit.script
def round_to_fp8_represented_as_int8(
    t: torch.Tensor,
    n_mantissa: int,
    out: Optional[torch.Tensor] = None,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Convert a bfloat16 tensor to FP8 representation as uint8 with stochastic rounding."""
    if out is None:
        out = torch.empty(t.numel(), dtype=torch.uint8)

    for i in range(t.numel()):
        out[i] = float_to_fp8(
            t.to(torch.bfloat16)[i].item(), n_mantissa, scaling_factor
        )

    return out.view(t.shape)


@torch.jit.script
def undo_int8_fp8(
    fp8_tensor: torch.Tensor,
    n_mantissa: int,
    target_dt: torch.dtype,
    out: Optional[torch.Tensor] = None,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Convert an FP8 tensor represented as uint8 back to bfloat16."""
    if out is None:
        out = torch.empty(fp8_tensor.numel(), dtype=target_dt)

    for i in range(fp8_tensor.numel()):
        out[i] = fp8_to_float(
            fp8_tensor.to(torch.uint8)[i].item(), n_mantissa, scaling_factor
        )

    return out.view(fp8_tensor.shape)


def simulate_stochastic_rounding_test(
    n_mantissa: int,
    scaling_factor: float,
    num_runs: int,
    tensor_size: int,
    rtol: float = 1e-2,
) -> bool:
    """Simulate the stochastic rounding test with a specified tensor size."""
    set_random_seed()
    console = Console()
    
    # Create config table
    config_table = Table(title="Test Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("n_mantissa", str(n_mantissa))
    config_table.add_row("scaling_factor", str(scaling_factor))
    config_table.add_row("num_runs", str(num_runs))
    config_table.add_row("tensor_size", str(tensor_size))
    console.print(config_table)

    original_tensor = torch.randn(tensor_size, dtype=torch.bfloat16)
    
    # Progress table
    results_table = Table(title="Simulation Results")
    results_table.add_column("Run", style="cyan")
    results_table.add_column("Mean Recovered Value", style="green")
    
    recovered_sum = torch.zeros_like(original_tensor, dtype=torch.float32)
    for run in range(num_runs):
        fp8_tensor = round_to_fp8_represented_as_int8(
            original_tensor, n_mantissa, scaling_factor=scaling_factor
        )
        recovered_values = undo_int8_fp8(
            fp8_tensor, n_mantissa, torch.float32, scaling_factor=scaling_factor
        )
        recovered_sum += recovered_values

        if (run + 1) % (num_runs // 10) == 0:
            results_table.add_row(
                str(run + 1), 
                f"{recovered_sum.mean():.6f}"
            )
    
    console.print(results_table)

    recovered_avg = recovered_sum / num_runs
    original_tensor_float32 = original_tensor.to(torch.float32)
    all_close = torch.allclose(original_tensor_float32, recovered_avg, rtol=rtol)

    # Final comparison table
    comparison_table = Table(title="Final Results")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Value", style="green")
    
    comparison_table.add_row("Original Mean", f"{original_tensor_float32.mean():.6f}")
    comparison_table.add_row("Recovered Mean", f"{recovered_avg.mean():.6f}")
    comparison_table.add_row("Maximum Difference", f"{torch.abs(original_tensor_float32 - recovered_avg).max():.6f}")
    comparison_table.add_row("Test Result", "✅ PASSED" if all_close else "❌ FAILED")
    
    console.print(comparison_table)

    if not all_close:
        # Error details table
        error_table = Table(title="Large Differences")
        error_table.add_column("Index", style="cyan")
        error_table.add_column("Original", style="green")
        error_table.add_column("Recovered", style="yellow")
        error_table.add_column("Difference", style="red")
        
        diff = torch.abs(original_tensor_float32 - recovered_avg)
        large_diff_indices = (diff > rtol).nonzero(as_tuple=True)[0]
        for i in large_diff_indices[:10]:
            error_table.add_row(
                str(i.item()),
                f"{original_tensor_float32[i]:.6f}",
                f"{recovered_avg[i]:.6f}",
                f"{diff[i]:.6f}"
            )
        console.print(error_table)

    return all_close


# Perform the simulation test with a large tensor size
# simulate_stochastic_rounding_test(n_mantissa=3, scaling_factor=1.0, num_runs=10, tensor_size=100000)

# Perform the reduced test with a smaller tensor size
simulate_stochastic_rounding_test(
    n_mantissa=3, scaling_factor=1.0, num_runs=10, tensor_size=1000
)
