"""Application-level tests for native Triton channels, not channel kernel tests."""
import numpy as np
import pytest

from spd.ansatz import binary_mera_parameter_shape
from examples.gradient.run_1d_tfi_hva_tmera import exact_value_gradient,spd_value_gradient


def _available():
    try:
        import torch
        from spd import triton_backend
        return torch.cuda.is_available() and callable(getattr(triton_backend,'contract_zero_forward',None))
    except (ImportError,RuntimeError):
        return False


@pytest.mark.skipif(not _available(),reason='native Triton channels and a CUDA GPU required')
@pytest.mark.parametrize('n,steps,seed,sigma',[(8,1,7,.05),(8,2,19,.15),(16,1,7,.05)])
def test_native_triton_hva_mera_matches_independent_exact_reference(n,steps,seed,sigma):
    params=np.random.default_rng(seed).normal(0,sigma,binary_mera_parameter_shape(n,steps))
    expected,gradient=exact_value_gradient(params,n,4,.9)
    value,mapped,diagnostics=spd_value_gradient(params,n,4,.9,channels=True,backend_name='triton')
    assert value == pytest.approx(expected,rel=0,abs=1e-9)
    np.testing.assert_allclose(mapped,gradient,rtol=0,atol=1e-8)
    assert diagnostics['backend']=='triton'
    assert diagnostics['term_cap']==min(4**n,1000000)
    assert diagnostics['discarded_terms_sum']==0
    assert diagnostics['backward_discarded_terms_sum']==0
    assert diagnostics['qubit_initialization_widths']==([8,4,2,0] if n==8 else [16,8,4,2,0])
    assert diagnostics['peak_allocated_bytes']>0
