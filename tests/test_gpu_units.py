"""The GPU repeatability check: a package is repeatable on CPU_AND_GPU at batch 1, or when its topk is fp32."""
import pytest

from inframe import detect

ct = pytest.importorskip("coremltools")


def _spec(precision):
    from coremltools.converters.mil import Builder as mb

    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 16))], opset_version=ct.target.iOS16)
    def prog(x):
        values, _ = mb.topk(x=mb.add(x=x, y=1.0), k=4, axis=-1)
        return values

    return ct.convert(prog, convert_to="mlprogram", compute_precision=precision, minimum_deployment_target=ct.target.iOS16,
                      skip_model_load=True).get_spec()


def test_topk_precision_of_fp16_fp32_and_mixed_packages():
    fp16 = _spec(ct.precision.FLOAT16)
    fp32 = _spec(ct.precision.FLOAT32)
    mixed = _spec(ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type != "topk"))  # what bench/detect/export_gpu.py does
    assert detect.topk_precision(fp16) == "fp16"
    assert detect.topk_precision(fp32) == "fp32"
    assert detect.topk_precision(mixed) == "fp32"
    assert not detect.gpu_repeatable(fp16, 8) and not detect.gpu_repeatable(fp16, 2)
    assert detect.gpu_repeatable(fp16, 1)  # batch 1 was repeatable with the fp16 topk (base_coreml_b1, small batch 1)
    assert detect.gpu_repeatable(mixed, 8)


def test_topk_precision_of_a_spec_it_cannot_read():
    assert detect.topk_precision(object()) is None
    assert detect.gpu_repeatable(object(), 8)  # no topk: nothing known to be unrepeatable


def test_default_units_are_the_gpu_and_the_others_stay_selectable(monkeypatch):
    monkeypatch.delenv(detect.UNITS_ENV, raising=False)
    assert detect.DEFAULT_UNITS == detect.compute_units_name() == "CPU_AND_GPU"
    for u in ("CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU"):
        monkeypatch.setenv(detect.UNITS_ENV, u.lower())
        assert detect.compute_units_name() == u
    monkeypatch.setenv(detect.UNITS_ENV, "nonsense")
    assert detect.compute_units_name() == "CPU_AND_GPU"


def test_stale_package_warning_names_the_re_export():
    fp16, mixed = _spec(ct.precision.FLOAT16), _spec(ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type != "topk"))
    w = detect.stale_package_warning("/m/small_coreml_b8", "small", 8, "CPU_AND_GPU", fp16)
    assert w and "stale" in w and "export_gpu.py small" in w and "/m/small_coreml_b8" in w
    assert detect.stale_package_warning("/m/p", "small", 8, "CPU_AND_GPU", mixed) is None  # fp32 topk: fine
    assert detect.stale_package_warning("/m/p", "small", 1, "CPU_AND_GPU", fp16) is None  # batch 1 was repeatable
    assert detect.stale_package_warning("/m/p", "small", 8, "CPU_ONLY", fp16) is None  # CPU units do not care
