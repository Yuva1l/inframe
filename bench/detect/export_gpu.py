"""Export an RF-DETR CoreML package whose GPU path is repeatable: fp16 everywhere except the one topk op, kept fp32.

    bench/detect/.venv/bin/python bench/detect/export_gpu.py small [--batch 8] [--out bench/detect/out/models] [--checkpoint PATH] [--force]

Why (docs/BENCHMARK.md section 22): with an fp16 topk (the two-stage query selection, top 300 of 1024 encoder proposals per image),
CPU_AND_GPU gives a different result on every call of the same batch-8 input (10 distinct outputs in 10 calls; person score up to 0.92
apart on a query between two passes), for any batch > 1 (batch 2 too) and not for batch 1. Keeping only the topk in fp32 makes the GPU
path bit-identical call to call, process to process, whatever the batch neighbours or the padding, at the same speed; CPU_ONLY outputs
of the package are bit-identical to the plain fp16 export, so it replaces <size>_coreml_b<batch> for every compute unit.

Same recipe as bench.py's export (rfdetr RFDETR<Size>.export(format="coreml", batch_size, coreml_precision="float16")), with
coremltools' FP16ComputePrecision(op_selector) skipping the topk. The new package goes to <out>/<size>_coreml_b<batch>/ ; an existing
package there is moved to <out>/old_<size>_coreml_b<batch>/ first (outside the detector's look-up); one that already has an fp32 topk is
kept unless --force. Uses the bench/detect venv
(rfdetr 1.11.2, torch 2.14.1, coremltools 9.0 were used for the measurements).
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile
import time

CLASSES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "base": "RFDETRBase"}
FP32_OPS = {"topk"}  # MIL op types kept in fp32


def topk_is_fp32(folder: str) -> bool:
    """Whether the package in `folder` already has an fp32 topk input (the check inframe.detect.topk_precision makes)."""
    import coremltools as ct
    from coremltools.proto import MIL_pb2

    pk = glob.glob(os.path.join(folder, "**", "*.mlpackage"), recursive=True)
    if not pk:
        return False
    f = ct.models.MLModel(pk[0], skip_model_load=True).get_spec().mlProgram.functions["main"]
    ops = f.block_specializations[f.opset].operations
    dtypes = {o.name: o.type.tensorType.dataType for op in ops for o in op.outputs}
    return any(op.type == "topk" and dtypes.get(op.inputs["x"].arguments[0].name) == MIL_pb2.DataType.Value("FLOAT32") for op in ops)


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("size", choices=sorted(CLASSES))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=os.environ.get("DETECT_MODELS", os.path.join(here, "out", "models")))
    ap.add_argument("--checkpoint", help="the RF-DETR .pth (default: rf-detr-<size>.pth in the inframe cache ~/Library/Caches/inframe/models, else in ~/.roboflow/models where fetch_weights.py puts it)")
    ap.add_argument("--force", action="store_true", help="export again even when the package already has an fp32 topk")
    a = ap.parse_args(argv)
    dest = os.path.join(a.out, f"{a.size}_coreml_b{a.batch}")
    if not a.force and topk_is_fp32(dest):
        print(f"{dest} already has an fp32 topk; nothing to do (--force exports again)")
        return 0
    cands = [a.checkpoint] if a.checkpoint else [os.path.expanduser(f"~/{d}/rf-detr-{a.size}.pth") for d in ("Library/Caches/inframe/models", ".roboflow/models")]
    ckpt = next((c for c in cands if os.path.isfile(c)), cands[0])
    if not os.path.isfile(ckpt):
        print(f"no checkpoint at {ckpt} (inframe fetch-models rf-detr-{a.size}.pth)", file=sys.stderr)
        return 2
    import coremltools as ct

    kept = []
    real_convert = ct.convert

    def keep_fp32(op):
        if op.op_type in FP32_OPS:
            kept.append(op.name)
            return False
        return True

    def convert(*args, **kw):  # rfdetr calls coremltools.convert(..., compute_precision=FLOAT16); only the precision changes
        kw["compute_precision"] = ct.transform.FP16ComputePrecision(op_selector=keep_fp32)
        return real_convert(*args, **kw)

    ct.convert = convert
    import rfdetr

    os.makedirs(a.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f".{a.size}_gpu_", dir=a.out)
    os.chmod(tmp, 0o755)
    t = time.time()
    getattr(rfdetr, CLASSES[a.size])(pretrain_weights=ckpt).export(output_dir=tmp, format="coreml", batch_size=a.batch, verbose=False,
                                                                  coreml_precision="float16")
    if not kept or not glob.glob(os.path.join(tmp, "**", "*.mlpackage"), recursive=True):
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"export failed (fp32 ops kept: {kept})", file=sys.stderr)
        return 1
    if os.path.exists(dest):
        old = os.path.join(a.out, f"old_{a.size}_coreml_b{a.batch}")
        shutil.rmtree(old, ignore_errors=True)
        os.replace(dest, old)
        print(f"moved the previous package to {old}")
    os.replace(tmp, dest)
    print(f"exported {dest} (fp32 ops: {kept}) in {time.time() - t:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
