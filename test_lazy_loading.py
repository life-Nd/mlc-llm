import tvm
import tvm.testing
from tvm import relax
from tvm.script import relax as R, tir as T
from tvm.script import ir as I
from mlc_llm.transform.lazy_loading import LazyLoading
@I.ir_module
class Module:
    @T.prim_func
    def transform_layout_IOHW_to_OIHW(
        w1: T.Buffer((3, 16, 3, 3), "float32"), out: T.Buffer((16, 3, 3, 3), "float32")
    ):
        for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
            with T.block("layout_transform"):
                o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(w1[i, o, h, w])
                T.writes(out[o, i, h, w])
                out[o, i, h, w] = w1[i, o, h, w]
                
    @R.function
    def main_transform_params(
        params: R.Tuple(
            R.Tensor((3, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
        )
    ) -> R.Tuple(
        R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
    ):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"),
            )
            gv: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((16, 3, 3, 3), dtype="float32"),
            ) = (lv, lv2)
            R.output(gv)
        return gv
    
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class After:
    @T.prim_func
    def transform_layout_IOHW_to_OIHW(w1: T.Buffer((3, 16, 3, 3), "float32"), out: T.Buffer((16, 3, 3, 3), "float32")):
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
            with T.block("layout_transform"):
                o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(w1[i, o, h, w])
                T.writes(out[o, i, h, w])
                out[o, i, h, w] = w1[i, o, h, w]

    @R.function
    def main_transform_params() -> R.Tuple(R.Object, R.Object):
        cls = After
        with R.dataflow():
            lv: R.Object = R.call_packed("get_item", R.prim_value(1), sinfo_args=(R.Object,))
            lv1: R.Object = R.call_packed("set_item", R.prim_value(0), lv, sinfo_args=(R.Object,))
            lv2: R.Object = R.call_packed("memory_free", lv, sinfo_args=(R.Object,))
            lv2_1: R.Object = R.call_packed("memory_free", lv, sinfo_args=(R.Object,))
            lv1_1: R.Object = R.call_packed("get_item", R.prim_value(0), sinfo_args=(R.Object,))
            lv3 = R.call_tir(cls.transform_layout_IOHW_to_OIHW, (lv1_1,), out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"))
            lv4: R.Object = R.call_packed("set_item", R.prim_value(1), lv3, sinfo_args=(R.Object,))
            lv5: R.Object = R.call_packed("memory_free", lv1_1, sinfo_args=(R.Object,))
            gv: R.Tuple(R.Object, R.Object) = lv1, lv4
            R.output(gv)
        return gv    
    
mod = LazyLoading()(Module)
print(mod.script())

import numpy as np
inputs = [tvm.nd.array(np.random.randn(3, 16, 3, 3).astype("float32")), tvm.nd.array(np.random.randn(16, 16, 3, 3).astype("float32"))]
@tvm.register_func("get_item")
def get_item(i):
    gpu_input = tvm.nd.array(inputs[i], device=tvm.gpu(0))
    return gpu_input

@tvm.register_func("memory_free")
def free_input(input):
    print(type(input))
    del input


res = []
@tvm.register_func("set_item")
def set_item(i, value):
    if len(res)<=i:
        res.extend([None]*(i-len(res)+1))
    res[i]=tvm.nd.array(value, device=tvm.cpu())
    print("set item", i, value.shape)
    
    
with tvm.target.Target("cuda"):
    mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
    ex = relax.build(mod, "cuda")
vm = relax.VirtualMachine(ex, tvm.cuda())
vm["main_transform_params"]()

print(res[0].numpy())
print(res[1].numpy())