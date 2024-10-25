import functools
import hashlib
import os
import re

from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType

from triton._C.libtriton import npu, ir, llvm, passes
from triton.backends.compiler import NPUBaseBackend, NPUTarget

@dataclass(frozen=True)
class NPUOptions:
    # GPU-specific options are used in several places.
    # For now, we just provide dummy values.
    num_warps: int = 0
    num_stages: int = 0
    num_ctas: int = 0
    cluster_dims: tuple = (1, 1, 1)
    extern_libs: dict = None
    debug: bool = False
    sanitize_overflow: bool = True
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    allow_fp8e4nv: bool = False

    # TODO: We may introduce NPU-specific options.
    def __post_init__(self):
        pass

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
    
class NPUBackend(NPUBaseBackend):

    @staticmethod
    def supports_target(target: NPUTarget):
        return target.backend == "npu"

    def __init__(self, target: NPUTarget) -> None:
        super().__init__(target)
        # TODO: Consider npubin extension for now
        self.binary_ext = "npubin"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in NPUOptions.__dataclass_fields__.keys() if k in opts}
        return NPUOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self):
        codegen_fns = dict()
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        # from triton.language.extra.npu import libdevice
        # return {"triton.language.extra.libdevice": libdevice}
        # TODO: return empty Dict currently
        return {}

    def load_dialects(self, ctx):
        npu.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        # This is the same as the Nvidia backend.
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttnir(mod, metadata, opt):
        # TTIR -> TTNIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        
        npu.passes.ttnpuir.add_triton_to_triton_npu_pipeline(pm)

        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)

        metadata["cluster_dims"] = (opt.cluster_dims[0], opt.cluster_dims[1], opt.cluster_dims[2])

        pm.run(mod)

        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        # warp-specialization mutates num_warps
        num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups
        metadata["threads_per_warp"] = 1

        mod = src
        # TritonNPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)

        npu.passes.ttnpuir.add_triton_npu_to_llvmir_pipeline(pm)
        passes.convert.add_math_to_llvmir(pm)
        npu.passes.ttnpuir.add_math_to_libm(pm)
        npu.passes.ttnpuir.add_vector_to_llvmir(pm)
        npu.passes.ttnpuir.add_memref_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        npu.passes.ttnpuir.add_func_to_llvmir(pm)

        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)

        # Find kernel fn
        kernel_names = npu.find_kernel_names(mod)
        assert len(kernel_names) == 1, f"expected exactly 1 kernel in a module, got {kernel_names}"

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)

        llvm.set_host_target(llvm_mod)
        #if options.extern_libs:
        #    paths = [path for (name, path) in options.extern_libs]
        #   llvm.link_extern_libs(llvm_mod, paths)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # Get some metadata
        metadata["shared"] = 0
        metadata["name"] = kernel_names[0]

        ret = str(llvm_mod)

        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_npubin(src, metadata, options):
        if os.environ.get("TRITON_NPU_ASM_DUMP", "0") == "1":
            print("********** Module ASM **********")
            print(llvm.translate_to_host_asm(src))

            from triton.runtime.cache import get_cache_manager
            asm = llvm.translate_to_host_asm(src, options.enable_fp_fusion)
            fn_cache_manager = get_cache_manager(metadata['hash'])
            fn_cache_manager.put(asm, f"{metadata['name']}.asm")
        ret = llvm.translate_to_npubin(src)
        return ret

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttnir"] = lambda src, metadata: self.make_ttnir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["npubin"] = lambda src, metadata: self.make_npubin(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        # TODO: Just toy name here, you can replace with real architecture and version.
        return f"v0.0.1 -- NPU"