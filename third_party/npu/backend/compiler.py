import functools
import hashlib
import os
import re

from dataclasses import dataclass
from typing import Any, Dict
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
        passes.ttir.add_rewrite_tensor_pointer(pm)
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
        passes.ttir.add_convert_to_ttnpuir(pm)

        #
        # TODO:
        #

        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)

        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # TritonNPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)

        npu.passes.ttnpuir.add_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)

         # TODO:
        if not llvm_mod:
            metadata["shared"] = 0
            return src

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # NPU doesn't have SMEM, but just to make it work for now.
        metadata["shared"] = 0

        # Cleanup
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_npubin(src, metadata, options):
        # Right now, src is just TTIR. Extract kernel name from tt.func.
        names = re.findall(r"\s+tt.func public @([a-zA-Z_][a-zA-Z0-9_]*)\(", str(src))
        assert len(names) == 1
        metadata["name"] = names[0]
        # TODO: Call llc to create an executable.
        return src

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttcir"] = lambda src, metadata: self.make_ttnir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["npubin"] = lambda src, metadata: self.make_npubin(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        # TODO: Just toy name here, you can replace with real architecture and version.
        return f"v0.0.1 -- NPU"