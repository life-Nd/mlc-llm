import tvm
from tvm import IRModule
from tvm import relax, te, tir
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import visitor, mutator, PyExprMutator, PyExprVisitor
from tvm.relax.analysis import remove_all_unused


@visitor 
class ForwardCollector(PyExprVisitor):
    def __init__(self, tuple_var: relax.Var, input_params: relax.Var) -> None:
        self.out_tuple_map = {}
        self.out_tuple_var = tuple_var
        self.input_params = input_params
        self.input_params_after_get_item = []
        self.is_tuple_get_item_input = False

    def visit_tuple_getitem_(self, op: relax.TupleGetItem) -> None:
        if op.tuple_value == self.input_params:
            self.is_tuple_get_item_input = True
        else:
            self.is_tuple_get_item_input = False
        super().visit_tuple_getitem_(op)

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if binding.var == self.out_tuple_var:
            assert isinstance(binding.value, relax.Tuple)
            for i, expr in enumerate(binding.value.fields):
                self.out_tuple_map[expr] = relax.PrimValue(i)
        else:
            self.is_tuple_get_item_input = False
            super().visit_var_binding_(binding)
            if self.is_tuple_get_item_input:
                self.input_params_after_get_item.append(binding.var)
    
    
    
    @staticmethod 
    def collect_out_tuple_map(tuple_var: relax.Var, func: relax.Function) -> dict:
        collector = ForwardCollector(tuple_var)
        collector.visit_expr(func)
        return collector.out_tuple_map

    
    
    
@visitor
class LivenessAnalysis(PyExprVisitor):
    def __init__(self, out_tuple_var:relax.Var, input_params: set) -> None:
        self.last_appear_in_var_binding = None
        self.out_tuple_var = out_tuple_var
        self.input_params = input_params
        self.var_liveness_end = {}
        
        
    def visit_dataflow_block_(self, block: relax.DataflowBlock) -> None:
        for binding in reversed(block.bindings):
            self.visit_binding(binding)
    
    def visit_dataflow_var_(self, op: relax.DataflowVar) -> None:
        if op in self.input_params:
            self.last_appear_in_var_binding.append(op)     
            self.input_params.remove(op)  
    
    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if self.out_tuple_var == binding.var:
            return
        self.last_appear_in_var_binding = []
        super().visit_var_binding_(binding)
        #param[i] is in output
        if binding.var in self.input_params:
            self.last_appear_in_var_binding.append(binding.var)
        self.var_liveness_end[binding.var] = self.last_appear_in_var_binding
        
    
@mutator
class LazyLoadingMutator(PyExprMutator):
    def __init__(self, mod: IRModule = None) -> None:
        super().__init__(mod)
        self.mod=mod
        self.get_item = None
        self.set_item = None
        self.input_tuple_param = None
        self.out_tuple_map = None
        self.out_tuple_var = None
        self.memory_free_insertion = None
        
    def transform(self, func:relax.Function) -> relax.Function:
        self.input_tuple_param = func.params[0]
        seq_expr = func.body
        self.out_tuple_var = seq_expr.body
        
        forward_collector = ForwardCollector(self.out_tuple_var, self.input_tuple_param)
        forward_collector.visit_expr(func)
        self.out_tuple_map = forward_collector.out_tuple_map
        input_params_set = set(forward_collector.input_params_after_get_item)
        liveness = LivenessAnalysis(self.out_tuple_var, input_params_set)
        liveness.visit_expr(func)
        self.memory_free_insertion=liveness.var_liveness_end
        new_body = self.visit_expr(func.body)
        return relax.Function([], new_body, relax.ObjectStructInfo(), func.attrs)
                
    def visit_tuple_getitem_(self, tuple_get_item: relax.TupleGetItem) -> relax.Expr:
        tuple_get_item = super().visit_tuple_getitem_(tuple_get_item)
        if tuple_get_item.tuple_value == self.input_tuple_param:
            return relax.Call(relax.ExternFunc("get_item"), [relax.PrimValue(tuple_get_item.index)], None, [relax.ObjectStructInfo()])
        else:
            return tuple_get_item
        
    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if binding.var in self.out_tuple_map:
            index = self.out_tuple_map[binding.var]
            value = self.visit_expr(binding.value)
            var_before_setitem = self.builder_.emit(value)
            new_var = self.builder_.emit(relax.Call(relax.ExternFunc("set_item"), [index, var_before_setitem], None, [relax.ObjectStructInfo()]))
            self.set_var_remap(binding.var.vid, new_var)
        else:
            super().visit_var_binding_(binding)
        if binding.var in self.memory_free_insertion:
            for var in self.memory_free_insertion[binding.var]:
                #handle param[i] in output 
                if var == binding.var:
                    assert binding.var in self.out_tuple_map
                    self.builder_.emit(relax.Call(relax.ExternFunc("memory_free"), [var_before_setitem], None, [relax.ObjectStructInfo()]))
                else:
                    self.builder_.emit(relax.Call(relax.ExternFunc("memory_free"), [self.get_var_remap(var.vid)], None, [relax.ObjectStructInfo()]))
    

@tvm.transform.module_pass(opt_level=0, name="LazyLoading")
class LazyLoading:
    def transform_module(
        self, mod: IRModule, ctx: tvm.transform.PassContext
    ) -> IRModule:
        
        mutator = LazyLoadingMutator(mod)
        for gv in mod.functions:
            if gv.name_hint.endswith("transform_params"):
                func = mod[gv]
                if not isinstance(func, relax.Function):
                    continue
                func = mutator.transform(func)
                mutator.builder_.update_func(gv, func)

        return mutator.builder_.get()
