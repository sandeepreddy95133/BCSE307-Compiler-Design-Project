from ir_core import SSAFunction, Constant
from importlib.util import spec_from_file_location, module_from_spec

# Load your optimization file
spec = spec_from_file_location(
    "optimizations_a",
    "optimizations_a (4).py"
)
opt = module_from_spec(spec)
spec.loader.exec_module(opt)

# Create a test SSA function
fn = SSAFunction("test")
entry = fn.add_block("entry")

# Create values
a = fn.fresh_value("a")
b = fn.fresh_value("b")
c = fn.fresh_value("c")
d = fn.fresh_value("d")

# a = 10 + 20
fn.emit(entry, "add", a, (Constant(10), Constant(20)))

# b = a + 0
fn.emit(entry, "add", b, (a, Constant(0)))

# c = 10 + 20   <-- duplicate expression
fn.emit(entry, "add", c, (Constant(10), Constant(20)))

# d = 999 + 1   <-- never used, should be removed by DCE
fn.emit(entry, "add", d, (Constant(999), Constant(1)))

# print(b)
fn.emit(entry, "print", None, (b,))

print("========== BEFORE ==========")
print(fn)

print("\n========== RUNNING OPTIMIZATIONS ==========")

opt.ConstantPropagation.run(fn, verbose=True)
opt.CommonSubexprElim.run(fn, verbose=True)
opt.DeadCodeElim.run(fn, verbose=True)

print("\n========== AFTER ==========")
print(fn)

print("\n========== STATS ==========")
print(fn.stats)