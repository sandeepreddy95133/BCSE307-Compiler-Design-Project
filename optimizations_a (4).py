from __future__ import annotations

from ir_core import Value, Constant, Instruction, PhiNode

# ============================================================
# CONSTANT PROPAGATION
# ============================================================

class ConstantPropagation:

    @staticmethod
    def run(fn, verbose=False, cause_log=None):
        changed = False
        local = True

        while local:
            local = False

            for block in fn.blocks.values():
                for instruction in block.instrs:
                    if instruction.dead:
                        continue
                    if instruction.dest is None:
                        continue

                    result = ConstantPropagation._try_fold(fn, instruction)
                    if result is None:
                        continue

                    kind, value = result
                    old = instruction.dest

                    if kind == "const":
                        constant = Constant(value)
                        fn.replace_uses(old, constant)
                        fn.replace_def(old, constant)
                        fn.defs[constant] = constant
                        instruction.dead = True
                    else:
                        fn.replace_uses(old, value)
                        instruction.dead = True

                    fn.stats["cp"] += 1
                    local = True
                    changed = True

                    if verbose:
                        print(f"    [CP] {old} <- {kind}:{value}")
                    if cause_log is not None:
                        cause_log.append(("CP", old, kind, value))

            for block in fn.blocks.values():
                for phi in block.phis:
                    if phi.dead:
                        continue

                    result = ConstantPropagation._try_fold_phi(fn, phi)
                    if result is None:
                        continue

                    kind, value = result
                    old = phi.dest

                    if kind == "const":
                        constant = Constant(value)
                        fn.replace_uses(old, constant)
                        fn.replace_def(old, constant)
                        fn.defs[constant] = constant
                        phi.dead = True
                    else:
                        fn.replace_uses(old, value)
                        phi.dead = True

                    fn.stats["cp"] += 1
                    local = True
                    changed = True

                    if verbose:
                        print(f"    [CP-phi] {old} <- {kind}:{value}")
                    if cause_log is not None:
                        cause_log.append(("CP-phi", old, kind, value))

        return changed

    @staticmethod
    def _resolve(fn, value):
        if isinstance(value, Constant):
            return value
        if value in fn.defs:
            node = fn.defs[value]
            if isinstance(node, Constant):
                return node
        return None

    @staticmethod
    def _try_fold_phi(fn, phi):
        if not phi.incoming:
            return None

        resolved = [
            ConstantPropagation._resolve(fn, value) or value
            for value, _ in phi.incoming
        ]

        constants = [v for v in resolved if isinstance(v, Constant)]

        if len(constants) == len(resolved):
            values = {c.value for c in constants}
            if len(values) == 1:
                return ("const", next(iter(values)))
            return None

        if all(value == resolved[0] for value in resolved):
            return ("alias", resolved[0])

        return None

    @staticmethod
    def _try_fold(fn, instruction):
        op = instruction.op
        resolved = [
            ConstantPropagation._resolve(fn, arg) or arg
            for arg in instruction.args
        ]

        if op == "add" and len(resolved) == 2:
            a, b = resolved
            if isinstance(a, Constant) and a.value == 0:
                return ("alias", b)
            if isinstance(b, Constant) and b.value == 0:
                return ("alias", a)

        if op == "sub" and len(resolved) == 2:
            a, b = resolved
            if isinstance(b, Constant) and b.value == 0:
                return ("alias", a)

        if op == "mul" and len(resolved) == 2:
            a, b = resolved
            if isinstance(a, Constant) and a.value == 0:
                return ("const", 0)
            if isinstance(b, Constant) and b.value == 0:
                return ("const", 0)
            if isinstance(a, Constant) and a.value == 1:
                return ("alias", b)
            if isinstance(b, Constant) and b.value == 1:
                return ("alias", a)

        if op == "div" and len(resolved) == 2:
            a, b = resolved
            if isinstance(b, Constant) and b.value == 1:
                return ("alias", a)

        if op == "copy" and len(resolved) == 1:
            a = resolved[0]
            if isinstance(a, Constant):
                return ("const", a.value)
            return ("alias", a)

        if all(isinstance(v, Constant) for v in resolved):
            values = [v.value for v in resolved]
            folded = ConstantPropagation._eval(op, values)
            if folded is not None:
                return ("const", folded)

        return None

    @staticmethod
    def _eval(op, values):
        try:
            if op == "add": return values[0] + values[1]
            if op == "sub": return values[0] - values[1]
            if op == "mul": return values[0] * values[1]
            if op == "div":
                if values[1] == 0:
                    return None
                return values[0] // values[1]
            if op == "lt": return 1 if values[0] < values[1] else 0
            if op == "gt": return 1 if values[0] > values[1] else 0
            if op == "eq": return 1 if values[0] == values[1] else 0
        except Exception:
            return None
        return None

# ============================================================
# COMMON SUBEXPRESSION ELIMINATION
# ============================================================

class CommonSubexprElim:

    @staticmethod
    def run(fn, verbose=False, cause_log=None):
        changed = False

        for block in fn.blocks.values():
            for instruction in block.instrs:
                if instruction.dead:
                    continue

                # MEMORY INVALIDATION
                if instruction.op in ("store", "call"):
                    if fn.cse_table:
                        if verbose:
                            print(
                                "    [CSE] invalidated "
                                f"table on {instruction.op}"
                            )
                        fn.cse_table.clear()
                    continue

                if instruction.dest is None:
                    continue

                key = CommonSubexprElim._key(fn, instruction)
                if key is None:
                    continue

                if key in fn.cse_table:
                    existing = fn.cse_table[key]
                    if (
                        existing != instruction.dest
                        and existing in fn.defs
                        and not getattr(fn.defs[existing], "dead", False)
                    ):
                        fn.replace_uses(instruction.dest, existing)
                        instruction.dead = True
                        fn.stats["cse"] += 1
                        changed = True
                        if verbose:
                            print(f"    [CSE] {instruction.dest} == {existing}")
                        if cause_log is not None:
                            cause_log.append(("CSE", instruction.dest, "alias", existing))
                        continue

                fn.cse_table[key] = instruction.dest

        for key in list(fn.cse_table.keys()):
            value = fn.cse_table[key]
            if (
                value not in fn.defs
                or getattr(fn.defs[value], "dead", False)
            ):
                del fn.cse_table[key]

        return changed

    @staticmethod
    def _key(fn, instruction):
        op = instruction.op
        if op not in ("add", "sub", "mul", "div", "lt", "gt", "eq"):
            return None

        resolved = [
            ConstantPropagation._resolve(fn, arg) or arg
            for arg in instruction.args
        ]

        if op in ("add", "mul", "eq") and len(resolved) == 2:
            a, b = resolved
            if repr(b) < repr(a):
                a, b = b, a
            resolved = [a, b]

        return (op,) + tuple(resolved)

# ============================================================
# DEAD CODE ELIMINATION
# ============================================================

class DeadCodeElim:

    @staticmethod
    def run(fn, verbose=False, cause_log=None):
        live = set()

        def mark(value):
            if value in live:
                return
            if isinstance(value, Constant):
                return
            live.add(value)
            node = fn.defs.get(value)
            if node is None:
                return
            if isinstance(node, Instruction):
                for arg in node.args:
                    if isinstance(arg, Value):
                        mark(arg)
            elif isinstance(node, PhiNode):
                for arg, _ in node.incoming:
                    if isinstance(arg, Value):
                        mark(arg)

        for block in fn.blocks.values():
            for instruction in block.instrs:
                if instruction.dead:
                    continue
                if instruction.op in ("store", "call", "ret", "br", "jmp", "print"):
                    for arg in instruction.args:
                        if isinstance(arg, Value):
                            mark(arg)

        changed = False

        for block in fn.blocks.values():
            for instruction in block.instrs:
                if instruction.dead:
                    continue
                if instruction.dest is None:
                    continue
                if instruction.dest not in live:
                    instruction.dead = True
                    fn.stats["dce"] += 1
                    changed = True
                    if verbose:
                        print(f"    [DCE] removed {instruction.dest} = {instruction.op}")
                    if cause_log is not None:
                        cause_log.append(("DCE", instruction.dest, "removed", None))

            for phi in block.phis:
                if phi.dead:
                    continue
                if phi.dest not in live:
                    phi.dead = True
                    fn.stats["dce"] += 1
                    changed = True
                    if cause_log is not None:
                        cause_log.append(("DCE-phi", phi.dest, "removed", None))

        return changed
