from __future__ import annotations

from ir_core import Value, Constant, Instruction, PhiNode
from optimizations_a import (
    ConstantPropagation,
    CommonSubexprElim,
    DeadCodeElim,
)

# ============================================================
# PHI COLLAPSE
# ============================================================

class PhiCollapse:

    @staticmethod
    def run(fn, verbose=False, cause_log=None):
        changed = False

        for block in fn.blocks.values():
            for phi in block.phis:
                if phi.dead:
                    continue
                if len(phi.incoming) < 1:
                    continue

                values = [v for v, _ in phi.incoming]
                first = values[0]

                if (
                    len(values) == 1
                    or all(v == first for v in values[1:])
                ):
                    fn.replace_uses(phi.dest, first)
                    phi.dead = True
                    fn.stats["phi"] += 1
                    changed = True
                    if verbose:
                        print(f"    [PHI] collapsed {phi.dest} -> {first}")
                    if cause_log is not None:
                        cause_log.append(("PHI", phi.dest, "collapse", first))

        return changed

# ============================================================
# CFG PRUNING
# ============================================================

class CFGPrune:

    @staticmethod
    def run(fn, verbose=False, cause_log=None):
        changed = False

        # CONSTANT BRANCH PRUNING
        for block in list(fn.blocks.values()):
            for instruction in block.instrs:
                if instruction.dead:
                    continue
                if instruction.op != "br":
                    continue
                if len(instruction.args) < 3:
                    continue

                condition = ConstantPropagation._resolve(
                    fn, instruction.args[0]
                )
                if condition is None:
                    continue

                true_target = instruction.args[1]
                false_target = instruction.args[2]

                if condition.value != 0:
                    live_target = true_target
                    dead_target = false_target
                else:
                    live_target = false_target
                    dead_target = true_target

                if dead_target not in block.succs:
                    continue

                fn.remove_edge(block.name, dead_target)
                instruction.op = "jmp"
                instruction.args = (live_target,)
                fn.stats["cfg"] += 1
                changed = True

                if verbose:
                    print(
                        f"    [CFG] constant branch in {block.name}: "
                        f"removed {dead_target}, jump -> {live_target}"
                    )
                if cause_log is not None:
                    cause_log.append(("CFG", block.name, "prune-edge", dead_target))

        # REMOVE UNREACHABLE BLOCKS
        removed = CFGPrune._remove_unreachable(fn, verbose, cause_log)
        if removed:
            changed = True

        fn.rebuild_uses()
        return changed

    @staticmethod
    def _remove_unreachable(fn, verbose=False, cause_log=None):
        if fn.entry is None:
            return False

        reachable = set()
        work = [fn.entry.name]

        while work:
            block_name = work.pop()
            if block_name in reachable:
                continue
            if block_name not in fn.blocks:
                continue
            reachable.add(block_name)
            for successor in fn.blocks[block_name].succs:
                if successor not in reachable:
                    work.append(successor)

        unreachable = [
            name for name in list(fn.blocks.keys())
            if name not in reachable
        ]

        if not unreachable:
            return False

        for block_name in unreachable:
            if block_name not in fn.blocks:
                continue

            block = fn.blocks[block_name]

            for pred in list(block.preds):
                fn.remove_edge(pred, block_name)

            for successor in list(block.succs):
                if successor not in fn.blocks:
                    continue
                fn.remove_edge(block_name, successor)
                successor_block = fn.blocks[successor]
                for phi in successor_block.phis:
                    old_length = len(phi.incoming)
                    phi.incoming = [
                        (value, pred)
                        for value, pred in phi.incoming
                        if pred != block_name
                    ]
                    if len(phi.incoming) != old_length:
                        if cause_log is not None:
                            cause_log.append(
                                ("CFG-phi", phi.dest, "prune", block_name)
                            )

            if verbose:
                print(f"    [CFG] removed unreachable block {block_name}")
            if cause_log is not None:
                cause_log.append(("CFG", block_name, "remove-block", None))

            del fn.blocks[block_name]

        return True

# ============================================================
# UNIFIED OPTIMIZATION LOOP
# ============================================================

class OptimizationLoop:

    def __init__(self, fn, max_iters=20, verbose=True):
        self.fn = fn
        self.max_iters = max_iters
        self.verbose = verbose
        self.iterations_run = 0
        self.cause_log = []
        self.trigger_log = []

    def run(self):
        iteration = 0

        while iteration < self.max_iters:
            iteration += 1

            if self.verbose:
                print(f"\n=== Iteration {iteration} ===")

            cp_log, cse_log, dce_log, phi_log, cfg_log = [], [], [], [], []

            cp = ConstantPropagation.run(self.fn, self.verbose, cp_log)
            self.trigger_log.append((iteration, "CP", cp))

            cse = CommonSubexprElim.run(self.fn, self.verbose, cse_log)
            self.trigger_log.append((iteration, "CSE", cse))

            dce = DeadCodeElim.run(self.fn, self.verbose, dce_log)
            self.trigger_log.append((iteration, "DCE", dce))

            phi = PhiCollapse.run(self.fn, self.verbose, phi_log)
            self.trigger_log.append((iteration, "PHI", phi))

            cfg = CFGPrune.run(self.fn, self.verbose, cfg_log)
            self.trigger_log.append((iteration, "CFG", cfg))

            for event in (cp_log + cse_log + dce_log + phi_log + cfg_log):
                self.cause_log.append((iteration,) + event)

            if self.verbose:
                print(
                    f"  CP:{cp}  CSE:{cse}  DCE:{dce}  "
                    f"PHI:{phi}  CFG:{cfg}"
                )

            if not (cp or cse or dce or phi or cfg):
                break

        self.iterations_run = iteration
        self.fn.metrics["iterations"] = iteration

        if self.verbose:
            print(f"\nConverged after {iteration} iteration(s).")

        return iteration
