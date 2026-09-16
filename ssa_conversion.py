from __future__ import annotations

from collections import defaultdict

from ir_core import Value, Constant, Instruction, PhiNode

# ============================================================
# DOMINATOR ANALYSIS
# ============================================================

class DominatorAnalysis:

    def __init__(self, fn):
        self.fn = fn
        self.dom = {}
        self.idom = {}
        self.children = defaultdict(list)

    def compute(self):
        blocks = list(self.fn.blocks.keys())
        if not blocks:
            return

        entry = self.fn.entry.name

        for block in blocks:
            if block == entry:
                self.dom[block] = {entry}
            else:
                self.dom[block] = set(blocks)

        changed = True
        while changed:
            changed = False
            for block in blocks:
                if block == entry:
                    continue
                preds = self.fn.blocks[block].preds
                if not preds:
                    continue
                new_dom = set.intersection(
                    *(self.dom[pred] for pred in preds)
                )
                new_dom.add(block)
                if new_dom != self.dom[block]:
                    self.dom[block] = new_dom
                    changed = True

        for block in blocks:
            if block == entry:
                self.idom[block] = None
                continue
            strict = self.dom[block] - {block}
            immediate = None
            for candidate in strict:
                if all(
                    candidate == other
                    or candidate not in self.dom[other]
                    for other in strict
                    if other != candidate
                ):
                    immediate = candidate
                    break
            self.idom[block] = immediate
            if immediate is not None:
                self.children[immediate].append(block)

    def dominance_frontier(self):
        frontier = defaultdict(set)
        for block in self.fn.blocks:
            preds = self.fn.blocks[block].preds
            if len(preds) < 2:
                continue
            for pred in preds:
                runner = pred
                while (
                    runner is not None
                    and runner != self.idom.get(block)
                ):
                    frontier[runner].add(block)
                    runner = self.idom.get(runner)
        return frontier

# ============================================================
# IR TO SSA CONVERSION
# ============================================================

class ToSSA:

    @staticmethod
    def run(fn, verbose=False):

        dominators = DominatorAnalysis(fn)
        dominators.compute()
        frontier = dominators.dominance_frontier()

        defs_by_base = defaultdict(set)
        for block_name, block in fn.blocks.items():
            for instruction in block.instrs:
                if instruction.dest is not None:
                    defs_by_base[instruction.dest.name].add(block_name)

        phi_placed = set()

        # PHI PLACEMENT
        for base, def_blocks in defs_by_base.items():
            work = list(def_blocks)
            seen = set()
            while work:
                definition_block = work.pop()
                if definition_block in seen:
                    continue
                seen.add(definition_block)
                for frontier_block in frontier.get(definition_block, set()):
                    key = (frontier_block, base)
                    if key in phi_placed:
                        continue
                    phi_dest = fn.fresh_value(base)
                    fn.add_phi(fn.blocks[frontier_block], phi_dest, [])
                    phi_placed.add(key)
                    if verbose:
                        print(f"    [ToSSA] placed phi {phi_dest} at {frontier_block}")
                    work.append(frontier_block)

        # SSA RENAMING
        rename_stack = defaultdict(list)
        exit_version = {}

        def new_version(base):
            value = fn.fresh_value(base)
            rename_stack[base].append(value)
            return value

        def current_version(base):
            if rename_stack[base]:
                return rename_stack[base][-1]
            return Value(base, 0)

        def rename_block(block_name):
            block = fn.blocks[block_name]
            pushed = []

            for phi in block.phis:
                base = phi.dest.name
                new_dest = new_version(base)
                pushed.append(base)
                old_dest = phi.dest
                if new_dest != old_dest:
                    if old_dest in fn.defs:
                        del fn.defs[old_dest]
                    fn.defs[new_dest] = phi
                    phi.dest = new_dest

            for instruction in block.instrs:
                new_args = []
                for arg in instruction.args:
                    if isinstance(arg, Constant):
                        new_args.append(arg)
                    elif isinstance(arg, Value):
                        new_args.append(current_version(arg.name))
                    else:
                        new_args.append(arg)
                instruction.args = tuple(new_args)

                if instruction.dest is not None:
                    base = instruction.dest.name
                    new_dest = new_version(base)
                    pushed.append(base)
                    old_dest = instruction.dest
                    if new_dest != old_dest:
                        if old_dest in fn.defs:
                            del fn.defs[old_dest]
                        fn.defs[new_dest] = instruction
                        instruction.dest = new_dest

            for base in rename_stack:
                if rename_stack[base]:
                    exit_version[(block_name, base)] = rename_stack[base][-1]

            for child in dominators.children.get(block_name, []):
                rename_block(child)

            for base in reversed(pushed):
                rename_stack[base].pop()

        if fn.entry is not None:
            rename_block(fn.entry.name)

        # FILL PHI INCOMING VALUES
        for block in fn.blocks.values():
            for phi in block.phis:
                base = phi.dest.name
                phi.incoming = []
                for pred in block.preds:
                    value = exit_version.get((pred, base))
                    if value is not None:
                        phi.incoming.append((value, pred))

        fn.rebuild_uses()

        for block in fn.blocks.values():
            for phi in block.phis:
                phi.incoming.sort(key=lambda item: item[1])

        return True
