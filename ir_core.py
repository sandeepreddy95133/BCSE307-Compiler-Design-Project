from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

# ============================================================
# IR DATA STRUCTURES
# ============================================================

@dataclass(eq=False)
class Value:
    name: str
    version: int = 0

    def __hash__(self):
        return hash((self.name, self.version))

    def __eq__(self, other):
        return (
            isinstance(other, Value)
            and self.name == other.name
            and self.version == other.version
        )

    def __repr__(self):
        return f"%{self.name}.{self.version}"

@dataclass(eq=False)
class Constant(Value):
    value: object = None

    def __init__(self, value):
        super().__init__(name=f"const_{value}", version=0)
        self.value = value

    def __repr__(self):
        return repr(self.value)

@dataclass(eq=False)
class Instruction:
    op: str
    dest: Optional[Value]
    args: tuple = ()
    id: int = -1
    dead: bool = False

    def __repr__(self):
        dead_mark = " [DEAD]" if self.dead else ""
        if self.dest is None:
            return f"  {self.op} {self.args}{dead_mark}"
        return f"  {self.dest} = {self.op} {self.args}{dead_mark}"

@dataclass(eq=False)
class PhiNode:
    dest: Value
    incoming: list = field(default_factory=list)
    dead: bool = False

    def __repr__(self):
        pairs = ", ".join(
            f"({value}, {block})"
            for value, block in self.incoming
        )
        dead_mark = " [DEAD]" if self.dead else ""
        return f"  {self.dest} = phi({pairs}){dead_mark}"

@dataclass
class BasicBlock:
    name: str
    instrs: list = field(default_factory=list)
    phis: list = field(default_factory=list)
    succs: list = field(default_factory=list)
    preds: list = field(default_factory=list)

    def __repr__(self):
        lines = [
            f"BB {self.name} "
            f"(preds={self.preds}, succs={self.succs}):"
        ]
        for phi in self.phis:
            lines.append(f"    {phi}")
        for instruction in self.instrs:
            lines.append(f"    {instruction}")
        return "\n".join(lines)

# ============================================================
# SSA FUNCTION
# ============================================================

class SSAFunction:

    def __init__(self, name):
        self.name = name
        self.blocks = {}
        self.entry = None
        self.defs = {}
        self.uses = defaultdict(set)
        self.next_version = defaultdict(int)
        self.next_instr_id = 0
        self.stats = {"cp": 0, "cse": 0, "dce": 0, "phi": 0, "cfg": 0}
        self.metrics = {
            "ir_writes": 0,
            "def_moves": 0,
            "use_rewires": 0,
            "iterations": 0
        }
        self.cse_table = {}

    def add_block(self, name):
        bb = BasicBlock(name)
        self.blocks[name] = bb
        if self.entry is None:
            self.entry = bb
        return bb

    def add_edge(self, source, destination):
        if destination not in self.blocks[source].succs:
            self.blocks[source].succs.append(destination)
        if source not in self.blocks[destination].preds:
            self.blocks[destination].preds.append(source)

    def remove_edge(self, source, destination):
        if source not in self.blocks:
            return
        if destination not in self.blocks:
            return
        if destination in self.blocks[source].succs:
            self.blocks[source].succs.remove(destination)
        if source in self.blocks[destination].preds:
            self.blocks[destination].preds.remove(source)

    def fresh_value(self, base):
        version = self.next_version[base]
        self.next_version[base] += 1
        return Value(base, version)

    def emit(self, block, op, dest, args):
        if dest is not None and dest in self.defs:
            raise ValueError(f"SSA violation: {dest}")

        instruction = Instruction(
            op=op, dest=dest, args=args, id=self.next_instr_id
        )
        self.next_instr_id += 1
        block.instrs.append(instruction)

        if dest is not None:
            self.defs[dest] = instruction

        for arg in args:
            if isinstance(arg, Value) and not isinstance(arg, Constant):
                self.uses[arg].add(instruction)

        return instruction

    def add_phi(self, block, dest, incoming):
        if dest in self.defs:
            raise ValueError(f"SSA violation: {dest}")

        phi = PhiNode(dest=dest, incoming=list(incoming))
        block.phis.append(phi)
        self.defs[dest] = phi

        for value, _ in phi.incoming:
            if isinstance(value, Value) and not isinstance(value, Constant):
                self.uses[value].add(phi)

        return phi

    def replace_uses(self, old, new):
        if old not in self.uses:
            return

        for user in list(self.uses[old]):
            if isinstance(user, Instruction):
                user.args = tuple(
                    new if arg == old else arg
                    for arg in user.args
                )
                self.metrics["ir_writes"] += 1
            elif isinstance(user, PhiNode):
                user.incoming = [
                    (new if value == old else value, block)
                    for value, block in user.incoming
                ]
                self.metrics["ir_writes"] += 1

            if isinstance(new, Value) and not isinstance(new, Constant):
                self.uses[new].add(user)

            self.metrics["use_rewires"] += 1

        self.uses[old] = set()

    def replace_def(self, old, new):
        if old not in self.defs:
            return
        node = self.defs[old]
        if isinstance(node, Instruction):
            node.dest = new
        elif isinstance(node, PhiNode):
            node.dest = new
        del self.defs[old]
        self.defs[new] = node
        self.metrics["def_moves"] += 1

    def rebuild_uses(self):
        self.uses.clear()
        for bb in self.blocks.values():
            for instruction in bb.instrs:
                for arg in instruction.args:
                    if isinstance(arg, Value) and not isinstance(arg, Constant):
                        self.uses[arg].add(instruction)
            for phi in bb.phis:
                for value, _ in phi.incoming:
                    if isinstance(value, Value) and not isinstance(value, Constant):
                        self.uses[value].add(phi)

    def __repr__(self):
        return "\n".join(
            repr(block) for block in self.blocks.values()
        )