"""Traversal over the relationship table. A lookup table, not a graph database.

Every edge Vidur stores is already a row, so walking the graph is an indexed
SQLite query with a visited set. That is enough at this scale and keeps the
whole thing inside one file with no dependency.

Traversal is depth-capped. An intelligence graph grows outward indefinitely, and
an uncapped walk over cyclic edges would not terminate.
"""

from __future__ import annotations

from dataclasses import dataclass

from vidur import storage

DEFAULT_MAX_DEPTH = 3


@dataclass(frozen=True)
class Edge:
    rel_type: str
    from_kind: str
    from_id: int
    to_kind: str
    to_id: int
    observation_id: int | None = None

    def other(self, kind: str, identifier: int) -> tuple[str, int] | None:
        if (self.from_kind, self.from_id) == (kind, identifier):
            return self.to_kind, self.to_id
        if (self.to_kind, self.to_id) == (kind, identifier):
            return self.from_kind, self.from_id
        return None


@dataclass(frozen=True)
class Node:
    kind: str
    identifier: int
    depth: int
    label: str = ""
    via: str = ""
    direction: str = "out"   # "out" when reached along from->to, "in" otherwise


def label_for(kind: str, identifier: int) -> str:
    """A human-readable name, or an explicit marker when there is none."""
    try:
        if kind == "ENTITY":
            for row in storage.list_entities():
                if row["id"] == identifier:
                    return row["name"]
        elif kind == "LOCATION":
            for row in storage.list_locations():
                if row["id"] == identifier:
                    return row["name"]
        elif kind == "EVENT":
            with storage.connect() as db:
                row = db.execute("SELECT title FROM events WHERE id=?", (identifier,)).fetchone()
                if row:
                    return f"EVT-{identifier} {row[0][:60]}"
            return f"EVT-{identifier}"
    except Exception:
        return ""
    return f"{kind}-{identifier}"


def edges_of(kind: str, identifier: int, *, rel_types: tuple[str, ...] | None = None) -> list[Edge]:
    """Every edge touching one node, in either direction."""
    conditions, values = [], []
    if rel_types:
        conditions.append(f"rel_type IN ({','.join('?' * len(rel_types))})")
        values.extend(rel_types)
    filter_sql = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    with storage.connect() as db:
        rows = db.execute(f"""
            SELECT * FROM relationships
            WHERE (from_kind=? AND from_id=?){filter_sql}
               OR (to_kind=? AND to_id=?){filter_sql}
        """, (kind, identifier, *values, kind, identifier, *values)).fetchall()
    return [Edge(rel_type=row["rel_type"], from_kind=row["from_kind"], from_id=row["from_id"],
                 to_kind=row["to_kind"], to_id=row["to_id"],
                 observation_id=row["observation_id"]) for row in rows]


def neighbours(kind: str, identifier: int, *, rel_types: tuple[str, ...] | None = None) -> list[Node]:
    """Immediate neighbours, depth 1."""
    found: dict[tuple[str, int], Node] = {}
    for edge in edges_of(kind, identifier, rel_types=rel_types):
        other = edge.other(kind, identifier)
        forward = (edge.from_kind, edge.from_id) == (kind, identifier)
        if other and other not in found:
            found[other] = Node(kind=other[0], identifier=other[1], depth=1,
                                label=label_for(*other), via=edge.rel_type,
                                direction="out" if forward else "in")
    return sorted(found.values(), key=lambda node: (node.kind, node.identifier))


def walk(start_kind: str, start_id: int, *, max_depth: int = DEFAULT_MAX_DEPTH,
         rel_types: tuple[str, ...] | None = None,
         kinds: tuple[str, ...] | None = None) -> list[Node]:
    """Breadth-first walk, inclusive of the start node at depth 0.

    Depth is capped and the visited set stops cycles. ``kinds`` restricts which
    node types are returned, which is how a caller asks for the actors in a
    region without walking out into unrelated territory.
    """
    limit = max(0, int(max_depth))
    start = Node(kind=start_kind, identifier=start_id, depth=0, label=label_for(start_kind, start_id))
    visited = {(start_kind, start_id)}
    frontier = [(start_kind, start_id)]
    reached = [start]
    for depth in range(1, limit + 1):
        following = []
        for kind, identifier in frontier:
            for edge in edges_of(kind, identifier, rel_types=rel_types):
                other = edge.other(kind, identifier)
                if not other or other in visited:
                    continue
                visited.add(other)
                following.append(other)
                forward = (edge.from_kind, edge.from_id) == (kind, identifier)
                if kinds is None or other[0] in kinds:
                    reached.append(Node(kind=other[0], identifier=other[1], depth=depth,
                                        label=label_for(*other), via=edge.rel_type,
                                        direction="out" if forward else "in"))
        frontier = following
        if not frontier:
            break
    return reached


def actors_for_event(event_id: int) -> list[Node]:
    """Entities linked to an event, at any depth but capped by default."""
    return [node for node in walk("EVENT", event_id)
            if node.kind == "ENTITY"]


def places_for_event(event_id: int) -> list[Node]:
    return [node for node in walk("EVENT", event_id)
            if node.kind == "LOCATION"]


def render(nodes: list[Node]) -> str:
    """A compact indented listing for the terminal."""
    if not nodes:
        return "no nodes"
    lines = []
    for node in nodes:
        indent = "  " * node.depth
        if node.depth == 0:
            lines.append(f"{node.label or f'{node.kind}-{node.identifier}'}")
        else:
            connector = f" --{node.via}--> " if node.direction == "out" else f" <--{node.via}-- "
            lines.append(f"{indent}{connector}{node.label or f'{node.kind}-{node.identifier}'}")
    return "\n".join(lines)
