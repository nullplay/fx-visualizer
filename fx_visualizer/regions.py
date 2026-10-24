"""Draw scheduler regions on the captured original FX graph."""

from __future__ import annotations

import html
import json
import math
import shutil
import subprocess
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


_COLORS = [
    "var(--widget-series-1, #2563eb)",
    "var(--widget-series-2, #d97706)",
    "var(--widget-series-3, #16a34a)",
    "var(--widget-series-4, #9333ea)",
    "var(--widget-series-5, #0891b2)",
    "var(--widget-series-6, #db2777)",
]
_BACKGROUND = "var(--widget-background, #ffffff)"
_FOREGROUND = "var(--widget-foreground, #111827)"
_MUTED = "var(--widget-muted, #6b7280)"
_BORDER = "var(--widget-border, #d1d5db)"
_SURFACE = "var(--widget-surface, #f3f4f6)"
_FUSION = "var(--widget-accent, #6366f1)"


def _number(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _corners(box):
    x, y, w, h = (box[k] for k in ("x", "y", "width", "height"))
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _bbox(points):
    xs, ys = zip(*points)
    return {"x": min(xs), "y": min(ys), "width": max(xs) - min(xs), "height": max(ys) - min(ys)}


def _overlap(a, b):
    return a["x"] <= b["x"] + b["width"] and b["x"] <= a["x"] + a["width"] and a["y"] <= b["y"] + b["height"] and b["y"] <= a["y"] + a["height"]


def _inside(point, rings):
    x, y = point
    hit = False
    for polygon in rings:
        previous = polygon[-1]
        for current in polygon:
            xi, yi = current
            xj, yj = previous
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                hit = not hit
            previous = current
    return hit


def _intersects_rectangle(box, rings, contains=None):
    if any((contains(p) if contains else _inside(p, rings)) for p in _corners(box)):
        return True
    return _boundary_hits_rectangle(box, rings)


def _boundary_hits_rectangle(box, rings):
    x0, y0 = box["x"], box["y"]
    x1, y1 = x0 + box["width"], y0 + box["height"]
    for polygon in rings:
        if not _overlap(box, _bbox(polygon)):
            continue
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            if max(a[0], b[0]) < x0 or min(a[0], b[0]) > x1 or max(a[1], b[1]) < y0 or min(a[1], b[1]) > y1:
                continue
            low, high = 0.0, 1.0
            dx, dy = b[0] - a[0], b[1] - a[1]
            for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
                if p == 0:
                    if q < 0:
                        low, high = 1.0, 0.0
                        break
                elif p < 0:
                    low = max(low, q / p)
                else:
                    high = min(high, q / p)
                if low > high:
                    break
            if low <= high:
                return True
    return False


class _Containment:
    def __init__(self, rings, bucket_size=10):
        self.bucket_size = bucket_size
        self.buckets = defaultdict(list)
        self.crossings = {}
        for polygon in rings:
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                if a[1] == b[1]:
                    continue
                first = math.floor(min(a[1], b[1]) / bucket_size)
                last = math.floor(max(a[1], b[1]) / bucket_size)
                for row in range(first, last + 1):
                    self.buckets[row].append((a, b))

    def contains(self, point):
        x, y = point
        if y not in self.crossings:
            segments = self.buckets[math.floor(y / self.bucket_size)]
            self.crossings[y] = sorted(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]) for a, b in segments if (a[1] > y) != (b[1] > y))
        crossings = self.crossings[y]
        return (len(crossings) - bisect_right(crossings, x)) % 2 == 1


def _members(node, order):
    declared = set(node.get("execution_fx_ids", [])) | set(node.get("indexing_fx_ids", []))
    known = sorted(declared & order.keys(), key=order.__getitem__)
    return known, sorted(declared - order.keys())


def _regions(capture):
    original = capture["original_graph"]
    order = {node["id"]: i for i, node in enumerate(original["nodes"])}
    lowering = []
    for i, node in enumerate(capture["stages"]["lowering"]["scheduler_nodes"]):
        members, unknown = _members(node, order)
        evidence = node.get("execution_evidence", {})
        lowering.append({
            "id": node["id"], "svg_id": f"lowering-region-{i}", "label": node.get("name", node["id"]),
            "member_fx_ids": members, "execution_fx_ids": list(node.get("execution_fx_ids", [])),
            "indexing_fx_ids": list(node.get("indexing_fx_ids", [])),
            "member_operation_names": list(node.get("member_operation_names", [node["id"]])),
            "unknown_fx_ids": sorted(set(unknown + evidence.get("unknown_fx_ids", []))),
            "source_evidence": evidence, "certainty": evidence.get("certainty", "not annotated"),
            "backend": node.get("backend"), "class": node.get("class"),
            "scheduler_type": node.get("scheduler_type"), "codegen_backend": node.get("codegen_backend"),
            "is_template": node.get("is_template", False), "color": _COLORS[i % len(_COLORS)],
        })
    owners = defaultdict(list)
    for group in lowering:
        for operation in group["member_operation_names"]:
            owners[operation].append(group)
    accepted = capture.get("accepted_fusions", [])
    fusion = []
    for node in capture["stages"]["fusion"]["scheduler_nodes"]:
        operations = list(node.get("member_operation_names", [node["id"]]))
        children = []
        partial_children = []
        for operation in operations:
            for group in owners.get(operation, []):
                destination = children if set(group["member_operation_names"]) <= set(operations) else partial_children
                if group not in destination:
                    destination.append(group)
        accepted_result = [event for event in accepted if event.get("result") == node["id"]]
        if len(children) < 2 and not accepted_result:
            continue
        members, unknown = _members(node, order)
        declared_members = members
        if children:
            members = sorted(set(members) | {key for group in children for key in group["member_fx_ids"]}, key=order.__getitem__)
        evidence = node.get("execution_evidence", {})
        fusion.append({
            "id": node["id"], "svg_id": f"fusion-region-{len(fusion)}", "label": node.get("name", node["id"]),
            "member_fx_ids": members, "declared_member_fx_ids": declared_members,
            "execution_fx_ids": list(node.get("execution_fx_ids", [])), "indexing_fx_ids": list(node.get("indexing_fx_ids", [])),
            "lowering_members": [group["id"] for group in children],
            "partial_lowering_members": [group["id"] for group in partial_children],
            "member_operation_names": operations,
            "unknown_operation_names": [name for name in operations if name not in owners],
            "accepted_fusion_events": accepted_result,
            "unknown_fx_ids": sorted(set(unknown + evidence.get("unknown_fx_ids", []))),
            "source_evidence": evidence, "certainty": evidence.get("certainty", "not annotated"),
            "backend": node.get("backend"), "class": node.get("class"),
            "scheduler_type": node.get("scheduler_type"), "codegen_backend": node.get("codegen_backend"),
            "is_template": node.get("is_template", False), "color": _FUSION,
        })
    return lowering, fusion


def _labels(node):
    kind = node.get("kind", "compute")
    geometry = node.get("geometry") or {}
    shape = node.get("shape") or geometry.get("shape")
    dtype = str(node.get("dtype") or geometry.get("dtype") or "").removeprefix("torch.")
    dtype = {"float32": "f32", "bfloat16": "bf16", "float16": "f16"}.get(dtype, dtype)
    size = "\u00d7".join(str(v) for v in shape) if isinstance(shape, (list, tuple)) else ""
    if kind in ("input", "shape") or node.get("fx_op") == "placeholder":
        label = f'{node["id"]} \u00b7 {kind}'
        detail = f"{size} {dtype}".strip()
        if not detail and geometry.get("kind") in ("list", "tuple"):
            items = geometry.get("items", [])
            detail = ",".join(str(v.get("value", v.get("kind", "?"))) for v in items)[:40]
        return [label, detail] if detail else [label]
    if kind == "output" or node.get("fx_op") == "output":
        return [str(node.get("label", "output")), size] if size else [str(node.get("label", "output"))]
    target = str(node.get("target") or node.get("op_label") or node.get("label") or node["id"])
    target = target.removeprefix("torch.ops.")
    if target.startswith("prims.convert_element_type"):
        target = f"cast \u2192 {dtype}"
    else:
        target = ".".join(target.split(".")[:2])
    target = target if len(target) <= 32 else target[:29] + "..."
    label = f'{node["id"]} \u00b7 {target}'
    return [label, f"{size} {dtype}".strip()] if node.get("is_matmul") or kind == "mm" else [label]


def _dot(original, lowering, fusion, clustered=True):
    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    lines = ["digraph fx {", 'graph [rankdir=LR, compound=true, newrank=true, nodesep=0.65, ranksep=0.65, pad=0.3, splines=true, fontname="Arial"];', 'node [shape=box, fontname="Arial", fontsize=11, margin="0.10,0.07", width=0, height=0];']
    for node in original["nodes"]:
        lines.append(f'{quote(node["id"])} [label={quote(chr(10).join(_labels(node)))}];')
    if clustered:
        memberships = defaultdict(list)
        for group in lowering:
            for key in group["member_fx_ids"]:
                memberships[key].append(group["id"])
        parent = {}
        for i, group in enumerate(fusion):
            for key in group["lowering_members"]:
                parent.setdefault(key, []).append(i)
        def cluster(group, i):
            unique = [key for key in group["member_fx_ids"] if memberships[key] == [group["id"]]]
            if not unique:
                return []
            return [f"subgraph cluster_lower_{i} {{", 'graph [label="", margin=28];', *(f"{quote(key)};" for key in unique), "}"]
        for j in range(len(fusion)):
            lines.extend([f"subgraph cluster_fused_{j} {{", 'graph [label="", margin=24];'])
            for i, group in enumerate(lowering):
                if parent.get(group["id"]) == [j]:
                    lines.extend(cluster(group, i))
            lines.append("}")
        for i, group in enumerate(lowering):
            if len(parent.get(group["id"], [])) != 1:
                lines.extend(cluster(group, i))
    for i, edge in enumerate(original["edges"]):
        lines.append(f'{quote(edge["source"])} -> {quote(edge["target"])} [id="layout_edge_{i}", arrowsize=0.6];')
    lines.append("}")
    return "\n".join(lines)


def _graph_layout(original, lowering, fusion, output_dir, *, clustered=True):
    if not shutil.which("dot"):
        raise FileNotFoundError("Graphviz 'dot' is required to lay out the original FX DAG")
    source = _dot(original, lowering, fusion, clustered=clustered)
    result = subprocess.run(["dot", "-Tjson"], input=source, capture_output=True, text=True)
    if result.returncode and clustered:
        clustered = False
        source = _dot(original, lowering, fusion, clustered=False)
        result = subprocess.run(["dot", "-Tjson"], input=source, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Graphviz could not lay out the captured DAG: {result.stderr.strip()}")
    (output_dir / "original_fx_regions.dot").write_text(source)
    layout = json.loads(result.stdout)
    height = float(layout["bb"].split(",")[3])
    objects = {node["name"]: node for node in layout["objects"]}
    nodes = []
    for original_node in original["nodes"]:
        geometry = objects[original_node["id"]]
        x, y = map(float, geometry["pos"].split(","))
        w, h = float(geometry["width"]) * 72, float(geometry["height"]) * 72
        nodes.append({**original_node, "is_matmul": original_node.get("is_matmul", False) or original_node.get("kind") == "mm", "labels": _labels(original_node), "center": [x, height - y], "bbox": {"x": x - w / 2, "y": height - y - h / 2, "width": w, "height": h}})
    geometry_edges = {edge.get("id"): edge for edge in layout.get("edges", [])}
    edges = []
    for i, original_edge in enumerate(original["edges"]):
        geometry = geometry_edges[f"layout_edge_{i}"]
        paths, arrows = [], []
        for draw in geometry.get("_draw_", []):
            if draw["op"] == "b":
                points = [(x, height - y) for x, y in draw["points"]]
                path = f"M {_number(points[0][0])} {_number(points[0][1])}"
                for j in range(1, len(points), 3):
                    path += " C " + " ".join(f"{_number(x)} {_number(y)}" for x, y in points[j:j + 3])
                paths.append(path)
        for draw in geometry.get("_hdraw_", []):
            if draw["op"] == "P":
                arrows.append([[x, height - y] for x, y in draw["points"]])
        edges.append({**original_edge, "id": original_edge.get("id", f"e{i}"), "paths": paths, "arrow_polygons": arrows})
    return nodes, edges, clustered



def _smooth(rings, radius):
    """Offset convex rings outward with constant-radius circular joins."""
    commands, sampled = [], []
    for polygon in rings:
        polygon = [point for i, point in enumerate(polygon) if math.dist(point, polygon[i - 1]) > 1e-8]
        if len(polygon) < 3:
            continue
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1]))
        if area < 0:
            polygon.reverse()
        if not radius:
            commands.extend({"command": "M" if i == 0 else "L", "points": [list(point)]} for i, point in enumerate(polygon))
            commands.append({"command": "Z", "points": []})
            sampled.append([list(point) for point in polygon])
            continue
        ring_commands, samples = [], []
        for i, vertex in enumerate(polygon):
            previous, following = polygon[i - 1], polygon[(i + 1) % len(polygon)]
            d1, d2 = math.dist(vertex, previous), math.dist(vertex, following)
            incoming = [(vertex[k] - previous[k]) / d1 for k in (0, 1)]
            outgoing = [(following[k] - vertex[k]) / d2 for k in (0, 1)]
            turn = math.atan2(incoming[0] * outgoing[1] - incoming[1] * outgoing[0], sum(incoming[k] * outgoing[k] for k in (0, 1)))
            if turn < -1e-8:
                raise ValueError("An outer envelope must be convex")
            start = [vertex[0] + incoming[1] * radius, vertex[1] - incoming[0] * radius]
            end = [vertex[0] + outgoing[1] * radius, vertex[1] - outgoing[0] * radius]
            ring_commands.append({"command": "L" if ring_commands else "M", "points": [start]})
            if abs(turn) < 1e-8:
                samples.append(start)
                continue
            angle = math.atan2(start[1] - vertex[1], start[0] - vertex[0])
            count = math.ceil(abs(turn) / (math.pi / 2))
            delta = turn / count
            for piece in range(count):
                a, b = angle + piece * delta, angle + (piece + 1) * delta
                first = start if piece == 0 else [vertex[0] + radius * math.cos(a), vertex[1] + radius * math.sin(a)]
                last = end if piece == count - 1 else [vertex[0] + radius * math.cos(b), vertex[1] + radius * math.sin(b)]
                handle = radius * 4 / 3 * math.tan(delta / 4)
                c1 = [first[0] - handle * math.sin(a), first[1] + handle * math.cos(a)]
                c2 = [last[0] + handle * math.sin(b), last[1] - handle * math.cos(b)]
                ring_commands.append({"command": "C", "points": [c1, c2, last]})
                for step in range(17):
                    t = step / 16
                    samples.append([(1 - t) ** 3 * first[k] + 3 * (1 - t) ** 2 * t * c1[k] + 3 * (1 - t) * t * t * c2[k] + t ** 3 * last[k] for k in (0, 1)])
        commands.extend(ring_commands)
        commands.append({"command": "Z", "points": []})
        sampled.append(samples)
    path = " ".join(command["command"] + "".join(f" {_number(p[0])} {_number(p[1])}" for p in command["points"]) for command in commands)
    return {"path": path, "path_commands": commands, "rings": sampled, "ring_roles": ["outer"] * len(sampled), "fill_rule": "nonzero", "components": len(sampled), "holes": 0, "bbox": _bbox([p for ring in sampled for p in ring]) if sampled else None, "rounding_method": "outward circular offset", "requested_corner_radius": radius, "minimum_corner_radius": radius if sampled and radius else None, "tight_corner_count": 0, "tight_tip_count": 0}


def _hull(points):
    ordered_points = sorted({tuple(point) for point in points})
    halves = []
    for ordered in (ordered_points, ordered_points[::-1]):
        half = []
        for point in ordered:
            while len(half) > 1:
                a, b = half[-2:]
                if (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) > 0:
                    break
                half.pop()
            half.append(point)
        halves.append(half[:-1])
    return halves[0] + halves[1]


def _outer_contour(group, nodes, children=()):
    members = set(group["member_fx_ids"])
    points = [point for node in nodes if node["id"] in members for point in _corners(node["bbox"])]
    points.extend(point for child in children for ring in child["rings"] for point in ring)
    shape = _smooth([_hull(points)], 24 if children else 30) if points else _smooth([], 0)
    shape["contour_method"] = "outward-rounded convex envelope"
    shape["geometry_status"] = "connected" if points else "unmapped_or_empty"
    return shape


def _contours(nodes, lowering, fusion):
    for group in lowering:
        group.update(_outer_contour(group, nodes))
    lower_by_id = {group["id"]: group for group in lowering}
    for group in fusion:
        children = [lower_by_id[key] for key in group["lowering_members"]]
        group.update(_outer_contour(group, nodes, children))


def _audit(nodes, lowering, fusion):
    by_id = {node["id"]: node for node in nodes}
    inclusion, intersections, enclosed, crossed, envelopes, nested = {}, {}, {}, {}, {}, {}
    containment = {group["svg_id"]: _Containment(group["rings"]) for group in [*lowering, *fusion]}
    for group in [*lowering, *fusion]:
        key = group["svg_id"]
        contains = containment[key].contains
        missing = [member for member in group["member_fx_ids"] if not all(contains(p) for p in _corners(by_id[member]["bbox"])) or _boundary_hits_rectangle(by_id[member]["bbox"], group["rings"])]
        foreign = [node["id"] for node in nodes if node["id"] not in group["member_fx_ids"] and group["bbox"] and _overlap(node["bbox"], group["bbox"]) and _intersects_rectangle(node["bbox"], group["rings"], contains)]
        inclusion[key] = not missing
        intersections[key] = foreign
        enclosed[key] = [node_id for node_id in foreign if all(contains(p) for p in _corners(by_id[node_id]["bbox"])) and not _boundary_hits_rectangle(by_id[node_id]["bbox"], group["rings"])]
        crossed[key] = [node_id for node_id in foreign if node_id not in enclosed[key]]
        group["incidental_enclosed_fx_ids"] = enclosed[key]
        group["incidental_intersected_fx_ids"] = crossed[key]
        convex = True
        for ring in group["rings"]:
            for i, b in enumerate(ring):
                a, c = ring[i - 1], ring[(i + 1) % len(ring)]
                turn = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
                convex &= turn >= -1e-7 * max(1, math.dist(a, b) * math.dist(b, c))
        envelopes[key] = convex and group["components"] == bool(group["path"]) and group["holes"] == 0
        if missing or not envelopes[key]:
            raise RuntimeError(f'Scheduler region {group["id"]} has missing member boxes {missing} or an invalid outer envelope')
    lower_by_id = {group["id"]: group for group in lowering}
    for group in fusion:
        contains = containment[group["svg_id"]].contains
        failed = [key for key in group["lowering_members"] if not all(contains(p) for ring in lower_by_id[key]["rings"] for p in ring)]
        nested[group["id"]] = not failed
        if failed:
            raise RuntimeError(f'Fused region {group["id"]} does not contain complete lowering regions {failed}')
    return {"full_member_inclusion": inclusion, "nonmember_intersections": intersections, "incidental_enclosed": enclosed, "incidental_boundary_intersections": crossed, "nonmember_overlap_allowed": True, "smooth_outer_envelopes": envelopes, "whole_lowering_contours_inside_fusion": nested, "passed": True}


def _place_labels(nodes, groups):
    labels = []
    for group in groups:
        if not group["bbox"]:
            group["label_bbox"], group["label_center"] = None, None
            continue
        box = group["bbox"]
        width, height = min(180, max(28, len(str(group["label"])) * 6)), 15
        candidates = [(box["x"] + width / 2, box["y"] - 10), (box["x"] + box["width"] - width / 2, box["y"] - 10), (box["x"] + width / 2, box["y"] + box["height"] + 11)]
        for node in nodes:
            if node["id"] in group["member_fx_ids"]:
                candidates.append((node["center"][0], node["bbox"]["y"] - 24))
        for cx, cy in candidates:
            candidate = {"x": cx - width / 2, "y": cy - height / 2, "width": width, "height": height}
            if not any(_overlap(candidate, node["bbox"]) for node in nodes) and not any(_overlap(candidate, label) for label in labels):
                break
        else:
            candidate = {"x": box["x"], "y": -40 - 18 * len(labels), "width": width, "height": height}
            cx, cy = candidate["x"] + width / 2, candidate["y"] + height / 2
        group["label_bbox"], group["label_center"] = candidate, [cx, cy]
        labels.append(candidate)


def _svg(payload, *, id_prefix=""):
    attr = lambda value: html.escape(str(value), quote=True)
    svg_id = lambda value: attr(id_prefix + value)
    view_box = " ".join(str(v) for v in payload["view_box"])
    x, y, w, h = payload["view_box"]
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{svg_id("scheduler-region-title")} {svg_id("scheduler-region-desc")}" viewBox="{view_box}" style="background:{_BACKGROUND};color:{_FOREGROUND}">', f'<title id="{svg_id("scheduler-region-title")}">Original FX with observed scheduler regions</title>', f'<desc id="{svg_id("scheduler-region-desc")}">The original directed FX graph is preserved. Solid broad convex envelopes show lowering groups; thicker dashed envelopes enclose their actual fused groups. Envelopes may also contain unrelated nodes. Select a boundary to highlight only its exact computation and indexing contributors. Shared contributors may belong to more than one group. Missing mappings are explicit in the accompanying payload.</desc>', f'<rect x="{_number(x)}" y="{_number(y)}" width="{_number(w)}" height="{_number(h)}" fill="{_BACKGROUND}"/>', f'<g id="{svg_id("fx-region-scene")}" data-scheduler-region-scene="true">', f'<g id="{svg_id("scheduler-region-boundaries")}">']
    for fused, groups in ((False, payload["lowering_groups"]), (True, payload["fusion_groups"])):
        for group in groups:
            cls = "fusion-boundary" if fused else "lowering-boundary"
            dash = ' stroke-dasharray="11 6"' if fused else ""
            chunks.append(f'<g id="{svg_id(group["svg_id"])}" class="{cls}" data-scheduler="{attr(group["id"])}" data-fx-ids="{attr(" ".join(group["member_fx_ids"]))}"><title>{attr(group["id"])}: computation {attr(", ".join(group["execution_fx_ids"]))}; indexing {attr(", ".join(group["indexing_fx_ids"]))}; {attr(group["geometry_status"])}</title>')
            if group["path"]:
                chunks.append(f'<path d="{group["path"]}" fill="{group["color"]}" fill-rule="{group["fill_rule"]}" fill-opacity="{0.025 if fused else 0.075}" stroke="{group["color"]}" stroke-width="{3.6 if fused else 2.1}"{dash} vector-effect="non-scaling-stroke"/>')
            chunks.append("</g>")
    chunks.append(f'</g><g id="{svg_id("scheduler-region-edges")}">')
    for i, edge in enumerate(payload["edges"]):
        chunks.append(f'<g id="{svg_id(f"fx-edge-{i}")}" class="fx-edge" data-source="{attr(edge["source"])}" data-target="{attr(edge["target"])}"><title>{attr(edge["source"])} \u2192 {attr(edge["target"])}: {attr(edge.get("label", edge.get("ports", edge.get("port", ""))))}</title>')
        chunks.extend(f'<path d="{path}" fill="none" stroke="{_MUTED}" stroke-width="1.15" stroke-opacity="0.8"/>' for path in edge["paths"])
        chunks.extend(f'<polygon points="{" ".join(f"{_number(px)},{_number(py)}" for px, py in polygon)}" fill="{_MUTED}"/>' for polygon in edge["arrow_polygons"])
        chunks.append("</g>")
    chunks.append(f'</g><g id="{svg_id("scheduler-region-nodes")}">')
    lower_by_id = {group["id"]: group for group in payload["lowering_groups"]}
    for i, node in enumerate(payload["nodes"]):
        box = node["bbox"]
        color = lower_by_id[node["lowering_groups"][0]]["color"] if node["lowering_groups"] else _BORDER
        if node["indexing_only"]:
            color = _MUTED
        dash = ' stroke-dasharray="4 3"' if node["indexing_only"] or node.get("kind") == "shape" else ""
        fill = _BACKGROUND if node["lowering_groups"] else _SURFACE
        tooltip = f'{node["id"]}: {node.get("target", node.get("label", ""))}; lowering={node["lowering_groups"]}; role={node["contribution_kind"]}'
        chunks.append(f'<g id="{svg_id(f"fx-node-{i}")}" class="fx-node" data-fx-id="{attr(node["id"])}" data-lowering-groups="{attr(" ".join(node["lowering_groups"]))}" data-fusion-groups="{attr(" ".join(node["fusion_groups"]))}" role="button" tabindex="0" aria-label="{attr(tooltip)}"><title>{attr(tooltip)}</title><rect x="{_number(box["x"])}" y="{_number(box["y"])}" width="{_number(box["width"])}" height="{_number(box["height"])}" rx="5" fill="{fill}" stroke="{color}" stroke-width="{2 if node.get("is_matmul") else 1.2}"{dash}/>' )
        for j, label in enumerate(node["labels"]):
            py = node["center"][1] + (j - (len(node["labels"]) - 1) / 2) * 14 + 4
            chunks.append(f'<text x="{_number(node["center"][0])}" y="{_number(py)}" text-anchor="middle" fill="{_FOREGROUND if j == 0 else _MUTED}" font-family="Arial,sans-serif" font-size="{11 if j == 0 else 10}" font-weight="{700 if node.get("is_matmul") and j == 0 else 400}">{html.escape(str(label))}</text>')
        chunks.append("</g>")
    chunks.append(f'</g><g id="{svg_id("scheduler-region-labels")}">')
    for fused, groups in ((False, payload["lowering_groups"]), (True, payload["fusion_groups"])):
        for group in groups:
            if group["label_bbox"]:
                box = group["label_bbox"]
                cx, cy = group["label_center"]
                cls = "fusion-boundary" if fused else "lowering-boundary"
                label = str(group["label"])
                visible = label if len(label) <= 26 else label[:23] + "..."
                chunks.append(f'<g class="{cls} region-label" data-scheduler="{attr(group["id"])}"><title>{attr(label)}</title><rect x="{_number(box["x"])}" y="{_number(box["y"])}" width="{_number(box["width"])}" height="{_number(box["height"])}" rx="3" fill="{_BACKGROUND}" fill-opacity="0.9"/><text x="{_number(cx)}" y="{_number(cy + 4)}" text-anchor="middle" fill="{group["color"]}" font-family="Arial,sans-serif" font-size="11" font-weight="700">{html.escape(visible)}</text></g>')
    chunks.append("</g></g></svg>")
    return "\n".join(chunks)


def layout_regions(capture: dict, output_dir: Path, *, clustered=True) -> dict:
    """Return original-FX geometry with observed lowering and fusion contours.

    Graphviz supplies the directed layout. Broad convex envelopes can include
    unrelated nodes. Interactive highlighting uses exact group membership.
    Membership comes only from execution and indexing annotations, never origins.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original = capture["original_graph"]
    if not original.get("nodes"):
        raise ValueError("The captured original FX graph has no nodes")
    ids = [node["id"] for node in original["nodes"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Original FX node IDs must be unique")
    if any(edge["source"] not in ids or edge["target"] not in ids for edge in original["edges"]):
        raise ValueError("An original FX edge has a missing endpoint")
    lowering, fusion = _regions(capture)
    nodes, edges, clustered = _graph_layout(original, lowering, fusion, output_dir, clustered=clustered)
    _contours(nodes, lowering, fusion)
    audit = _audit(nodes, lowering, fusion)
    _place_labels(nodes, [*lowering, *fusion])
    computation = {key for group in lowering for key in group["execution_fx_ids"]}
    indexing = set(capture.get("execution_annotation", {}).get("original_indexing_fx_ids", [])) | {key for group in lowering for key in group["indexing_fx_ids"]}
    for node in nodes:
        key = node["id"]
        node["lowering_groups"] = [group["id"] for group in lowering if key in group["member_fx_ids"]]
        node["fusion_groups"] = [group["id"] for group in fusion if key in group["member_fx_ids"]]
        node["indexing_only"] = key in indexing and key not in computation
        node["contribution_kind"] = "computation_and_indexing" if key in computation and key in indexing else ("computation" if key in computation else ("indexing" if key in indexing else "unassigned_or_graph_boundary"))
    points = [p for node in nodes for p in _corners(node["bbox"])]
    points.extend(p for group in [*lowering, *fusion] for ring in group["rings"] for p in ring)
    points.extend(p for group in [*lowering, *fusion] if group["label_bbox"] for p in _corners(group["label_bbox"]))
    canvas = _bbox(points)
    payload = {
        "name": capture.get("name", original.get("name", "captured_fx")), "model": original.get("source_model"),
        "source_capture": capture.get("source_capture"), "source_info": {"source_graph": original.get("source_graph"), "pattern_hash": original.get("pattern_hash"), "compile": capture.get("compile", {}), "backend_execution_proof": capture.get("backend_execution_proof", {})},
        "view_box": [canvas["x"] - 12, canvas["y"] - 12, canvas["width"] + 24, canvas["height"] + 24],
        "nodes": nodes, "edges": edges, "lowering_groups": lowering, "fusion_groups": fusion,
        "indexing_fx_ids": [key for key in ids if key in indexing],
        "indexing_memberships": {key: [group["id"] for group in lowering if key in group["indexing_fx_ids"]] for key in ids if key in indexing},
        "geometry_audit": audit, "geometry_method": {"layout": "Graphviz LR", "unique_owner_clusters": clustered, "contours": "outward-rounded convex envelopes", "padding": {"lowering": 30, "fusion": 24}, "fill_rule": "nonzero", "nonmember_overlap": "allowed; membership comes from scheduler evidence"},
        "counts": {"nodes": len(nodes), "edges": len(edges), "lowering_groups": len(lowering), "fusion_groups": len(fusion), "post_fusion_scheduler_nodes": len(capture["stages"]["fusion"]["scheduler_nodes"])},
        "fx_status": capture.get("fx_status", {}), "execution_annotation": capture.get("execution_annotation", {}),
        "artifacts": {"svg": str(output_dir / "original_fx_regions.svg"), "payload": str(output_dir / "region_payload.json")},
    }
    (output_dir / "region_payload.json").write_text(json.dumps(payload, indent=2))
    (output_dir / "original_fx_regions.svg").write_text(_svg(payload))
    return payload
