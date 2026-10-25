import base64
import copy
import gzip
import importlib
import importlib.util
import inspect
import io
import json
import math
import sys
import subprocess
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import Mock, patch

from fx_visualizer import visualize_graph, write_index
from fx_visualizer.integrations.better_benchmark import visualize_repro, visualize_repro_matmul_comparison
from fx_visualizer.regions import _smooth, layout_regions
from fx_visualizer.rendering import (
    _pack_viewers,
    render_matmul_comparison,
    render_capture,
    visualize_matmul_comparison,
)


BB_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("oracle_harness", "repro_harness", "input_codec"))


class WidgetHTML(HTMLParser):
    def __init__(self, widget):
        super().__init__()
        self.elements = []
        self.feed(widget)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def decode_widget_viewers(widget):
    payloads = [attrs["data-fx-payload"] for _, attrs in WidgetHTML(widget).elements if "data-fx-payload" in attrs]
    if len(payloads) != 1:
        raise ValueError(f"Expected one compressed widget payload, got {len(payloads)}")
    decoded = json.loads(gzip.decompress(base64.b64decode(payloads[0])))
    if "viewers" not in decoded:
        return [decoded]
    return [
        {"root_id": viewer["root_id"], "svg": "".join(decoded["svg_lines"][index] for index in viewer["svg_lines"]), "data": viewer["data"]}
        for viewer in decoded["viewers"]
    ]


class TestViewerPacking(unittest.TestCase):
    def test_shared_markup_recovers_exact_svg_metadata_and_independent_mode_data(self):
        shared = "".join(f'<path d="M{index} {index * index} C{index + 1} {index * 7919} {index + 2} {index * 3571} {index + 3} {index * 1013}"/>\r\n' for index in range(700))
        viewers = [
            {
                "root_id": mode,
                "svg": f'<svg id="{mode}" viewBox="0 0 42 17">\n<title>{mode} \u03bb</title>\r{shared}</svg>',
                "data": {
                    "view_box": [0, 0, 42, 17],
                    "edges": [{"source": "mm", "target": "output", "ports": [["output", 14], ["tuple", 1]]}],
                    "nodes": [{"id": "output", "geometry": {"kind": "tuple", "items": [{"shape": [16, 32]}, {"shape": [16, 1]}]}}],
                    "fusion_groups": [{"id": mode, "member_fx_ids": ["mm"], "lowering_members": ["dot", "cast"]}],
                },
            }
            for mode in ("template", "native")
        ]
        before = copy.deepcopy(viewers)
        packed = _pack_viewers(viewers)
        encoded = gzip.compress(json.dumps(packed, separators=(",", ":")).encode())
        independent_bytes = sum(len(gzip.compress(json.dumps(viewer, separators=(",", ":")).encode())) for viewer in viewers)
        self.assertLess(len(encoded), independent_bytes * 0.75)
        decoded = json.loads(gzip.decompress(encoded))
        restored = [
            {"root_id": viewer["root_id"], "svg": "".join(decoded["svg_lines"][index] for index in viewer["svg_lines"]), "data": viewer["data"]}
            for viewer in decoded["viewers"]
        ]
        self.assertEqual(restored, before)
        restored[0]["data"]["fusion_groups"][0]["member_fx_ids"].clear()
        restored[0]["data"]["nodes"][0]["geometry"]["items"][0]["shape"][0] = 999
        self.assertEqual(restored[1:], before[1:])
        self.assertEqual(viewers, before)


def inside_region(point, rings):
    x, y = point
    enclosed_rings = 0
    for ring in rings:
        inside = False
        for a, b in zip(ring, ring[1:] + ring[:1]):
            if (a[1] > y) != (b[1] > y):
                crossing = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                if x < crossing:
                    inside = not inside
        enclosed_rings += inside
    return enclosed_rings % 2 == 1


def scheduler_group(group_id, operation_names, computation=(), indexing=(), unknown=()):
    return {
        "id": group_id,
        "name": group_id,
        "member_operation_names": list(operation_names),
        "execution_fx_ids": list(computation),
        "indexing_fx_ids": list(indexing),
        "origin_fx_ids": ["activation", "unmapped"],
        "execution_evidence": {
            "unknown_fx_ids": list(unknown),
            "certainty": "partial" if unknown else "observed IR boundaries",
        },
    }


def capture_fixture():
    nodes = [
        {"id": node_id, "kind": kind, "target": target, "label": node_id}
        for node_id, kind, target in (
            ("input", "input", "input"),
            ("activation", "aten", "aten.add.Tensor"),
            ("shared_view", "aten", "aten.view.default"),
            ("left_mm", "mm", "aten.mm.default"),
            ("right_mm", "mm", "aten.mm.default"),
            ("cast", "prims", "prims.convert_element_type.default"),
            ("unmapped", "aten", "aten.neg.default"),
            ("left_output", "output", "output"),
            ("right_output", "output", "output"),
            ("unmapped_output", "output", "output"),
        )
    ]
    edges = [
        {"source": source, "target": target, "ports": ports}
        for source, target, ports in (
            ("input", "activation", [["argument", "self"], ["argument", "other"]]),
            ("activation", "shared_view", [["argument", "self"]]),
            ("shared_view", "left_mm", [["argument", "self"]]),
            ("shared_view", "right_mm", [["argument", "self"]]),
            ("input", "left_mm", [["argument", "mat2"]]),
            ("input", "right_mm", [["argument", "mat2"]]),
            ("left_mm", "cast", [["argument", "a"]]),
            ("input", "unmapped", [["argument", "self"]]),
            ("cast", "left_output", [["output", 0]]),
            ("right_mm", "right_output", [["output", 1]]),
            ("unmapped", "unmapped_output", [["output", 2]]),
        )
    ]
    lowering = [
        scheduler_group("prepare", ["arithmetic-leaf"], ["activation"]),
        scheduler_group("lower-left", ["template-a"], ["left_mm"], ["shared_view"]),
        scheduler_group("lower-cast", ["cast-a"], ["cast"]),
        scheduler_group("lower-right", ["template-b"], ["right_mm"], ["shared_view"]),
        scheduler_group("unknown-tail", ["tail-no-source"], unknown=["unmapped"]),
    ]
    fusion = [
        scheduler_group("post-prepare", ["arithmetic-leaf"], ["activation"]),
        scheduler_group("unrelated-fused-name", ["cast-a", "template-a"], ["cast", "left_mm"], ["shared_view"]),
        scheduler_group("post-right", ["template-b"], ["right_mm"], ["shared_view"]),
        scheduler_group("post-unknown", ["tail-no-source"], unknown=["unmapped"]),
    ]
    return {
        "name": "synthetic_shared_view",
        "original_graph": {"nodes": nodes, "edges": edges},
        "stages": {
            "lowering": {"scheduler_nodes": lowering},
            "fusion": {"scheduler_nodes": fusion},
        },
    }


def capture_cpu_graph_fixture(gm, example_inputs, output_dir, **options):
    """Provide synthetic scheduler evidence over real CPU input geometry."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    from torch.utils import _pytree as pytree
    from fx_visualizer.graph import input_geometry, normalize_fx_graph

    captured = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
    with FakeTensorMode(allow_fallback_kernels=False) as mode:
        fake_inputs = pytree.tree_map(lambda value: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value, tuple(example_inputs))
        FakeTensorProp(captured, mode=mode).propagate_dont_convert_inputs(*fake_inputs)
    original, _ = normalize_fx_graph(captured, example_inputs)
    geometries, aliases, storages = input_geometry(example_inputs)
    computation = [node["id"] for node in original["nodes"] if node["fx_op"] == "call_function"]
    group = scheduler_group("stub-lowering", ["stub-operation"], computation=computation)
    group["origin_fx_ids"] = computation
    output_dir.mkdir(parents=True)
    capture = {
        "name": output_dir.name,
        "original_graph": original,
        "compile": {
            "mode": "schedule_only", "status": "success", "entrypoint": "test scheduler capture stub",
            "options": dict(options.get("config") or {}), "inputs": geometries,
            "input_aliases": aliases, "input_storage_groups": storages,
            "execution_succeeded": False, "autotuning_performed": False,
        },
        "stages": {
            "lowering": {"scheduler_nodes": [group]},
            "fusion": {"scheduler_nodes": [copy.deepcopy(group)]},
        },
        "capture_path": str(output_dir / "capture.json"),
    }
    Path(capture["capture_path"]).write_text(json.dumps(capture))
    return capture


class TestBoundarySmoothing(unittest.TestCase):
    def test_acute_corner_has_a_real_rounding_radius(self):
        shape = _smooth([[(0, 0), (400, 0), (0, 100)]], 30)
        curve_samples = 0
        current = first = None
        tangents = []
        for segment in shape["path_commands"]:
            command = segment["command"]
            points = [first] if command == "Z" else segment["points"]
            if command == "M":
                current = first = points[0]
                continue
            if command in ("L", "Z"):
                delta = [points[0][k] - current[k] for k in (0, 1)]
                length = math.hypot(*delta)
                if length > 1e-8:
                    unit = [value / length for value in delta]
                    tangents.append((unit, unit))
            else:
                self.assertEqual(command, "C")
                controls = [current, *points]
                directions = []
                for step in range(11):
                    t = step / 10
                    a, b, c, d = controls
                    velocity = [3 * (1 - t) ** 2 * (b[k] - a[k]) + 6 * (1 - t) * t * (c[k] - b[k]) + 3 * t * t * (d[k] - c[k]) for k in (0, 1)]
                    acceleration = [6 * ((1 - t) * (c[k] - 2 * b[k] + a[k]) + t * (d[k] - 2 * c[k] + b[k])) for k in (0, 1)]
                    speed = math.hypot(*velocity)
                    cross = abs(velocity[0] * acceleration[1] - velocity[1] * acceleration[0])
                    self.assertGreater(speed, 0)
                    if cross > 1e-8:
                        radius = speed ** 3 / cross
                        self.assertGreaterEqual(radius, 28, (command, t, radius))
                        curve_samples += 1
                    directions.append([value / speed for value in velocity])
                tangents.append((directions[0], directions[-1]))
            current = points[-1]
        self.assertGreater(curve_samples, 0, "The outline must include rounded corners")
        for (_, end), (start, _) in zip(tangents, tangents[1:] + tangents[:1]):
            self.assertGreater(sum(a * b for a, b in zip(end, start)), 0.995, "Contour joins must preserve the tangent direction")


class TestSchedulerVisualization(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        self.capture = capture_fixture()
        self.snapshot = copy.deepcopy(self.capture)
        self.payload = layout_regions(self.capture, Path(self.output.name))
        self.lowering = {group["id"]: group for group in self.payload["lowering_groups"]}
        self.fusion = {group["id"]: group for group in self.payload["fusion_groups"]}

    def test_shared_indexing_belongs_to_both_groups_without_copying_computation(self):
        left = self.lowering["lower-left"]
        right = self.lowering["lower-right"]
        self.assertEqual(set(left["member_fx_ids"]), {"left_mm", "shared_view"})
        self.assertEqual(set(right["member_fx_ids"]), {"right_mm", "shared_view"})
        self.assertEqual(left["execution_fx_ids"], ["left_mm"])
        self.assertEqual(right["execution_fx_ids"], ["right_mm"])
        self.assertEqual(left["indexing_fx_ids"], ["shared_view"])
        self.assertEqual(right["indexing_fx_ids"], ["shared_view"])
        self.assertNotIn("activation", left["member_fx_ids"])
        self.assertNotIn("unmapped", left["member_fx_ids"])

    def test_fusion_uses_actual_operation_members_and_excludes_other_matmul(self):
        fused = self.fusion["unrelated-fused-name"]
        self.assertEqual(set(fused["lowering_members"]), {"lower-left", "lower-cast"})
        self.assertEqual(set(fused["member_fx_ids"]), {"shared_view", "left_mm", "cast"})
        self.assertEqual(set(fused["execution_fx_ids"]), {"left_mm", "cast"})
        self.assertEqual(fused["indexing_fx_ids"], ["shared_view"])
        self.assertNotIn("right_mm", fused["member_fx_ids"])
        self.assertNotIn("lower-right", fused["lowering_members"])
        for lower_id in fused["lowering_members"]:
            for ring in self.lowering[lower_id]["rings"]:
                for point in ring:
                    self.assertTrue(inside_region(point, fused["rings"]), (lower_id, point))
        for group in [*self.lowering.values(), *self.fusion.values()]:
            members = set(group["member_fx_ids"])
            if not members:
                continue
            self.assertEqual(group["components"], 1)
            self.assertEqual(group["holes"], 0)
            for node in self.payload["nodes"]:
                if node["id"] not in members:
                    continue
                bbox = node["bbox"]
                left, top = bbox["x"], bbox["y"]
                right, bottom = left + bbox["width"], top + bbox["height"]
                for point in ((left, top), (left, bottom), (right, top), (right, bottom)):
                    with self.subTest(group=group["id"], node=node["id"], point=point):
                        self.assertTrue(inside_region(point, group["rings"]))

    def test_broad_fusion_envelope_can_enclose_a_nonmember_without_changing_membership(self):
        nodes = [
            {"id": node_id, "kind": kind, "target": target, "label": node_id}
            for node_id, kind, target in (
                ("input", "input", "input"),
                ("shared_view", "aten", "aten.view.default"),
                ("left_mm", "mm", "aten.mm.default"),
                ("middle_branch", "aten", "aten.neg.default"),
                ("right_mm", "mm", "aten.mm.default"),
                ("left_output", "output", "output"),
                ("middle_output", "output", "output"),
                ("right_output", "output", "output"),
            )
        ]
        edges = [
            {"source": source, "target": target, "ports": [["argument", "self"]]}
            for source, target in (
                ("input", "shared_view"),
                ("shared_view", "left_mm"),
                ("shared_view", "middle_branch"),
                ("shared_view", "right_mm"),
                ("left_mm", "left_output"),
                ("middle_branch", "middle_output"),
                ("right_mm", "right_output"),
            )
        ]
        capture = {
            "name": "parallel_mm_with_incidental_branch",
            "original_graph": {"nodes": nodes, "edges": edges},
            "stages": {
                "lowering": {"scheduler_nodes": [
                    scheduler_group("left", ["dot-left"], ["left_mm"], ["shared_view"]),
                    scheduler_group("middle", ["neg"], ["middle_branch"]),
                    scheduler_group("right", ["dot-right"], ["right_mm"], ["shared_view"]),
                ]},
                "fusion": {"scheduler_nodes": [
                    scheduler_group("joined-mm", ["dot-left", "dot-right"], ["left_mm", "right_mm"], ["shared_view"]),
                    scheduler_group("post-middle", ["neg"], ["middle_branch"]),
                ]},
            },
        }
        before = copy.deepcopy(capture)
        artifacts = render_capture(capture=capture, output_dir=Path(self.output.name) / "incidental", clustered=False)
        payload = json.loads(Path(artifacts["payload"]).read_text())
        fused = next(group for group in payload["fusion_groups"] if group["id"] == "joined-mm")
        node = next(node for node in payload["nodes"] if node["id"] == "middle_branch")
        box = node["bbox"]
        for x in (box["x"], box["x"] + box["width"]):
            for y in (box["y"], box["y"] + box["height"]):
                self.assertTrue(inside_region((x, y), fused["rings"]), "The unrelated parallel branch should be visibly enclosed")
        self.assertEqual(fused["components"], 1)
        self.assertEqual(fused["holes"], 0)
        self.assertEqual(fused["ring_roles"], ["outer"])
        self.assertEqual(set(fused["member_fx_ids"]), {"left_mm", "right_mm", "shared_view"})
        self.assertEqual(set(fused["execution_fx_ids"]), {"left_mm", "right_mm"})
        self.assertEqual(fused["indexing_fx_ids"], ["shared_view"])
        self.assertEqual(fused["lowering_members"], ["left", "right"])
        self.assertIn("middle_branch", fused["incidental_enclosed_fx_ids"])
        audit = payload["geometry_audit"]
        self.assertTrue(audit["nonmember_overlap_allowed"])
        self.assertIn("middle_branch", audit["incidental_enclosed"][fused["svg_id"]])
        self.assertTrue(audit["passed"])
        self.assertEqual(capture, before)
        embedded = decode_widget_viewers(Path(artifacts["widget"]).read_text())[0]["data"]
        group = next(group for group in embedded["fusion_groups"] if group["id"] == "joined-mm")
        self.assertEqual(group["member_fx_ids"], fused["member_fx_ids"])
        self.assertEqual(group["lowering_members"], ["left", "right"])
        middle = next(group for group in embedded["lowering_groups"] if group["id"] == "middle")
        self.assertEqual(middle["member_fx_ids"], ["middle_branch"])
        self.assertNotIn(middle["id"], group["lowering_members"])
        self.assertIn("middle_branch", group["incidental_enclosed_fx_ids"])

    def test_original_ports_and_unmapped_nodes_are_preserved(self):
        self.assertEqual(self.capture, self.snapshot)

        original_ids = {node["id"] for node in self.capture["original_graph"]["nodes"]}
        self.assertEqual({node["id"] for node in self.payload["nodes"]}, original_ids)
        original_edges = self.capture["original_graph"]["edges"]
        actual_edges = [{key: edge[key] for key in ("source", "target", "ports")} for edge in self.payload["edges"]]
        self.assertCountEqual(actual_edges, original_edges)
        unknown = self.lowering["unknown-tail"]
        self.assertEqual(unknown["member_fx_ids"], [])
        self.assertEqual(unknown["execution_fx_ids"], [])
        self.assertEqual(unknown["indexing_fx_ids"], [])
        self.assertEqual(unknown["unknown_fx_ids"], ["unmapped"])
        self.assertEqual(unknown["certainty"], "partial")
        self.assertEqual(unknown["geometry_status"], "unmapped_or_empty")
        self.assertEqual(unknown["rings"], [])
        self.assertEqual(unknown["path"], "")
        self.assertIsNone(unknown["bbox"])
        saved = json.loads((Path(self.output.name) / "region_payload.json").read_text())
        self.assertEqual(saved, json.loads(json.dumps(self.payload)))
        self.assertTrue((Path(self.output.name) / "original_fx_regions.svg").is_file())

    def test_unchanged_multi_operation_scheduler_keeps_boundary_without_new_fusion(self):
        capture = copy.deepcopy(self.capture)
        for stage in ("lowering", "fusion"):
            capture["stages"][stage]["scheduler_nodes"][0]["member_operation_names"].append("existing-second-leaf")
        payload = layout_regions(capture, Path(self.output.name) / "unchanged")
        self.assertEqual([group["id"] for group in payload["fusion_groups"]], [node["id"] for node in capture["stages"]["fusion"]["scheduler_nodes"]])
        unchanged = next(group for group in payload["fusion_groups"] if group["id"] == "post-prepare")
        self.assertEqual(unchanged["lowering_members"], ["prepare"])
        self.assertEqual(unchanged["accepted_fusion_events"], [])
        self.assertTrue(unchanged["path"])

    def test_every_remaining_scheduler_class_has_one_dashed_boundary(self):
        capture = copy.deepcopy(self.capture)
        final = capture["stages"]["fusion"]["scheduler_nodes"]
        final.append(scheduler_group("post-foreach", ["foreach-leaf"], ["unmapped"]))
        classes = ["SchedulerNode", "FusedSchedulerNode", "ExternKernelSchedulerNode", "NopKernelSchedulerNode", "ForeachKernelSchedulerNode"]
        for node, name in zip(final, classes):
            node["class"] = name
        before = copy.deepcopy(capture)
        artifacts = render_capture(capture=capture, output_dir=Path(self.output.name) / "all_classes")
        payload = json.loads(Path(artifacts["payload"]).read_text())
        expected = {node["id"]: node["class"] for node in final}
        self.assertEqual({group["id"]: group["class"] for group in payload["fusion_groups"]}, expected)
        self.assertEqual(artifacts["fusion_region_count"], len(final))
        svg = ET.parse(artifacts["svg"])
        outlined = {}
        for element in svg.iter():
            if element.attrib.get("class") != "fusion-boundary":
                continue
            paths = element.findall("{http://www.w3.org/2000/svg}path")
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].attrib["stroke-dasharray"], "11 6")
            outlined[element.attrib["data-scheduler"]] = element.attrib["data-scheduler-class"]
        self.assertEqual(outlined, expected)
        unknown = next(group for group in payload["fusion_groups"] if group["id"] == "post-unknown")
        self.assertEqual(unknown["member_fx_ids"], [])
        self.assertEqual(unknown["matmul_fx_ids"], [])
        self.assertEqual(unknown["unknown_fx_ids"], ["unmapped"])
        self.assertEqual(unknown["geometry_status"], "unmapped_or_empty")
        self.assertEqual(unknown["contour_method"], "unmapped scheduler marker")
        data = decode_widget_viewers(Path(artifacts["widget"]).read_text())[0]["data"]
        self.assertEqual({group["id"]: group["class"] for group in data["fusion_groups"]}, expected)
        self.assertIn("After-fusion boundaries 5", data["summary"])
        self.assertEqual(capture, before)

    def test_matmul_labels_count_computation_and_keep_exact_fused_members(self):
        capture = copy.deepcopy(self.capture)
        for stage in ("lowering", "fusion"):
            capture["stages"][stage]["scheduler_nodes"][0]["indexing_fx_ids"] = ["right_mm"]
        artifacts = render_capture(capture=capture, output_dir=Path(self.output.name) / "matmul_members")
        payload = json.loads(Path(artifacts["payload"]).read_text())
        groups = {group["id"]: group for group in payload["fusion_groups"]}
        self.assertEqual(groups["unrelated-fused-name"]["matmul_fx_ids"], ["left_mm"])
        self.assertEqual(groups["post-right"]["matmul_fx_ids"], ["right_mm"])
        self.assertEqual(groups["post-prepare"]["matmul_fx_ids"], [])
        self.assertIn("right_mm", groups["post-prepare"]["indexing_fx_ids"])
        svg = ET.parse(artifacts["svg"])
        labels = {element.attrib["data-scheduler"]: element.find("{http://www.w3.org/2000/svg}text").text for element in svg.iter() if element.attrib.get("class") == "fusion-boundary region-label"}
        self.assertTrue(labels["unrelated-fused-name"].endswith("MM 1"))
        self.assertTrue(labels["post-right"].endswith("MM 1"))
        self.assertTrue(labels["post-prepare"].endswith("MM 0"))
        data = decode_widget_viewers(Path(artifacts["widget"]).read_text())[0]["data"]
        self.assertEqual({group["id"]: group["matmul_fx_ids"] for group in data["fusion_groups"]}, {name: group["matmul_fx_ids"] for name, group in groups.items()})

    def test_saved_capture_widget_preserves_members_edges_and_local_ids(self):
        previous_ids = set()
        for index in range(2):
            artifacts = render_capture(capture=self.capture, output_dir=Path(self.output.name) / str(index))
            widget = Path(artifacts["widget"]).read_text()
            markup = WidgetHTML(widget)
            decoded = decode_widget_viewers(widget)[0]
            data = decoded["data"]
            self.assertTrue("edges" in data, "The embedded widget must retain original argument ports and edges")
            expected_edges = self.capture["original_graph"]["edges"]
            actual_edges = [{key: edge[key] for key in ("source", "target", "ports")} for edge in data["edges"]]
            self.assertCountEqual(actual_edges, expected_edges)
            for field in ("lowering_groups", "fusion_groups"):
                expected = {group["id"]: group for group in self.payload[field]}
                actual = {group["id"]: group for group in data[field]}
                self.assertEqual(set(actual), set(expected))
                for group_id, group in actual.items():
                    for member_field in ("member_fx_ids", "execution_fx_ids", "indexing_fx_ids"):
                        self.assertEqual(group[member_field], expected[group_id][member_field])
            root_id = next(attrs["id"] for tag, attrs in markup.elements if tag == "section")
            svg = ET.fromstring(decoded["svg"])
            svg_ids = [element.attrib["id"] for element in svg.iter() if "id" in element.attrib]
            source_ids = [element.attrib["data-fx-id"] for element in svg.iter() if "data-fx-id" in element.attrib]
            self.assertCountEqual(source_ids, [node["id"] for node in self.capture["original_graph"]["nodes"]])
            ids = [attrs["id"] for _, attrs in markup.elements if "id" in attrs] + svg_ids
            self.assertEqual(len(ids), len(set(ids)))
            self.assertFalse(previous_ids.intersection(ids))
            self.assertTrue(all(svg_id.startswith(root_id + "-") for svg_id in svg_ids))
            previous_ids.update(ids)
            saved_capture = json.loads(Path(artifacts["capture"]).read_text())
            self.assertEqual(saved_capture, self.snapshot)
        self.assertEqual(self.capture, self.snapshot)

    def test_live_api_persists_public_name_for_saved_capture_reload(self):
        inputs = [object(), [16, 128]]
        graph = object()
        destination = Path(self.output.name) / "my_graph"

        def fake_capture(actual_graph, actual_inputs, output_dir, **kwargs):
            self.assertIs(actual_graph, graph)
            self.assertIs(actual_inputs, inputs)
            self.assertEqual(kwargs, {"config": {"epilogue_fusion": False}, "triton_matmul": False, "schedule_only": True})
            output_dir.mkdir(parents=True)
            capture = dict(copy.deepcopy(self.capture), name="compiler", capture_path=str(output_dir / "capture.json"))
            Path(capture["capture_path"]).write_text(json.dumps(capture))
            return capture

        capture_module = types.ModuleType("fx_visualizer.capture")
        capture_module.capture_scheduler_stages = fake_capture
        with patch.dict(sys.modules, {"fx_visualizer.capture": capture_module}):
            artifacts = visualize_graph(gm=graph, example_inputs=inputs, output_dir=destination, config={"epilogue_fusion": False}, triton_matmul=False)
        saved = json.loads((destination / "compiler" / "capture.json").read_text())
        self.assertEqual(saved["name"], "my_graph")
        self.assertEqual(saved["original_graph"]["name"], "my_graph")
        self.assertEqual(json.loads(Path(artifacts["capture"]).read_text()), saved)
        reloaded = render_capture(capture=destination / "compiler" / "capture.json", output_dir=destination / "reload")
        self.assertEqual(reloaded["name"], "my_graph")


class TestMatmulComparison(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        self.template = capture_fixture()
        nodes = {node["id"]: node for node in self.template["original_graph"]["nodes"]}
        nodes["activation"]["actual_args"] = {"kind": "tuple", "items": [{"ref": "input"}, {"ref": "input"}]}
        nodes["activation"]["actual_kwargs"] = {"kind": "dict", "items": [[["str", "alpha"], {"literal": ["int", 1]}]]}
        nodes["input"]["geometry"] = {
            "kind": "tensor", "shape": [16, 16], "stride": [16, 1],
            "dtype": "torch.bfloat16", "device": "cuda:0", "storage_offset": 0,
        }
        self.template["compile"] = {"status": "success", "options": {"triton.native_matmul": False}}
        self.native = copy.deepcopy(self.template)
        self.native["name"] = "synthetic_shared_view_native"
        self.native["compile"]["options"]["triton.native_matmul"] = True
        self.native["stages"]["fusion"]["scheduler_nodes"] = [
            scheduler_group("native-merged", ["arithmetic-leaf", "template-a", "cast-a", "template-b"], ["activation", "left_mm", "cast", "right_mm"], ["shared_view"]),
            scheduler_group("native-unknown", ["tail-no-source"], unknown=["unmapped"]),
        ]
        self.snapshot = (copy.deepcopy(self.template), copy.deepcopy(self.native))

    def test_comparison_preserves_all_embedded_sources_and_uses_independent_dom_ids(self):
        previous_ids = set()
        for index in range(2):
            with self.subTest(render=index):
                captures = {"template": self.template, "native": self.native}
                before = copy.deepcopy(captures)
                artifacts = render_matmul_comparison(
                    template_capture=self.template, native_capture=self.native, output_dir=Path(self.output.name) / str(index),
                )
                self.assertEqual(set(artifacts["variants"]), set(captures))
                widget = Path(artifacts["widget"]).read_text()
                markup = WidgetHTML(widget)
                self.assertEqual(sum("data-fx-payload" in attrs for _, attrs in markup.elements), 1)
                embedded = decode_widget_viewers(widget)
                self.assertEqual(len(embedded), len(captures))
                mode_buttons = [attrs for tag, attrs in markup.elements if tag == "button" and "data-matmul-mode" in attrs]
                self.assertEqual([attrs["data-matmul-mode"] for attrs in mode_buttons], list(captures))
                for attrs in mode_buttons:
                    self.assertEqual(attrs["type"], "button")
                    self.assertEqual(attrs["aria-pressed"], str(attrs["data-matmul-mode"] == "template").lower())
                panels = [attrs["data-matmul-panel"] for _, attrs in markup.elements if "data-matmul-panel" in attrs]
                self.assertEqual(panels, list(captures))
                self.assertIn("addEventListener('click'", widget)
                ids = [attrs["id"] for _, attrs in markup.elements if "id" in attrs]
                for (mode, capture), decoded in zip(captures.items(), embedded):
                    data = decoded["data"]
                    self.assertCountEqual([node["id"] for node in data["nodes"]], [node["id"] for node in capture["original_graph"]["nodes"]])
                    actual_edges = [{key: edge[key] for key in ("source", "target", "ports")} for edge in data["edges"]]
                    self.assertCountEqual(actual_edges, capture["original_graph"]["edges"])
                    variant = artifacts["variants"][mode]
                    standalone_widget = Path(variant["widget"]).read_text()
                    standalone = decode_widget_viewers(standalone_widget)[0]
                    self.assertEqual({key: decoded[key] for key in ("svg", "data")}, standalone)
                    standalone_id = next(attrs["id"] for tag, attrs in WidgetHTML(standalone_widget).elements if tag == "section")
                    self.assertEqual(decoded["root_id"], standalone_id)
                    payload = json.loads(Path(variant["payload"]).read_text())
                    for field in ("lowering_groups", "fusion_groups"):
                        self.assertEqual(
                            {group["id"]: group["member_fx_ids"] for group in data[field]},
                            {group["id"]: group["member_fx_ids"] for group in payload[field]},
                        )
                    svg = ET.fromstring(decoded["svg"])
                    ids.extend(element.attrib["id"] for element in svg.iter() if "id" in element.attrib)
                    self.assertEqual(json.loads(Path(variant["capture"]).read_text()), capture)
                self.assertEqual(len(ids), len(set(ids)))
                self.assertFalse(previous_ids.intersection(ids))
                previous_ids.update(ids)
                self.assertEqual(captures, before)

    def test_comparison_keeps_source_geometry_and_camera_fixed_across_different_groups(self):
        native = copy.deepcopy(self.native)
        native["stages"]["lowering"]["scheduler_nodes"] = [
            scheduler_group("native-prepare", ["arithmetic-leaf"], ["activation"]),
            scheduler_group("native-left", ["template-a", "cast-a"], ["left_mm", "cast"], ["shared_view"]),
            scheduler_group("native-right", ["template-b"], ["right_mm"], ["shared_view"]),
            scheduler_group("native-unknown", ["tail-no-source"], unknown=["unmapped"]),
        ]
        before = copy.deepcopy((self.template, native))
        artifacts = render_matmul_comparison(template_capture=self.template, native_capture=native, output_dir=Path(self.output.name) / "fixed_geometry")
        payloads = [json.loads(Path(variant["payload"]).read_text()) for variant in artifacts["variants"].values()]
        self.assertEqual([len(payload["lowering_groups"]) for payload in payloads], [5, 4])
        self.assertEqual([variant["after_fusion_count"] for variant in artifacts["variants"].values()], [4, 2])
        reference = payloads[0]
        positions = {node["id"]: (node["bbox"], node["center"]) for node in reference["nodes"]}
        routes = [{key: edge[key] for key in ("source", "target", "paths", "arrow_polygons")} for edge in reference["edges"]]
        svg_routes = None
        for mode, payload in zip(artifacts["variants"], payloads):
            with self.subTest(mode=mode):
                self.assertEqual({node["id"]: (node["bbox"], node["center"]) for node in payload["nodes"]}, positions)
                self.assertEqual([{key: edge[key] for key in ("source", "target", "paths", "arrow_polygons")} for edge in payload["edges"]], routes)
                self.assertEqual(payload["view_box"], reference["view_box"])
                svg = ET.fromstring(Path(artifacts["variants"][mode]["svg"]).read_text())
                self.assertEqual(list(map(float, svg.attrib["viewBox"].split())), reference["view_box"])
                actual_routes = [
                    ([path.attrib["d"] for path in element.findall("{http://www.w3.org/2000/svg}path")], [arrow.attrib["points"] for arrow in element.findall("{http://www.w3.org/2000/svg}polygon")])
                    for element in svg.iter() if element.attrib.get("class") == "fx-edge"
                ]
                if svg_routes is None:
                    svg_routes = actual_routes
                self.assertEqual(actual_routes, svg_routes)
        widget = Path(artifacts["widget"]).read_text()
        for decoded in decode_widget_viewers(widget):
            self.assertEqual(decoded["data"]["view_box"], reference["view_box"])
            self.assertEqual({node["id"]: node["bbox"] for node in decoded["data"]["nodes"]}, {node_id: geometry[0] for node_id, geometry in positions.items()})
            self.assertEqual(list(map(float, ET.fromstring(decoded["svg"]).attrib["viewBox"].split())), reference["view_box"])
        self.assertEqual((self.template, native), before)

    def test_different_fx_ports_arguments_and_input_geometry_are_rejected(self):
        mutations = (
            ("edge argument port", ("edges", 0, "ports"), [["argument", "other"], ["argument", "self"]]),
            ("scalar literal", ("nodes", 1, "actual_kwargs", "items", 0, 1, "literal", 1), 2),
            ("argument reference", ("nodes", 1, "actual_args", "items", 0, "ref"), "shared_view"),
            ("stride", ("nodes", 0, "geometry", "stride"), [1, 16]),
            ("storage offset", ("nodes", 0, "geometry", "storage_offset"), 16),
            ("operation", ("nodes", 3, "target"), "aten.bmm.default"),
        )
        for index, (label, path, value) in enumerate(mutations):
            with self.subTest(difference=label):
                native = copy.deepcopy(self.native)
                parent = native["original_graph"]
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
                before = copy.deepcopy(native)
                destination = Path(self.output.name) / f"mismatch_{index}"
                with self.assertRaises(ValueError):
                    render_matmul_comparison(template_capture=self.template, native_capture=native, output_dir=destination)
                self.assertFalse(destination.exists(), "Reject incompatible captures before creating comparison artifacts")
                self.assertEqual(native, before)
                self.assertEqual(self.template, self.snapshot[0])

    def test_multi_matmul_report_counts_actual_leaves_instead_of_cast_continuations(self):
        capture = copy.deepcopy(self.template)
        for node in capture["original_graph"]["nodes"]:
            if node["id"] in ("left_mm", "right_mm"):
                node["is_matmul"] = True
        capture["matmul_lowering"] = [
            {"fx_id": "left_mm", "native_dot_reduction_leaves": ["template-a"]},
            {"fx_id": "right_mm", "native_dot_reduction_leaves": ["template-b"]},
        ]
        post_nodes = capture["stages"]["fusion"]["scheduler_nodes"]
        continuation = post_nodes[1]
        continuation["execution_fx_ids"].append("right_mm")
        before = copy.deepcopy(capture)
        artifacts = render_capture(capture=capture, output_dir=Path(self.output.name) / "one_actual_mm")
        self.assertEqual(artifacts["multi_matmul_source_schedulers"], [])
        self.assertEqual(capture, before)

        continuation["member_operation_names"].append("template-b")
        del post_nodes[2]
        before = copy.deepcopy(capture)
        artifacts = render_capture(capture=capture, output_dir=Path(self.output.name) / "two_actual_mm")
        self.assertEqual(artifacts["multi_matmul_source_schedulers"], [
            {"id": continuation["id"], "matmul_fx_ids": ["left_mm", "right_mm"], "certainty": "observed matmul leaves"},
        ])
        self.assertEqual(capture, before)

    def test_live_api_defaults_to_two_schedule_only_modes_and_preserves_config(self):
        graph = object()
        inputs = [object(), [16, 16]]
        config = {"emulate_precision_casts": True, "epilogue_fusion": False}
        original_config = copy.deepcopy(config)
        calls = []

        def fake_capture(actual_graph, actual_inputs, output_dir, **kwargs):
            self.assertIs(actual_graph, graph)
            self.assertIs(actual_inputs, inputs)
            calls.append(copy.deepcopy(kwargs))
            source = self.native if kwargs["config"]["triton.native_matmul"] else self.template
            capture = copy.deepcopy(source)
            output_dir.mkdir(parents=True)
            capture["capture_path"] = str(output_dir / "capture.json")
            Path(capture["capture_path"]).write_text(json.dumps(capture))
            return capture

        capture_module = types.ModuleType("fx_visualizer.capture")
        capture_module.capture_scheduler_stages = fake_capture
        with patch.dict(sys.modules, {"fx_visualizer.capture": capture_module}):
            artifacts = visualize_matmul_comparison(gm=graph, example_inputs=inputs, output_dir=Path(self.output.name) / "live", config=config)
        self.assertEqual(calls, [
            {"config": {**config, "triton.native_matmul": False}, "triton_matmul": True, "schedule_only": True},
            {"config": {**config, "triton.native_matmul": True}, "triton_matmul": True, "schedule_only": True},
        ])
        self.assertEqual(config, original_config)
        self.assertTrue(Path(artifacts["widget"]).is_file())

    def test_explicit_native_policy_is_rejected_before_either_capture(self):
        capture_module = types.ModuleType("fx_visualizer.capture")
        capture_module.capture_scheduler_stages = Mock()
        for native in (False, True):
            with self.subTest(native=native), patch.dict(sys.modules, {"fx_visualizer.capture": capture_module}):
                config = {"triton.native_matmul": native, "emulate_precision_casts": True}
                before = copy.deepcopy(config)
                with self.assertRaises(ValueError):
                    visualize_matmul_comparison(gm=object(), example_inputs=[object()], output_dir=Path(self.output.name) / str(native), config=config)
                self.assertEqual(config, before)
        capture_module.capture_scheduler_stages.assert_not_called()

    def test_single_live_api_always_stops_after_scheduler_fusion(self):
        capture_module = types.ModuleType("fx_visualizer.capture")
        calls = []

        def fake_capture(graph, inputs, output_dir, **kwargs):
            calls.append(kwargs)
            output_dir.mkdir(parents=True)
            capture = copy.deepcopy(self.template)
            capture["capture_path"] = str(output_dir / "capture.json")
            Path(capture["capture_path"]).write_text(json.dumps(capture))
            return capture

        capture_module.capture_scheduler_stages = fake_capture
        with patch.dict(sys.modules, {"fx_visualizer.capture": capture_module}):
            visualize_graph(gm=object(), example_inputs=[object()], output_dir=Path(self.output.name) / "schedule", triton_matmul=True)
        self.assertEqual(calls, [
            {"config": None, "triton_matmul": True, "schedule_only": True},
        ])


class TestGenericGraphAPI(unittest.TestCase):
    def setUp(self):
        self.capture_module = importlib.import_module("fx_visualizer.capture")
        importlib.import_module("torch.fx.passes.fake_tensor_prop")
        self.capture = Mock(side_effect=capture_cpu_graph_fixture)

    def test_non_matmul_scheduler_api_keeps_nested_arguments_and_storage(self):
        import torch

        class NonMatmul(torch.nn.Module):
            def forward(self, value, other):
                total = torch.ops.aten.add.Tensor(value, other[0])
                return {"value": total, "reduction": (torch.ops.aten.sum.dim_IntList(total, [1]), other[1])}

        gm = torch.fx.symbolic_trace(NonMatmul())
        code = gm.code
        base = torch.arange(72, dtype=torch.float32).reshape(8, 9)
        view = base[1:5:2, 2:7]
        example_inputs = [view, [view, 3]]
        blocked = {name: None for name in ("oracle_harness", "repro_harness", "input_codec", "numerics")}
        with tempfile.TemporaryDirectory() as output, patch.dict(sys.modules, blocked), patch.object(self.capture_module, "capture_scheduler_stages", self.capture):
            artifacts = visualize_graph(gm=gm, example_inputs=example_inputs, output_dir=Path(output) / "generic")
            self.assertIs(self.capture.call_args.args[0], gm)
            self.assertIs(self.capture.call_args.args[1], example_inputs)
            self.assertEqual(self.capture.call_args.kwargs, {"config": None, "triton_matmul": False, "schedule_only": True})
            payload = json.loads(Path(artifacts["payload"]).read_text())
            capture = json.loads(Path(artifacts["capture"]).read_text())
            self.assertEqual(gm.code, code)
            self.assertEqual(len(payload["nodes"]), len(list(gm.graph.nodes)))
            self.assertEqual(artifacts["lowering_count"], 1)
            self.assertEqual(artifacts["capture_mode"], "schedule_only")
            self.assertFalse(any(node["is_matmul"] for node in payload["nodes"]))
            inputs = [node for node in payload["nodes"] if node["fx_op"] == "placeholder"]
            self.assertEqual(inputs[0]["geometry"]["stride"], [18, 1])
            self.assertEqual(inputs[0]["geometry"]["storage_offset"], 11)
            self.assertEqual(inputs[1]["geometry"]["items"][0], inputs[0]["geometry"])
            aliases = capture["compile"]["input_aliases"]
            self.assertEqual(aliases[0]["alias_group"], aliases[1]["alias_group"])
            returned = [node for node in payload["nodes"] if node["fx_op"] == "output"][0]
            self.assertEqual(len(returned["output_ports"]), 3)
            self.assertEqual(returned["actual_args"]["items"][0]["kind"], "dict")
            decoded = decode_widget_viewers(Path(artifacts["widget"]).read_text())[0]["data"]
            self.assertEqual(decoded["nodes"][-1]["output_ports"], returned["output_ports"])
            with patch.dict(sys.modules, {"torch": None}):
                replay = render_capture(capture=artifacts["capture"], output_dir=Path(output) / "offline")
            self.assertEqual(json.loads(Path(replay["capture"]).read_text()), capture)

    @unittest.skipUnless(BB_AVAILABLE, "Optional Better Benchmark checkout is not on PYTHONPATH")
    def test_non_matmul_bb_repro_preserves_saved_geometry_and_output_arguments(self):
        import torch

        source = '''import torch
from repro_harness import make_inputs_from_config
_repro_version = 3
class Repro(torch.nn.Module):
    def forward(self, value):
        one = torch.ops.aten.full.default([2, 3], 1.0, dtype=torch.float32, device=torch.device("cpu"))
        total = torch.ops.aten.add.Tensor(value, one)
        mean = torch.ops.aten.mean.dim(value, [1], False)
        view = torch.ops.aten.permute.default(value, [1, 0])
        return {"statistics": (total, mean), "view": view}
def make_inputs(config):
    return make_inputs_from_config(config)
'''
        shapes = {"points": [{"shape_hash": "saved", "inputs": [[[2, 3], "f32", {"st": [9, 1], "off": 7, "dev": "cpu"}]]}]}
        blocked = {name: None for name in ("matmul_capture", "matmul_dedup", "matmul_shape_cleanup", "matmul_similarity")}
        with tempfile.TemporaryDirectory() as output:
            directory = Path(output) / "non_matmul"
            directory.mkdir()
            (directory / "repro.py").write_text(source)
            (directory / "shapes.json").write_text(json.dumps(shapes))
            with patch.dict(sys.modules, blocked), patch.object(self.capture_module, "capture_scheduler_stages", self.capture), patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("Unexpected CUDA initialization")):
                artifacts = visualize_repro(repro_path=directory / "repro.py", output_dir=Path(output) / "viewer", configuration="saved")
            self.assertTrue(self.capture.call_args.kwargs["schedule_only"])
            payload = json.loads(Path(artifacts["payload"]).read_text())
            inputs = [node for node in payload["nodes"] if node["fx_op"] == "placeholder"]
            self.assertEqual(inputs[0]["geometry"]["device"], "cpu")
            self.assertEqual(inputs[0]["geometry"]["stride"], [9, 1])
            self.assertEqual(inputs[0]["geometry"]["storage_offset"], 7)
            self.assertTrue(any(node["target"] == "aten.full.default" for node in payload["nodes"]))
            self.assertFalse(any(node["is_matmul"] for node in payload["nodes"]))
            self.assertEqual(artifacts["capture_mode"], "schedule_only")
            capture = json.loads(Path(artifacts["capture"]).read_text())
            self.assertFalse(capture["compile"]["execution_succeeded"])
            returned = [node for node in payload["nodes"] if node["fx_op"] == "output"][0]
            self.assertEqual(len(returned["output_ports"]), 3)
            self.assertTrue(all(port["geometry"] is not None for port in returned["output_ports"]))
            outputs = {port["fx_name"]: port["geometry"] for port in returned["output_ports"]}
            self.assertEqual(outputs["mean_dim"]["shape"], [2])
            self.assertEqual(outputs["mean_dim"]["stride"], [1])
            self.assertEqual(outputs["mean_dim"]["storage_offset"], 0)
            self.assertEqual(outputs["permute_default"]["shape"], [3, 2])
            self.assertEqual(outputs["permute_default"]["stride"], [1, 9])
            self.assertEqual(outputs["permute_default"]["storage_offset"], 7)
            self.assertEqual(outputs["permute_default"]["layout"], "torch.strided")
            self.assertFalse(outputs["permute_default"]["is_conj"])
            self.assertFalse(outputs["permute_default"]["is_neg"])

    def test_metadata_preflight_skips_mutation_rng_and_python_calls(self):
        import operator
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from fx_visualizer.tracing import populate_metadata

        for target in (torch.ops.aten.add_.Tensor, torch.ops.aten.rand_like.default, operator.neg,
                       torch.ops.prims.inductor_random.default, torch.ops.prims.inductor_randint.default):
            with self.subTest(target=target):
                graph = torch.fx.Graph()
                value = graph.placeholder("value")
                args = (value, value) if target == torch.ops.aten.add_.Tensor else (value,)
                result = graph.call_function(target, args)
                graph.output(result)
                gm = torch.fx.GraphModule(torch.nn.Module(), graph)
                with FakeTensorMode(allow_fallback_kernels=False) as mode, patch("torch.fx.passes.fake_tensor_prop.FakeTensorProp.propagate_dont_convert_inputs") as propagate:
                    report = populate_metadata(gm, [torch.empty(2)], mode)
                self.assertEqual(report["status"], "partial")
                self.assertEqual(report["skipped_nodes"], [result.name])
                propagate.assert_not_called()

    @unittest.skipUnless(BB_AVAILABLE, "Optional Better Benchmark checkout is not on PYTHONPATH")
    def test_nested_saved_inputs_keep_aliases_offsets_and_flags_without_shared_factory_changes(self):
        import torch

        source = '''import torch
from repro_harness import make_inputs_from_config
class Repro(torch.nn.Module):
    def forward(self, bundle, shape):
        total = torch.ops.aten.add.Tensor(bundle[0], bundle[1][0])
        view = torch.ops.aten.view.default(total, shape)
        return {"view": view, "scalar": bundle[1][1], "other": bundle[2]}
def make_inputs(config):
    return make_inputs_from_config(config)
'''
        entries = [
            ["tuple", [
                [[2, 3], "c64", {"st": [7, 1], "off": 2, "alias": 0, "dev": "cpu", "x": {"is_conj": True}}],
                ["list", [
                    [[2, 3], "c64", {"st": [7, 1], "off": 2, "alias": 0, "dev": "cpu"}],
                    ["sc", 3],
                ]],
                [[2, 3], "f32", {"st": [9, 1], "off": 7, "dev": "cpu", "x": {"requires_grad": True, "is_neg": True}}],
            ]],
            ["S", [6], {"x": {"container": "tuple"}}],
        ]
        shapes = {"points": [{"shape_hash": "nested", "inputs": entries, "alias_group_nbytes": [96]}]}
        blocked = {name: None for name in ("matmul_capture", "matmul_dedup", "matmul_shape_cleanup", "matmul_similarity")}
        with tempfile.TemporaryDirectory() as output:
            directory = Path(output) / "nested"
            directory.mkdir()
            (directory / "repro.py").write_text(source)
            (directory / "shapes.json").write_text(json.dumps(shapes))
            with patch.dict(sys.modules, blocked), patch.object(self.capture_module, "capture_scheduler_stages", self.capture):
                artifacts = visualize_repro(repro_path=directory, output_dir=Path(output) / "viewer", configuration="nested")
            self.assertTrue(self.capture.call_args.kwargs["schedule_only"])
            capture = json.loads(Path(artifacts["capture"]).read_text())
            geometries = capture["compile"]["inputs"]
            self.assertEqual(geometries[0]["kind"], "tuple")
            first, nested, standalone = geometries[0]["items"]
            self.assertEqual(nested["kind"], "list")
            self.assertEqual((first["storage_offset"], first["stride"], first["is_conj"]), (2, [7, 1], True))
            self.assertEqual((standalone["storage_offset"], standalone["stride"]), (7, [9, 1]))
            self.assertTrue(standalone["is_neg"])
            self.assertTrue(standalone["requires_grad"])
            self.assertEqual(geometries[1]["kind"], "tuple")
            aliases = capture["compile"]["input_aliases"]
            self.assertEqual(aliases[0]["alias_group"], aliases[1]["alias_group"])
            self.assertNotEqual(aliases[0]["alias_group"], aliases[2]["alias_group"])
            self.assertEqual(capture["compile"]["input_storage_groups"][0]["nbytes"], 96)
            self.assertFalse(capture["compile"]["execution_succeeded"])

    @unittest.skipUnless(BB_AVAILABLE, "Optional Better Benchmark checkout is not on PYTHONPATH")
    def test_real_input_adaptation_preserves_factory_values_and_nested_storage(self):
        import torch
        from fx_visualizer.integrations._bb_inputs import make_repro_inputs
        from repro_harness import make_inputs_from_config

        standalone = {"kind": "tensor", "shape": [2, 3], "stride": [9, 1], "storage_offset": 7,
                      "dtype": "float32", "device": "cpu", "requires_grad": True}
        shared = {"kind": "tensor", "shape": [2, 3], "stride": [6, 1], "storage_offset": 3,
                  "dtype": "float32", "device": "cpu", "alias_group": 0}
        config = {"inputs": [{"kind": "tuple", "items": [standalone, {"kind": "list", "items": [shared, shared]}]},
                             {"kind": "shape", "dims": [6], "container": "tuple"}], "alias_group_nbytes": [64]}
        original = copy.deepcopy(config)
        sampled = []

        def factory(point):
            values = make_inputs_from_config(point)
            sampled.extend(value.clone() for value in values if isinstance(value, torch.Tensor))
            return values

        inputs = make_repro_inputs(config, factory)
        value, aliases = inputs[0]
        self.assertIsInstance(inputs[0], tuple)
        self.assertIsInstance(aliases, list)
        self.assertEqual(inputs[1], (6,))
        self.assertEqual((value.storage_offset(), value.stride()), (7, (9, 1)))
        self.assertTrue(value.requires_grad)
        self.assertEqual(aliases[0].untyped_storage()._cdata, aliases[1].untyped_storage()._cdata)
        self.assertEqual(aliases[0].untyped_storage().nbytes(), 64)
        self.assertEqual(aliases[0].storage_offset(), 3)
        for tensor, expected in zip([value, *aliases], sampled):
            torch.testing.assert_close(tensor, expected)
        actual = torch.ops.aten.view.default(torch.ops.aten.add.Tensor(value, aliases[0]), inputs[1])
        torch.testing.assert_close(actual, (sampled[0] + sampled[1]).reshape(6))
        self.assertEqual(config, original)


class TestReproAPI(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        self.point = {"compile_policy": {"inductor": {"emulate_precision_casts": True}}}
        self.inputs, self.graph = [object()], object()
        self.make_inputs = Mock(return_value=self.inputs)
        self.visualize = Mock(return_value={"name": "example"})
        self.comparison = Mock(return_value={"name": "comparison"})
        self.dependencies = {name: types.ModuleType(name) for name in ("oracle_harness", "repro_harness", "input_codec")}
        self.dependencies["oracle_harness"].load_repro_module = Mock(return_value=types.SimpleNamespace(Repro=object, make_inputs=self.make_inputs))
        self.dependencies["repro_harness"].load_shape_configs = Mock(return_value={"point": self.point})
        self.dependencies["repro_harness"].compile_policy_from_config = lambda point: point.get("compile_policy", {})

    def run_api(self, *, comparison=False, **kwargs):
        with patch.dict(sys.modules, self.dependencies), patch("fx_visualizer.integrations.better_benchmark.trace_module", return_value=self.graph), patch("fx_visualizer.integrations.better_benchmark.visualize_graph", self.visualize), patch("fx_visualizer.integrations.better_benchmark.visualize_matmul_comparison", self.comparison):
            api = visualize_repro_matmul_comparison if comparison else visualize_repro
            return api(repro_path=self.output.name, output_dir=Path(self.output.name) / "viewer", **kwargs)

    def test_saved_precision_and_input_setting_reach_scheduler_capture(self):
        self.run_api(config={"epilogue_fusion": False})
        self.make_inputs.assert_called_once_with(self.point)
        self.assertEqual(self.visualize.call_args.args, ())
        self.assertIs(self.visualize.call_args.kwargs["gm"], self.graph)
        self.assertIs(self.visualize.call_args.kwargs["example_inputs"], self.inputs)
        self.assertEqual(self.visualize.call_args.kwargs["config"], {"emulate_precision_casts": True, "epilogue_fusion": False})
        self.assertNotIn("mode", self.visualize.call_args.kwargs)
        self.assertNotIn("schedule_only", self.visualize.call_args.kwargs)
        self.comparison.assert_not_called()

    def test_saved_policy_conflict_fails_before_input_creation(self):
        for comparison in (False, True):
            with self.subTest(comparison=comparison), self.assertRaises(ValueError):
                self.run_api(comparison=comparison, config={"emulate_precision_casts": False})
        self.make_inputs.assert_not_called()
        self.visualize.assert_not_called()
        self.comparison.assert_not_called()

    def test_comparison_reuses_one_saved_input_setting_and_compile_policy(self):
        self.run_api(comparison=True, config={"epilogue_fusion": False})
        self.make_inputs.assert_called_once_with(self.point)
        self.comparison.assert_called_once()
        self.assertEqual(self.comparison.call_args.args, ())
        self.assertIs(self.comparison.call_args.kwargs["gm"], self.graph)
        self.assertIs(self.comparison.call_args.kwargs["example_inputs"], self.inputs)
        self.assertEqual(self.comparison.call_args.kwargs["config"], {"emulate_precision_casts": True, "epilogue_fusion": False})
        self.visualize.assert_not_called()

    def test_comparison_native_conflict_fails_before_input_creation(self):
        for key in ("triton.native_matmul", "triton.native-matmul"):
            for native in (False, True):
                self.point["compile_policy"]["inductor"][key] = native
                with self.subTest(key=key, native=native, source="saved"), self.assertRaises(ValueError):
                    self.run_api(comparison=True)
                del self.point["compile_policy"]["inductor"][key]
                with self.subTest(key=key, native=native, source="explicit"), self.assertRaises(ValueError):
                    self.run_api(comparison=True, config={key: native})
        self.make_inputs.assert_not_called()
        self.visualize.assert_not_called()
        self.comparison.assert_not_called()


class TestStandaloneImports(unittest.TestCase):
    def test_default_import_and_cli_help_need_neither_torch_nor_better_benchmark(self):
        code = """import sys
for name in ('torch', 'oracle_harness', 'repro_harness', 'input_codec', 'numerics'):
    sys.modules[name] = None
import fx_visualizer
from fx_visualizer.cli import main
if hasattr(fx_visualizer, 'visualize_repro'):
    raise AssertionError('Better Benchmark adapters must be explicit imports')
main(['--help'])
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("usage: fx-visualize", result.stdout)
        self.assertIn("--capture", result.stdout)
        self.assertIn("--bb-root", result.stdout)

    def test_adapter_missing_dependency_has_an_actionable_error(self):
        before = sys.path[:]
        blocked = {name: None for name in ("oracle_harness", "repro_harness", "input_codec")}
        with patch.dict(sys.modules, blocked), self.assertRaisesRegex(ImportError, "bb_root=.*better-benchmark"):
            visualize_repro(repro_path="repros/example", output_dir="unused")
        self.assertEqual(sys.path, before)

    def test_external_checkout_resolves_repro_and_restores_path_on_success_and_error(self):
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            for name in ("oracle_harness", "repro_harness", "input_codec"):
                (root / (name + ".py")).write_text("")
            directory = root / "repros" / "example"
            directory.mkdir(parents=True)
            before = sys.path[:]
            modules = {name: types.ModuleType(name) for name in ("oracle_harness", "repro_harness", "input_codec")}
            for name, module in modules.items():
                module.__file__ = str(root / (name + ".py"))
            modules["oracle_harness"].load_repro_module = Mock(return_value=types.SimpleNamespace(Repro=object, make_inputs=lambda config: []))
            modules["repro_harness"].compile_policy_from_config = lambda config: {}
            modules["repro_harness"].load_shape_configs = Mock(return_value={"saved": {}})

            def trace(module):
                self.assertEqual(sys.path[0], str(root))
                return object()

            with patch.dict(sys.modules, {**modules, "numerics": None}), patch("fx_visualizer.integrations.better_benchmark.trace_module", side_effect=trace), patch("fx_visualizer.integrations.better_benchmark.visualize_graph", return_value={"html": "example.html"}):
                result = visualize_repro(repro_path="repros/example", bb_root=root, output_dir=root / "viewer")
                self.assertEqual(result["html"], "example.html")
                modules["oracle_harness"].load_repro_module.assert_called_once_with(directory)
                self.assertEqual(sys.path, before)
                with self.assertRaises(ValueError):
                    visualize_repro(repro_path="repros/example", bb_root=root, output_dir=root / "viewer", configuration="missing")
                self.assertEqual(sys.path, before)

    def test_external_checkout_rejects_a_cached_helper_from_another_repository(self):
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            for name in ("oracle_harness", "repro_harness", "input_codec"):
                (root / (name + ".py")).write_text("")
            cached = types.ModuleType("oracle_harness")
            cached.__file__ = str(root / "different_checkout" / "oracle_harness.py")
            before = sys.path[:]
            with patch.dict(sys.modules, {"oracle_harness": cached}), self.assertRaisesRegex(ImportError, "fresh Python process"):
                visualize_repro(repro_path="repros/example", bb_root=root, output_dir=root / "viewer")
            self.assertEqual(sys.path, before)


class TestPublicAPIContract(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        destination = Path(self.output.name) / "viewer"
        self.live_apis = [
            (visualize_graph, {"gm": object(), "example_inputs": [], "output_dir": destination}),
            (visualize_repro, {"repro_path": self.output.name, "output_dir": destination}),
            (visualize_matmul_comparison, {"gm": object(), "example_inputs": [], "output_dir": destination}),
            (visualize_repro_matmul_comparison, {"repro_path": self.output.name, "output_dir": destination}),
        ]
        capture = capture_fixture()
        self.saved_apis = [
            (render_capture, {"capture": capture, "output_dir": destination}),
            (render_matmul_comparison, {"template_capture": capture, "native_capture": capture, "output_dir": destination}),
            (write_index, {"artifacts": [], "output_dir": destination}),
        ]

    def test_all_public_api_arguments_are_keyword_only(self):
        for api, kwargs in self.live_apis + self.saved_apis:
            with self.subTest(api=api.__name__):
                for parameter in inspect.signature(api).parameters.values():
                    self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                with self.assertRaises(TypeError):
                    api(*kwargs.values())
                self.assertFalse(kwargs["output_dir"].exists())

    def test_removed_capture_options_are_rejected_by_every_live_api(self):
        for api, kwargs in self.live_apis:
            for option in ({"mode": "scheduler"}, {"schedule_only": True}, {"schedule_only": False}):
                with self.subTest(api=api.__name__, option=option), self.assertRaises(TypeError):
                    api(**kwargs, **option)
                self.assertFalse(kwargs["output_dir"].exists())


class TestVisualizerCLI(unittest.TestCase):
    def setUp(self):
        from fx_visualizer import cli

        self.cli = cli
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)

    def test_scheduler_and_comparison_forward_configuration_and_precision_options(self):
        for comparison in (False, True):
            with self.subTest(comparison=comparison), patch("fx_visualizer.integrations.better_benchmark.visualize_repro", return_value={"name": "example"}) as single, patch("fx_visualizer.integrations.better_benchmark.visualize_repro_matmul_comparison", return_value={"name": "comparison"}) as compare, patch.object(self.cli, "write_index", return_value=Path(self.output.name) / "index.html") as index, redirect_stdout(io.StringIO()):
                options = ["--repro", self.output.name, "--bb-root", self.output.name, "--output-dir", self.output.name, "--configuration", "saved", "--inductor-config", "emulate-precision-casts=true"]
                if comparison:
                    options.append("--compare-matmul")
                self.cli.main(options)
                visualize, unused = (compare, single) if comparison else (single, compare)
                visualize.assert_called_once()
                unused.assert_not_called()
                self.assertEqual(visualize.call_args.args, ())
                self.assertEqual(visualize.call_args.kwargs["repro_path"], Path(self.output.name))
                self.assertEqual(visualize.call_args.kwargs["output_dir"], Path(self.output.name))
                self.assertEqual(visualize.call_args.kwargs["configuration"], "saved")
                self.assertEqual(visualize.call_args.kwargs["bb_root"], Path(self.output.name))
                self.assertNotIn("mode", visualize.call_args.kwargs)
                self.assertNotIn("schedule_only", visualize.call_args.kwargs)
                self.assertEqual(visualize.call_args.kwargs["config"], {"emulate_precision_casts": True})
                self.assertEqual(index.call_args.args, ())
                self.assertEqual(index.call_args.kwargs["artifacts"], [visualize.return_value])

    def test_removed_compile_flags_are_rejected(self):
        for flag in (["--mode", "scheduler"], ["--full-compile"]):
            with self.subTest(flag=flag), patch("fx_visualizer.integrations.better_benchmark.visualize_repro") as visualize, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.cli.main(["--repro", self.output.name, "--output-dir", self.output.name, *flag])
                self.assertEqual(raised.exception.code, 2)
                visualize.assert_not_called()

    def test_saved_comparison_requires_exactly_two_captures_and_keeps_mode_order(self):
        root = Path(self.output.name)
        paths = [root / f"{mode}.json" for mode in ("template", "native")]
        for count in (1, 2, 3):
            with self.subTest(captures=count), patch.object(self.cli, "render_matmul_comparison", return_value={"name": "saved"}) as render, patch.object(self.cli, "write_index", return_value=root / "index.html"), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                sources = paths[:count] if count <= 2 else paths + [paths[0]]
                options = ["--compare-matmul", "--output-dir", self.output.name]
                for path in sources:
                    options.extend(["--capture", str(path)])
                if count != 2:
                    with self.assertRaises(SystemExit) as raised:
                        self.cli.main(options)
                    self.assertEqual(raised.exception.code, 2)
                    render.assert_not_called()
                else:
                    self.cli.main(options)
                    self.assertEqual(render.call_args.args, ())
                    self.assertEqual(render.call_args.kwargs["template_capture"], sources[0])
                    self.assertEqual(render.call_args.kwargs["native_capture"], sources[1])
