import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fx_visualizer.execution import annotate_scheduler_execution
from fx_visualizer.rendering import render_capture


_GRAPH_ID = "lowering-1"


def fx_ref(name):
    return {"name": name, "graph_id": _GRAPH_ID}


def loop_operation(name, target, *args):
    return {"name": name, "op": "call_method", "target": target, "args": [{"name": "ops"}, *args]}


def split_mean_capture():
    definitions = [
        ("x", "placeholder", "x", []),
        ("relu", "call_function", "aten.relu.default", ["x"]),
        ("mean", "call_function", "aten.mean.dim", ["relu"]),
        ("compare", "call_function", "aten.le.Scalar", ["relu"]),
        ("output", "output", "output", ["mean", "compare"]),
    ]
    original = [
        {"id": f"n{index}", "fx_name": name, "fx_op": op, "target": target, "is_matmul": False}
        for index, (name, op, target, _) in enumerate(definitions)
    ]
    source_ids = {node["fx_name"]: node["id"] for node in original}
    transformed = [
        {"name": name, "op": op, "target": target, "origin_fx_ids": [source_ids[name]],
         "args": [fx_ref(parent) for parent in parents], "kwargs": {}}
        for name, op, target, parents in definitions
    ]
    values = [
        {"fx": fx_ref(name), "value": {
            "ir_chain": [{"type": ir_type, "origin_node": fx_ref(name)}],
            "terminal_ir_type": ir_type, "buffer_names": buffers, "contains_view": False,
        }}
        for name, ir_type, buffers in (
            ("relu", "Pointwise", []),
            ("mean", "ComputedBuffer", ["buf1"]),
            ("compare", "ComputedBuffer", ["buf2"]),
        )
    ]
    bodies = {
        "op0": [
            loop_operation("load", "load", "arg0_1", 0),
            loop_operation("relu", "relu", {"name": "load"}),
            loop_operation("cast", "to_dtype", {"name": "relu"}, "float32"),
            loop_operation("sum", "reduction", "float32", "float32", "sum", {"name": "cast"}),
            loop_operation("store", "store_reduction", "buf0", 0, {"name": "sum"}),
        ],
        "op1": [
            loop_operation("load", "load", "buf0", 0),
            loop_operation("div", "truediv", {"name": "load"}, 169.0),
            loop_operation("cast", "to_dtype", {"name": "div"}, "bfloat16"),
            loop_operation("store", "store", "buf1", 0, {"name": "cast"}),
        ],
        "op2": [
            loop_operation("load", "load", "arg0_1", 0),
            loop_operation("relu", "relu", {"name": "load"}),
            loop_operation("compare", "le", {"name": "relu"}, 0),
            loop_operation("store", "store", "buf2", 0, {"name": "compare"}),
        ],
    }
    lowering = []
    for name, root, buffer, read in (
        ("op0", "mean", "buf0", "arg0_1"),
        ("op1", "mean", "buf1", "buf0"),
        ("op2", "compare", "buf2", "arg0_1"),
    ):
        leaf = {
            "name": name, "buffer_names": [buffer], "reads": [{"name": read}],
            "origin_node": fx_ref(root), "data_origins": [fx_ref(root)],
            "origin_fx_ids": [source_ids["relu"], source_ids[root]],
            "loop_ir": {"blocks": [{"nodes": bodies[name]}]},
            "ir_type": "Reduction" if name == "op0" else "Pointwise",
            "reduction_type": "sum" if name == "op0" else None,
            "is_template": False, "is_extern": False,
        }
        lowering.append({"id": name, "member_operation_names": [name], "leaf_members": [leaf]})
    fusion = [{
        "id": "fused", "member_operation_names": ["op2", "op0", "op1"],
        "leaf_members": [copy.deepcopy(node["leaf_members"][0]) for node in lowering],
    }]
    return {
        "name": "split_mean", "compile": {"mode": "schedule_only"},
        "original_graph": {"nodes": original},
        "stages": {
            "lowering": {"graph_id": _GRAPH_ID, "transformed_fx": transformed,
                         "final_ir_values": values, "scheduler_nodes": lowering,
                         "nodes": [{"id": node["id"]} for node in lowering]},
            "fusion": {"scheduler_nodes": fusion, "nodes": [{"id": "fused"}]},
        },
    }


def native_bias_cast_capture():
    capture = split_mean_capture()
    stage = capture["stages"]["lowering"]
    capture["original_graph"]["nodes"][2].update(target="aten.addmm.default", is_matmul=True)
    capture["original_graph"]["nodes"].extend([
        {"id": "n5", "fx_name": "bias", "fx_op": "placeholder", "target": "bias", "is_matmul": False},
        {"id": "n6", "fx_name": "bias_cast", "fx_op": "call_function", "target": "prims.convert_element_type.default", "is_matmul": False},
    ])
    stage["transformed_fx"][2].update(target="aten.addmm.default", args=[fx_ref("bias_cast"), fx_ref("relu"), fx_ref("x")])
    stage["transformed_fx"].extend([
        {"name": "bias", "op": "placeholder", "target": "bias", "origin_fx_ids": ["n5"], "args": [], "kwargs": {}},
        {"name": "bias_cast", "op": "call_function", "target": "prims.convert_element_type.default", "origin_fx_ids": ["n6"], "args": [fx_ref("bias"), "torch.bfloat16"], "kwargs": {}},
    ])
    stage["final_ir_values"].extend([
        {"fx": fx_ref(name), "value": {
            "ir_chain": [{"type": kind, "origin_node": fx_ref(name)}],
            "terminal_ir_type": kind, "buffer_names": buffers, "contains_view": False,
        }} for name, kind, buffers in (("bias", "InputBuffer", ["arg1_1"]), ("bias_cast", "Pointwise", []))
    ])
    producer, consumer, _ = stage["scheduler_nodes"]
    producer["leaf_members"][0].update(is_native_matmul=True, reduction_type="dot")
    producer["leaf_members"][0]["loop_ir"]["blocks"][0]["nodes"] = [
        loop_operation("load", "load", "arg0_1", 0),
        loop_operation("relu", "relu", {"name": "load"}),
        loop_operation("dot", "dot", {"name": "relu"}, {"name": "load"}),
        loop_operation("reduce", "reduction", "float32", "float32", "dot", {"name": "dot"}),
        loop_operation("store", "store_reduction", "buf0", 0, {"name": "reduce"}),
    ]
    leaf = consumer["leaf_members"][0]
    leaf["reads"].append({"name": "arg1_1"})
    leaf["loop_ir"]["blocks"][0]["nodes"] = [
        loop_operation("load", "load", "buf0", 0),
        loop_operation("bias_load", "load", "arg1_1", 0),
        loop_operation("bias_cast", "to_dtype", {"name": "bias_load"}, "torch.bfloat16"),
        loop_operation("add", "add", {"name": "load"}, {"name": "bias_cast"}),
        loop_operation("store", "store", "buf1", 0, {"name": "add"}),
    ]
    capture["stages"]["fusion"]["scheduler_nodes"][0]["leaf_members"] = [copy.deepcopy(node["leaf_members"][0]) for node in stage["scheduler_nodes"]]
    return capture


class TestSchedulerExecution(unittest.TestCase):
    def test_mean_split_preserves_shared_mean_without_repeating_relu(self):
        capture = split_mean_capture()
        original = copy.deepcopy(capture["original_graph"])
        values = copy.deepcopy(capture["stages"]["lowering"]["final_ir_values"])
        self.assertIs(annotate_scheduler_execution(capture), capture)
        op0, op1, op2 = capture["stages"]["lowering"]["scheduler_nodes"]
        self.assertEqual(op0["execution_fx_ids"], ["n1", "n2"])
        self.assertEqual(op1["execution_fx_ids"], ["n2"])
        self.assertEqual(op2["execution_fx_ids"], ["n1", "n3"])
        self.assertEqual(capture["fx_status"]["n2"]["lowering"], ["op0", "op1"])
        self.assertEqual(capture["fx_status"]["n1"]["lowering"], ["op0", "op2"])
        evidence = op1["leaf_members"][0]["execution_evidence"]
        self.assertEqual(evidence["visited_fx_names"], ["mean"])
        self.assertEqual(evidence["materialized_input_cuts"], [{
            "fx": "mean", "buffer": "buf0", "owner": ["op0"],
            "matched_read_dependency": True, "kind": "same_fx_split_stage",
        }])
        self.assertEqual(evidence["certainty"], "observed IR boundaries")
        fused = capture["stages"]["fusion"]["scheduler_nodes"][0]
        self.assertEqual(fused["execution_fx_ids"], ["n1", "n2", "n3"])
        self.assertEqual(fused["execution_evidence"]["actual_member_operation_names"], ["op2", "op0", "op1"])
        self.assertEqual(capture["original_graph"], original)
        self.assertEqual(capture["stages"]["lowering"]["final_ir_values"], values)

    def test_sibling_roots_need_both_strong_dependency_and_actual_load(self):
        for case in ("no_dependency", "weak_dependency", "missing_loop", "different_load"):
            with self.subTest(case=case):
                capture = split_mean_capture()
                op1 = capture["stages"]["lowering"]["scheduler_nodes"][1]
                leaf = op1["leaf_members"][0]
                if case == "no_dependency":
                    leaf["reads"] = []
                elif case == "weak_dependency":
                    leaf["reads"][0]["weak"] = True
                elif case == "missing_loop":
                    leaf["loop_ir"]["blocks"] = []
                else:
                    leaf["loop_ir"]["blocks"][0]["nodes"][0]["args"][1] = "arg0_1"
                annotate_scheduler_execution(capture)
                self.assertEqual(op1["execution_fx_ids"], ["n1", "n2"])
                self.assertEqual(leaf["execution_evidence"]["materialized_input_cuts"], [])

    def test_shared_source_ids_do_not_replace_exact_transformed_root_identity(self):
        capture = split_mean_capture()
        stage = capture["stages"]["lowering"]
        other_mean = copy.deepcopy(stage["transformed_fx"][2])
        other_mean["name"] = "other_mean"
        stage["transformed_fx"].insert(-1, other_mean)
        other_value = copy.deepcopy(stage["final_ir_values"][1])
        other_value["fx"] = fx_ref("other_mean")
        other_value["value"]["ir_chain"][0]["origin_node"] = fx_ref("other_mean")
        other_value["value"]["buffer_names"] = ["buf0"]
        stage["final_ir_values"].append(other_value)
        producer = stage["scheduler_nodes"][0]["leaf_members"][0]
        producer["origin_node"] = fx_ref("other_mean")
        producer["data_origins"] = [fx_ref("other_mean")]
        annotate_scheduler_execution(capture)
        consumer = stage["scheduler_nodes"][1]
        self.assertEqual(consumer["execution_fx_ids"], ["n1", "n2"])
        self.assertEqual(consumer["leaf_members"][0]["execution_evidence"]["materialized_input_cuts"], [])

    def test_ambiguous_buffer_ownership_does_not_establish_a_split(self):
        capture = split_mean_capture()
        stage = capture["stages"]["lowering"]
        other_owner = stage["scheduler_nodes"][2]["leaf_members"][0]
        other_owner["buffer_names"].append("buf0")
        other_owner["origin_node"] = fx_ref("mean")
        other_owner["data_origins"] = [fx_ref("mean")]
        annotate_scheduler_execution(capture)
        consumer = stage["scheduler_nodes"][1]
        self.assertEqual(consumer["execution_fx_ids"], ["n1", "n2"])
        self.assertEqual(consumer["leaf_members"][0]["execution_evidence"]["materialized_input_cuts"], [])

    def test_foreign_stage_roots_do_not_establish_a_split_by_name(self):
        for source in ("origin_node", "data_origins"):
            with self.subTest(source=source):
                capture = split_mean_capture()
                stage = capture["stages"]["lowering"]
                producer = stage["scheduler_nodes"][0]["leaf_members"][0]
                foreign = {"name": "mean", "graph_id": "another-stage"}
                producer["origin_node"] = foreign if source == "origin_node" else None
                producer["data_origins"] = [] if source == "origin_node" else [foreign]
                annotate_scheduler_execution(capture)
                self.assertEqual(stage["scheduler_nodes"][0]["execution_fx_ids"], [])
                self.assertEqual(producer["execution_evidence"]["actual_root_fx_names"], [])
                consumer = stage["scheduler_nodes"][1]
                self.assertEqual(consumer["execution_fx_ids"], ["n1", "n2"])
                self.assertEqual(consumer["leaf_members"][0]["execution_evidence"]["materialized_input_cuts"], [])

    def test_split_with_other_inputs_marks_upstream_membership_partial(self):
        for declared in (True, False):
            with self.subTest(raw_input_declared=declared):
                capture = split_mean_capture()
                consumer = capture["stages"]["lowering"]["scheduler_nodes"][1]
                leaf = consumer["leaf_members"][0]
                if declared:
                    leaf["reads"].append({"name": "arg0_1"})
                operations = leaf["loop_ir"]["blocks"][0]["nodes"]
                operations[1:1] = [
                    loop_operation("raw_load", "load", "arg0_1", 0),
                    loop_operation("raw_relu", "relu", {"name": "raw_load"}),
                ]
                operations.insert(-2, loop_operation("add", "add", {"name": "div"}, {"name": "raw_relu"}))
                operations[-2]["args"][1] = {"name": "add"}
                annotate_scheduler_execution(capture)
                evidence = leaf["execution_evidence"]
                self.assertEqual(consumer["execution_fx_ids"], ["n2"])
                self.assertEqual(evidence["unknown_fx_ids"], ["n1"])
                self.assertEqual(evidence["certainty"], "partial")
                self.assertEqual(evidence["uncertainties"][0]["kind"], "split_stage_with_other_inputs")
                self.assertEqual(evidence["uncertainties"][0]["buffers"], ["arg0_1"])
                self.assertEqual(capture["fx_status"]["n1"]["uncertain_membership"]["lowering"], ["op1"])

    def test_native_dot_finalization_does_not_count_a_second_matmul(self):
        capture = split_mean_capture()
        stage = capture["stages"]["lowering"]
        capture["original_graph"]["nodes"][2].update(target="aten.mm.default", is_matmul=True)
        stage["transformed_fx"][2].update(target="aten.mm.default", args=[fx_ref("relu"), fx_ref("x")])
        producer = stage["scheduler_nodes"][0]["leaf_members"][0]
        producer.update(is_native_matmul=True, reduction_type="dot")
        producer["loop_ir"]["blocks"][0]["nodes"] = [
            loop_operation("load", "load", "arg0_1", 0),
            loop_operation("relu", "relu", {"name": "load"}),
            loop_operation("dot", "dot", {"name": "relu"}, {"name": "load"}),
            loop_operation("reduce", "reduction", "float32", "float32", "dot", {"name": "dot"}),
            loop_operation("store", "store_reduction", "buf0", 0, {"name": "reduce"}),
        ]
        annotate_scheduler_execution(capture)
        op0, op1, _ = stage["scheduler_nodes"]
        self.assertEqual(op0["native_original_mm_ids"], ["n2"])
        self.assertEqual(op0["native_matmul_leaf_count"], 1)
        self.assertEqual(op1["execution_fx_ids"], [])
        self.assertEqual(op1["native_matmul_leaf_count"], 0)
        evidence = op1["leaf_members"][0]["execution_evidence"]
        self.assertEqual(evidence["unknown_fx_ids"], ["n2"])
        self.assertEqual(evidence["uncertainties"][0]["kind"], "matmul_result_conversion_after_materialized_reduction")
        fused = capture["stages"]["fusion"]["scheduler_nodes"][0]
        self.assertEqual(fused["matmul_leaf_count"], 1)
        self.assertEqual(fused["native_original_mm_ids"], ["n2"])

    def test_native_bias_cast_belongs_to_epilogue_and_not_dot_reduction(self):
        capture = native_bias_cast_capture()
        annotate_scheduler_execution(capture)
        producer, epilogue, _ = capture["stages"]["lowering"]["scheduler_nodes"]
        self.assertNotIn("n6", producer["execution_fx_ids"])
        self.assertEqual(epilogue["execution_fx_ids"], ["n6"])
        self.assertEqual(epilogue["native_matmul_leaf_count"], 0)
        self.assertEqual(capture["fx_status"]["n6"]["lowering"], ["op1"])
        proof = capture["fx_status"]["n6"]["inlined_cast_proofs"][0]
        self.assertEqual(proof["actual_fx_name"], "bias_cast")
        self.assertEqual(proof["input_buffer"], "arg1_1")
        self.assertEqual(proof["target_dtype"], "torch.bfloat16")
        self.assertEqual(proof["output_buffers"], ["buf1"])
        fused = capture["stages"]["fusion"]["scheduler_nodes"][0]
        self.assertIn("n6", fused["execution_fx_ids"])
        self.assertEqual(fused["matmul_leaf_count"], 1)
        before = copy.deepcopy(capture)
        annotate_scheduler_execution(capture)
        self.assertEqual(capture, before)

    def test_inlined_cast_requires_exact_input_dtype_origin_use_and_store_path(self):
        for case in ("no_conversion", "different_dtype", "different_buffer", "weak_read", "unused_conversion", "foreign_ir_origin", "materialized_cast", "unrelated_fx_root", "foreign_loop_reference", "missing_input_binding"):
            with self.subTest(case=case):
                capture = native_bias_cast_capture()
                stage = capture["stages"]["lowering"]
                leaf = stage["scheduler_nodes"][1]["leaf_members"][0]
                operations = leaf["loop_ir"]["blocks"][0]["nodes"]
                if case == "no_conversion":
                    operations[2]["target"] = "neg"
                elif case == "different_dtype":
                    operations[2]["args"][-1] = "torch.float32"
                elif case == "different_buffer":
                    operations[1]["args"][1] = "other_buffer"
                elif case == "weak_read":
                    leaf["reads"][-1]["weak"] = True
                elif case == "unused_conversion":
                    operations[3]["args"][-1] = {"name": "load"}
                elif case == "foreign_ir_origin":
                    stage["final_ir_values"][-1]["value"]["ir_chain"][-1]["origin_node"]["graph_id"] = "other-stage"
                elif case == "materialized_cast":
                    stage["final_ir_values"][-1]["value"]["buffer_names"] = ["cast_buffer"]
                elif case == "unrelated_fx_root":
                    leaf["origin_node"] = fx_ref("compare")
                    leaf["data_origins"] = [fx_ref("compare")]
                elif case == "foreign_loop_reference":
                    operations[2]["args"][1]["graph_id"] = "other-loop"
                else:
                    stage["final_ir_values"][-2]["value"] = None
                annotate_scheduler_execution(capture)
                self.assertNotIn("n6", stage["scheduler_nodes"][1]["execution_fx_ids"])
                self.assertEqual(leaf["execution_evidence"]["inlined_cast_proofs"], [])

    def test_offline_replay_refreshes_stale_membership_outside_scheduler_preview(self):
        capture = split_mean_capture()
        capture["compile"]["mode"] = "compiled"
        capture["execution_annotation"] = {"version": 2}
        for node in capture["stages"]["lowering"]["scheduler_nodes"]:
            node.update(execution_fx_ids=node["leaf_members"][0]["origin_fx_ids"], indexing_fx_ids=[])
        capture["stages"]["fusion"]["scheduler_nodes"][0].update(execution_fx_ids=["n1", "n2", "n3"], indexing_fx_ids=[])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "old_capture.json"
            source.write_text(json.dumps(capture))
            with patch("fx_visualizer.regions.layout_regions", return_value={
                "lowering_groups": [], "fusion_groups": [], "nodes": [],
            }) as layout, patch("fx_visualizer.rendering._build_widget", return_value="<section>Viewer</section>"):
                result = render_capture(capture=source, output_dir=Path(temporary) / "replay")
            replay = json.loads(Path(result["capture"]).read_text())
            self.assertEqual(replay["execution_annotation"]["version"], 4)
            self.assertEqual(replay["stages"]["lowering"]["scheduler_nodes"][1]["execution_fx_ids"], ["n2"])
            self.assertEqual(layout.call_args.args[0]["stages"]["lowering"]["scheduler_nodes"][1]["execution_fx_ids"], ["n2"])
            self.assertEqual(json.loads(source.read_text()), capture)


if __name__ == "__main__":
    unittest.main()
