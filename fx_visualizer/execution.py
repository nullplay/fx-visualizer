"""Annotate actual scheduler captures without equating ancestry with execution.

Membership comes from captured live IR bindings, LoopBody FX operations and
transformed FX objects. Generated source is retained as an artifact, never parsed.
"""

from __future__ import annotations

from collections import defaultdict


_VIEW_SOURCE_NAMES = {
    "view", "_unsafe_view", "permute", "expand", "squeeze", "unsqueeze",
    "alias", "detach", "slice", "select", "transpose", "t", "as_strided",
}


def _source_may_be_indexing(node):
    # AOT can insert a view carrying an arithmetic producer's source history.
    target = node.get("target", "")
    if target == "<built-in function getitem>":
        return True
    if node.get("fx_op") == "call_method":
        return target in _VIEW_SOURCE_NAMES | {"reshape", "flatten"}
    parts = target.split(".")
    return len(parts) > 1 and parts[0] in {"aten", "prims"} and parts[1] in _VIEW_SOURCE_NAMES | {"reshape", "flatten"}


def ordered_ids(values):
    return sorted(set(values), key=lambda value: (int(value[1:]) if value.startswith("n") and value[1:].isdigit() else 10**12, value))


def ir_routes(value):
    if not value:
        return []
    if "ir_chain" in value:
        return [value]
    children = value.get("values", ())
    if isinstance(children, dict):
        children = children.values()
    return [route for child in children for route in ir_routes(child)]


def loop_nodes(leaf):
    return [node for block in leaf.get("loop_ir", {}).get("blocks", []) for node in block["nodes"]]


def loop_args(node):
    args = node.get("args", [])
    if node.get("op") == "call_method" and args and isinstance(args[0], dict) and args[0].get("name") == "ops":
        return args[1:]
    return args


def scheduler_backend_evidence(capture):
    compile_info = capture.get("compile", {})
    schedule_only = compile_info.get("mode") == "schedule_only"
    return {
        "method": "actual scheduler leaf types and selected caller objects",
        "status": "not_generated" if schedule_only else "scheduler_observed",
        "output_code_paths": compile_info.get("artifacts", {}).get("output_code", []),
        "selected_runtime_backend_proven": False,
        "reason": "Scheduler evidence does not establish runtime kernel launches; generated source is retained without parsing.",
    }


class _ExecutionAnnotator:
    def __init__(self, capture):
        self.capture = capture
        self.stage = capture["stages"]["lowering"]
        self.graph_id = self.stage["graph_id"]
        self.fx = {node["name"]: node for node in self.stage["transformed_fx"]}
        self.values = {entry["fx"]["name"]: entry for entry in self.stage.get("final_ir_values", [])}
        self.original = {node["id"]: node for node in capture["original_graph"]["nodes"] if not node.get("synthetic")}
        self.mm_ids = {key for key, node in self.original.items() if node.get("is_matmul")}
        self.parents = {name: self.references([node.get("args"), node.get("kwargs")]) for name, node in self.fx.items()}
        self.leaves = {leaf["name"]: leaf for node in self.stage["scheduler_nodes"] for leaf in node["leaf_members"]}
        self.buffer_owners = defaultdict(set)
        self.buffer_roots = defaultdict(set)
        for name, leaf in self.leaves.items():
            roots, _ = self.leaf_roots(leaf)
            for buffer in leaf["buffer_names"]:
                self.buffer_owners[buffer].add(name)
                self.buffer_roots[buffer].update(roots)
        self.indexing_names = {name for name in self.fx if self.is_indexing(name)}
        self.original_indexing = {key for name in self.indexing_names for key in self.fx[name].get("origin_fx_ids", [])
                                  if key in self.original and not self.original[key].get("is_matmul")
                                  and _source_may_be_indexing(self.original[key])}
        self.pure_indexing = set(self.original_indexing)
        for key in self.original_indexing:
            target = self.original[key].get("target", "").split(".")
            operation = target[1] if len(target) > 1 else target[0]
            if operation in {"reshape", "flatten"}:
                self.pure_indexing.discard(key)
        self.native_materialized_buffers = {}
        for leaf in self.leaves.values():
            roots, _ = self.leaf_roots(leaf)
            if self.native_matmul_proofs(leaf, roots):
                for name in roots:
                    if self.source_ids([name]) & self.mm_ids:
                        self.native_materialized_buffers.setdefault(name, set()).update(leaf["buffer_names"])
        self.annotations = {}

    def references(self, value):
        if isinstance(value, dict):
            if value.get("graph_id") == self.graph_id and value.get("name") in self.fx:
                return [value["name"]]
            return list(dict.fromkeys(name for item in value.values() for name in self.references(item)))
        if isinstance(value, (list, tuple)):
            return list(dict.fromkeys(name for item in value for name in self.references(item)))
        return []

    def is_indexing(self, name):
        value = self.values.get(name, {}).get("value")
        routes = ir_routes(value)
        if routes and all(route.get("contains_view") for route in routes):
            return True
        node = self.fx[name]
        parents = self.parents[name]
        if node.get("is_getitem") or node.get("target") == "<built-in function getitem>":
            if parents and self.values.get(parents[0], {}).get("value", {}).get("type") in {"tuple", "list", "dict"}:
                return True
        identities = {route.get("ir_object_id") for route in routes} - {None}
        return bool(identities) and any(
            identities == {route.get("ir_object_id") for route in ir_routes(self.values.get(parent, {}).get("value"))}
            for parent in parents
        )

    def source_ids(self, names, indexing=False):
        ids = {key for name in names for key in self.fx.get(name, {}).get("origin_fx_ids", []) if key in self.original}
        return ids & self.original_indexing if indexing else ids - self.pure_indexing

    def direct_lazy_proofs(self, leaf):
        root = leaf.get("data_origin_node") or leaf.get("origin_node") or {}
        node = self.fx.get(root.get("name"), {})
        parents = self.parents.get(node.get("name"), [])
        if len(parents) != 1:
            return []
        root_target = node.get("target", "").split(".")
        if len(root_target) < 2 or root_target[0] not in ("aten", "prims"):
            return []
        parent = parents[0]
        lazy = self.fx[parent]
        lazy_target = lazy.get("target", "").split(".")
        if len(lazy_target) < 2 or lazy_target[0] not in ("aten", "prims"):
            return []
        args = lazy.get("args", [])
        if len(args) != 2 or not isinstance(args[1], (int, float)) or lazy.get("kwargs", {}).get("alpha", 1) != 1:
            return []
        routes = ir_routes(self.values.get(parent, {}).get("value"))
        if len(routes) != 1 or routes[0].get("terminal_ir_type") != "Pointwise" or routes[0].get("buffer_names"):
            return []
        terminal = routes[0]["ir_chain"][-1]
        if terminal.get("origin_node", {}).get("name") != parent or terminal.get("origin_node", {}).get("graph_id") != self.graph_id:
            return []
        proofs = []
        for block in leaf.get("loop_ir", {}).get("blocks", []):
            operations = {operation["name"]: operation for operation in block["nodes"]}
            def resolve(value):
                return operations.get(value.get("name"), {}) if isinstance(value, dict) else {}
            for store in operations.values():
                stored = loop_args(store)
                if store.get("target") != "store" or len(stored) < 3 or stored[0] not in leaf["buffer_names"]:
                    continue
                actual_root = resolve(stored[2])
                root_args = loop_args(actual_root)
                if actual_root.get("target") != root_target[1] or len(root_args) != 1:
                    continue
                actual_lazy = resolve(root_args[0])
                if actual_lazy.get("target") != lazy_target[1]:
                    continue
                constants = [resolve(value) for value in loop_args(actual_lazy)]
                if not any(value.get("target") == "constant" and loop_args(value)[:1] == [args[1]] for value in constants):
                    continue
                ids = {key for key in self.source_ids([parent]) if self.original[key].get("target") == lazy["target"]}
                if ids:
                    proofs.append({"method": "exact direct FX edge + retained lazy Pointwise origin + captured LoopBody dataflow",
                                   "execution_fx_ids": ordered_ids(ids), "actual_root_fx_name": node["name"],
                                   "actual_lazy_fx_name": parent, "actual_lazy_fx_args": args,
                                   "actual_ir_origin_node": terminal["origin_node"], "output_buffer": stored[0],
                                   "loop_operations": [actual_lazy, actual_root, store],
                                   "upstream_split_buffer_boundary_warning_preserved": True})
        return proofs

    def annotate_leaf(self, name):
        leaf = self.leaves[name]
        own_buffers = set(leaf["buffer_names"])
        read_buffers = {dep["name"] for dep in leaf["reads"] if not dep.get("weak")}
        loaded_buffers = {args[0] for node in loop_nodes(leaf) if node.get("target") == "load"
                          and (args := loop_args(node)) and isinstance(args[0], str)}
        body_origins = leaf.get("data_origins", [])
        roots, root_method = self.leaf_roots(leaf)
        matmul_proofs = self.native_matmul_proofs(leaf, roots)
        matmul_roots = {name for proof in matmul_proofs for name in proof["actual_root_fx_names"]}
        if matmul_roots:
            roots = sorted(matmul_roots)
            root_method = "direct matmul roots proven by actual loop product/reduction IR"
        visited, indexing, unknown_names = set(), set(), set()
        cuts, issues = [], []
        direct_ids = self.source_ids(roots)

        def walk(fx_name, is_root=False):
            if fx_name in visited or fx_name in indexing:
                return
            node = self.fx[fx_name]
            if node["op"] not in {"call_function", "call_method", "call_module"}:
                return
            entry = self.values.get(fx_name)
            if not entry or entry.get("value") is None:
                unknown_names.add(fx_name)
                issues.append({"kind": "missing_final_ir_binding", "fx": fx_name})
                return
            routes = ir_routes(entry["value"])
            buffers = {buffer for route in routes for buffer in route.get("buffer_names", [])}
            matmul_buffers = self.native_materialized_buffers.get(fx_name, set())
            buffers |= matmul_buffers
            external = buffers - own_buffers
            metadata = self.is_indexing(fx_name)
            if metadata:
                indexing.add(fx_name)
                for parent in self.parents[fx_name]:
                    if self.fx[parent]["op"] == "call_function" and self.is_indexing(parent):
                        walk(parent)
                if external and not is_root:
                    for buffer in sorted(external):
                        cuts.append({"fx": fx_name, "buffer": buffer, "owner": sorted(self.buffer_owners.get(buffer, [])), "matched_read_dependency": buffer in read_buffers})
                    if external - read_buffers:
                        issues.append({"kind": "metadata_materialized_cut_not_a_read", "fx": fx_name, "buffers": sorted(external - read_buffers)})
                    return
                # A tuple getter may be the exact origin_node assigned to a
                # reduction output. Its body belongs to the producing tuple op.
                for parent in self.parents[fx_name]:
                    walk(parent, is_root=is_root and entry["value"].get("type") not in {"tuple", "list", "dict"})
                return
            if external and (not is_root or matmul_buffers - own_buffers):
                for buffer in sorted(external):
                    cuts.append({"fx": fx_name, "buffer": buffer, "owner": sorted(self.buffer_owners.get(buffer, [])), "matched_read_dependency": buffer in read_buffers})
                if external - read_buffers:
                    issues.append({"kind": "materialized_cut_not_a_read", "fx": fx_name, "buffers": sorted(external - read_buffers)})
                if is_root and matmul_buffers - own_buffers:
                    unknown_names.add(fx_name)
                    issues.append({"kind": "matmul_result_conversion_after_materialized_reduction", "fx": fx_name,
                                   "reduction_buffers": sorted(matmul_buffers), "note": "This stage uses an already materialized matmul reduction result; it does not establish a new matmul computation."})
                return
            if not routes and entry["value"].get("type") not in {"tuple", "list", "dict"}:
                return
            visited.add(fx_name)
            # One FX operation can have materialized stages absent from its final binding.
            split_inputs = {buffer for buffer in (read_buffers & loaded_buffers) - own_buffers
                            if self.buffer_roots.get(buffer) == {fx_name} and len(self.buffer_owners[buffer]) == 1}
            if split_inputs:
                for buffer in sorted(split_inputs):
                    cuts.append({"fx": fx_name, "buffer": buffer, "owner": sorted(self.buffer_owners[buffer]),
                                 "matched_read_dependency": True, "kind": "same_fx_split_stage"})
                other_inputs = (read_buffers | loaded_buffers) - own_buffers - split_inputs
                if not other_inputs:
                    return
                issues.append({"kind": "split_stage_with_other_inputs", "fx": fx_name, "buffers": sorted(other_inputs),
                               "note": "Other inputs may still carry upstream lazy computations; ancestry needs direct body evidence."})
            parents = self.parents[fx_name]
            if fx_name in matmul_roots and node["target"] in {"aten.addmm.default", "aten.baddbmm.default"}:
                parents = self.references(node.get("args", [])[1:3])
            for parent in parents:
                walk(parent)

        for fx_name in roots:
            walk(fx_name, is_root=True)
        executed = self.source_ids(visited)
        indexed = self.source_ids(indexing, indexing=True)
        unknown = self.source_ids(unknown_names)
        if not roots:
            issues.append({"kind": "no_precise_body_root"})
            unknown |= set(leaf["origin_fx_ids"]) - self.original_indexing
        if issues:
            unknown |= executed - direct_ids
            executed &= direct_ids
        native_proofs = self.native_matmul_proofs(leaf, roots)
        lowered_mm = {key for proof in native_proofs for key in proof["execution_fx_ids"]}
        unproven_mm = (executed & self.mm_ids) - lowered_mm
        if not leaf.get("is_template") and not leaf.get("is_extern") and unproven_mm:
            unknown |= unproven_mm
            executed -= unproven_mm
            issues.append({"kind": "matmul_provenance_on_non_template_body", "note": "Matmul provenance alone does not prove a dot computation in this body."})
        if leaf.get("scheduler_type") == "NopKernelSchedulerNode":
            executed.clear()
        direct_body_proofs = self.direct_lazy_proofs(leaf)
        direct_body_ids = {key for proof in direct_body_proofs for key in proof["execution_fx_ids"]}
        executed |= direct_body_ids
        unknown -= direct_body_ids
        evidence = {"method": "walk actual transformed FX from actual loop/template roots; stop at other named materialized buffers",
                    "root_method": root_method, "actual_root_fx_names": roots,
                    "actual_loop_body_available": bool(leaf.get("loop_body")),
                    "actual_data_origins_available": bool(body_origins),
                    "visited_fx_names": sorted(visited), "indexing_fx_names": sorted(indexing),
                    "materialized_input_cuts": cuts, "read_dependency_names": sorted(read_buffers),
                    "own_buffer_names": sorted(own_buffers), "direct_root_fx_ids": ordered_ids(direct_ids),
                    "unknown_fx_ids": ordered_ids(unknown), "uncertainties": issues,
                    "direct_lazy_body_proofs": direct_body_proofs,
                    "native_dot_reduction_proofs": native_proofs,
                    "excluded_ancestry_fx_ids": ordered_ids(set(leaf["origin_fx_ids"]) - executed - indexed),
                    "certainty": "partial" if issues else "observed IR boundaries",
                    "limitation": "Membership describes lowered computations. Indexing and container selection are separate; CSE/cast simplification can remove individual instructions. Shared lazy operations may occur in more than one kernel."}
        template_root = leaf.get("origin_node") or {}
        template_ids = ordered_ids(set(template_root.get("origin_fx_ids", [])) & self.mm_ids) if leaf.get("is_template") else []
        template_count = int(bool(leaf.get("is_template")))
        native_ids = ordered_ids(key for proof in native_proofs for key in proof["execution_fx_ids"])
        native_count = int(bool(native_proofs))
        evidence.update({"template_leaf_count": template_count, "template_original_mm_ids": template_ids,
                         "template_selected_backend": leaf.get("backend") if template_count else None,
                         "native_matmul_leaf_count": native_count, "native_original_mm_ids": native_ids})
        result = {"execution_fx_ids": ordered_ids(executed), "indexing_fx_ids": ordered_ids(indexed),
                  "template_leaf_count": template_count, "template_original_mm_ids": template_ids,
                  "native_matmul_leaf_count": native_count, "native_original_mm_ids": native_ids,
                  "execution_evidence": evidence}
        leaf.update(result)
        self.annotations[name] = result
        return result

    def leaf_roots(self, leaf):
        root = leaf.get("data_origin_node") or leaf.get("origin_node")
        if root and root.get("graph_id") == self.graph_id and root.get("name") in self.fx:
            return [root["name"]], "actual data/origin_node"
        candidates = {node["name"] for node in leaf.get("data_origins", [])
                      if node.get("graph_id") == self.graph_id and node["name"] in self.fx}
        used = {parent for node in candidates for parent in self.parents[node] if parent in candidates}
        return sorted(candidates - used), "sinks of actual loop data_origins; internal split stage"

    def native_matmul_proofs(self, leaf, roots):
        if not leaf.get("is_native_matmul") or leaf.get("reduction_type") != "dot":
            return []
        matmul_roots = [name for name in roots if self.source_ids([name]) & self.mm_ids]
        ids = self.source_ids(matmul_roots) & self.mm_ids
        if not ids:
            return []
        operations = loop_nodes(leaf)
        products = [node for node in operations if node.get("op") == "call_method" and node.get("target") == "dot"]
        reductions = [node for node in operations if node.get("op") == "call_method" and node.get("target") == "reduction"
                      and len(loop_args(node)) > 2 and loop_args(node)[2] == "dot"]
        if not products or not reductions:
            return []
        return [{"method": "actual direct FX root + ComputedBuffer dot reduction + captured LoopBody FX operations",
                 "kind": "native_dot_reduction", "execution_fx_ids": ordered_ids(ids), "actual_root_fx_names": matmul_roots,
                 "ir_type": leaf["ir_type"], "ir_reduction_type": leaf["reduction_type"],
                 "scheduler_is_native_matmul": True,
                 "loop_products": products, "loop_reductions": reductions,
                 "generated_kernel_or_runtime_proven": False}]

    def annotate(self):
        for name in self.leaves:
            self.annotate_leaf(name)
        for stage_name in ("lowering", "fusion"):
            stage = self.capture["stages"][stage_name]
            updated = {}
            for node in stage["scheduler_nodes"]:
                names = node["member_operation_names"]
                members = [self.annotations[name] for name in names if name in self.annotations]
                missing = [name for name in names if name not in self.annotations]
                template_count = sum(member["template_leaf_count"] for member in members)
                template_ids = ordered_ids(key for member in members for key in member["template_original_mm_ids"])
                native_count = sum(member["native_matmul_leaf_count"] for member in members)
                native_ids = ordered_ids(key for member in members for key in member["native_original_mm_ids"])
                fields = {"execution_fx_ids": ordered_ids(key for member in members for key in member["execution_fx_ids"]),
                          "indexing_fx_ids": ordered_ids(key for member in members for key in member["indexing_fx_ids"]),
                          "template_leaf_count": template_count, "template_original_mm_ids": template_ids,
                          "native_matmul_leaf_count": native_count, "native_original_mm_ids": native_ids,
                          "matmul_leaf_count": template_count + native_count,
                          "matmul_original_mm_ids": ordered_ids(template_ids + native_ids),
                          "execution_evidence": {"method": "union of actual pre-fusion leaf memberships", "actual_member_operation_names": names,
                                                 "template_leaf_count": template_count, "template_original_mm_ids": template_ids,
                                                 "native_matmul_leaf_count": native_count, "native_original_mm_ids": native_ids,
                                                 "member_evidence": {name: self.annotations[name]["execution_evidence"] for name in names if name in self.annotations},
                                                 "unknown_fx_ids": ordered_ids(key for member in members for key in member["execution_evidence"]["unknown_fx_ids"]),
                                                 "missing_members": missing, "certainty": "partial" if missing or any(member["execution_evidence"]["uncertainties"] for member in members) else "observed IR boundaries"}}
                node.update(fields)
                for leaf in node["leaf_members"]:
                    if leaf["name"] in self.annotations:
                        leaf.update(self.annotations[leaf["name"]])
                updated[node["id"]] = fields
            for node in stage["nodes"]:
                node.update(updated.get(node["id"], {"execution_fx_ids": [], "indexing_fx_ids": [], "template_leaf_count": 0,
                                                    "template_original_mm_ids": [], "execution_evidence": {"method": "boundary; no GPU computation", "template_leaf_count": 0, "template_original_mm_ids": []}}))
        status = {}
        for key, original in self.original.items():
            lowered = [node["id"] for node in self.capture["stages"]["lowering"]["scheduler_nodes"] if key in node["execution_fx_ids"]]
            fused = [node["id"] for node in self.capture["stages"]["fusion"]["scheduler_nodes"] if key in node["execution_fx_ids"]]
            indexed = {stage: [node["id"] for node in self.capture["stages"][stage]["scheduler_nodes"] if key in node["indexing_fx_ids"]] for stage in ("lowering", "fusion")}
            uncertain = {stage: [node["id"] for node in self.capture["stages"][stage]["scheduler_nodes"] if key in node["execution_evidence"]["unknown_fx_ids"]] for stage in ("lowering", "fusion")}
            if original["fx_op"] == "placeholder":
                role, note = "input", "Original boundary input; shape containers may be specialized or flattened by AOT."
            elif original["fx_op"] == "output":
                role, note = "output", "Original output pytree; this is not a scheduler computation."
            elif original["fx_op"] == "get_attr":
                role, note = "constant", "Original module attribute; this is a scheduler boundary."
            elif key in self.original_indexing and lowered:
                role, note = "compute_and_indexing", "Observed computation and view/indexing contributions, such as a reshape that also copies."
            elif key in self.original_indexing:
                role, note = "indexing", "View, layout indexing or tuple selection; no independent arithmetic kernel."
            elif lowered:
                role, note = "compute", "Observed direct/lazy computation membership with named materialized input boundaries."
            else:
                role, note = "unknown", "No direct execution membership was established; complete IR provenance remains available."
            status[key] = {"role": role, "note": note, "lowering": lowered, "fusion": fused,
                           "indexing_membership": indexed, "uncertain_membership": uncertain,
                           "transformed_fx_names": [name for name, node in self.fx.items() if key in node.get("origin_fx_ids", [])],
                           "direct_lazy_body_proofs": [proof for annotation in self.annotations.values()
                                                       for proof in annotation["execution_evidence"]["direct_lazy_body_proofs"]
                                                       if key in proof["execution_fx_ids"]]}
        self.capture["fx_status"] = status
        self.capture["execution_annotation"] = {"version": 2, "method": "actual final IR bindings + owned intermediate buffer dependencies + actual transformed FX argument DAG + exact pre-fusion member union",
                                                "never_use_origin_sets_as_execution": True, "original_indexing_fx_ids": ordered_ids(self.original_indexing),
                                                "unknown_original_compute_fx_ids": ordered_ids(key for key, value in status.items() if value["role"] == "unknown"),
                                                "not_an_exclusive_partition": True}
        proof = scheduler_backend_evidence(self.capture)
        templates = []
        for node in self.capture["stages"]["fusion"]["scheduler_nodes"]:
            for leaf in node["leaf_members"]:
                if leaf.get("is_template"):
                    root = leaf.get("origin_node") or {}
                    templates.append({"scheduler_node": node["id"], "leaf": leaf["name"], "backend": leaf["backend"],
                                      "selected_caller": leaf.get("selected_caller"),
                                      "direct_matmul_fx_ids": ordered_ids(set(root.get("origin_fx_ids", [])) & self.mm_ids)})
        covered = {key for template in templates for key in template["direct_matmul_fx_ids"]}
        proof.update({"actual_template_leaves": templates, "original_matmul_fx_ids": ordered_ids(self.mm_ids),
                      "template_leaf_count": len(templates), "template_original_mm_ids": ordered_ids(covered),
                      "actual_template_root_matmul_fx_ids": ordered_ids(covered),
                      "all_original_matmuls_proven_triton": False})
        self.capture["matmul_lowering"] = []
        for key in ordered_ids(self.mm_ids):
            native = [name for name, value in self.annotations.items() if key in value["native_original_mm_ids"]]
            template = [name for name, value in self.annotations.items() if key in value["template_original_mm_ids"]]
            extern = [name for name, value in self.annotations.items() if self.leaves[name].get("is_extern") and key in value["execution_fx_ids"]]
            self.capture["matmul_lowering"].append({"fx_id": key, "fx_name": self.original[key]["fx_name"], "target": self.original[key]["target"],
                                                   "native_dot_reduction_leaves": native,
                                                   "template_leaves": template, "extern_leaves": extern,
                                                   "unknown": not (native or template or extern),
                                                   "runtime_backend_proven": bool(template) and proof["all_original_matmuls_proven_triton"]})
        self.capture["backend_execution_proof"] = proof
        multi_mm = []
        for node in self.capture["stages"]["fusion"]["scheduler_nodes"]:
            if node["matmul_leaf_count"] > 1:
                multi_mm.append({"scheduler_node": node["id"], "template_leaf_count": node["template_leaf_count"],
                                 "template_original_mm_ids": node["template_original_mm_ids"],
                                 "native_matmul_leaf_count": node["native_matmul_leaf_count"], "native_original_mm_ids": node["native_original_mm_ids"],
                                 "matmul_leaf_count": node["matmul_leaf_count"], "matmul_original_mm_ids": node["matmul_original_mm_ids"]})
        return {"name": self.capture.get("name", "FX graph"), "execution_success": self.capture.get("compile", {}).get("execution_succeeded"),
                "scheduler_nodes": {stage: len(self.capture["stages"][stage]["scheduler_nodes"]) for stage in ("lowering", "fusion")},
                "original_matmuls": len(self.mm_ids), "triton_template_matmuls": len(covered),
                "native_dot_matmuls": len({key for value in self.annotations.values() for key in value["native_original_mm_ids"]}),
                "all_original_matmuls_proven_triton": proof["all_original_matmuls_proven_triton"],
                "multi_matmul_fused_nodes": multi_mm, "unknown_compute_fx_ids": self.capture["execution_annotation"]["unknown_original_compute_fx_ids"],
                "partial_leaf_memberships": [name for name, annotation in self.annotations.items() if annotation["execution_evidence"]["uncertainties"]],
                "runtime_kernel_calls_in_generated_wrapper": proof.get("declared_runtime_calls"),
                "runtime_matmul_template_calls": None,
                "unique_runtime_matmul_template_bodies": None,
                "eager_wrapper_tensor_computations": proof.get("eager_wrapper_tensor_computations", []),
                "note": "Execution success and numerical agreement are separate. Membership uncertainty is explicit, and complete provenance remains unchanged."}


def annotate_scheduler_execution(capture):
    """Add execution and indexing evidence to an actual scheduler capture.

    Mutates and returns ``capture``. The original graph, complete provenance,
    compiler IR and source artifacts remain unchanged. Indexing is identified
    from actual view IR, exact pass-through IR identities and container getters;
    operation names alone never make an operation metadata-only. Memberships
    can overlap, and internal split stages with incomplete evidence remain
    explicitly partial. This function does not import PyTorch or run a compiler.
    """
    stages = capture.get("stages", {})
    if not all(stage in stages for stage in ("lowering", "fusion")):
        raise ValueError("Capture must contain actual lowering and fusion stages")
    if not stages["lowering"].get("final_ir_values") and stages["lowering"].get("scheduler_nodes"):
        raise ValueError("Capture has scheduler nodes but no final IR observations")
    capture["execution_audit"] = _ExecutionAnnotator(capture).annotate()
    return capture
