import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from fx_visualizer.projection import project_capture, replay_projection
from fx_visualizer.regions import _regions


class TestSourceProjection(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.path = Path(self.scratch.name) / 'full.json.gz'
        self.original = {'nodes': [
            {'id': 'n100', 'fx_name': 'mm', 'fx_op': 'call_function', 'target': 'aten.mm.default',
             'kind': 'compute', 'is_matmul': True, 'geometry': {'shape': [8, 16], 'dtype': 'torch.bfloat16'}},
            {'id': 'n101', 'fx_name': 'mul', 'fx_op': 'call_function', 'target': 'aten.mul.Tensor',
             'kind': 'compute', 'geometry': {'shape': [8, 16], 'dtype': 'torch.bfloat16'}},
        ], 'edges': [{'source': 'n100', 'target': 'n101'}]}
        def node(name, ids, root, body):
            return {'id': name, 'name': name, 'class': 'SchedulerNode', 'scheduler_type': 'SchedulerNode',
                    'member_operation_names': [name], 'origin_fx_ids': ids, 'execution_fx_ids': ids,
                    'indexing_fx_ids': [], 'reads': [{'name': 'buf0'}], 'writes': [{'name': 'buf1'}],
                    'leaf_members': [{'name': name, 'execution_fx_ids': ids, 'origin_fx_ids': ids,
                                     'origin_node': {'name': root, 'origin_fx_ids': ids}, 'loop_ir': body}],
                    'execution_evidence': {'certainty': 'observed IR boundaries', 'unknown_fx_ids': []}}
        first = node('op1', ['n100'], 'mm', {'operations': ['load', 'dot', 'reduction']})
        second = node('op2', ['n100', 'n101'], 'mul', {'operations': ['load', 'mul', 'store']})
        self.source = {'original_graph': self.original, 'compile': {'status': 'success', 'mode': 'schedule_only'},
                       'stages': {stage: {'graph_id': 42, 'scheduler_nodes': [first, second],
                                         'edges': [{'source': 'op1', 'target': 'op2'}]}
                                  for stage in ('lowering', 'fusion')},
                       'matmul_lowering': [{'fx_id': 'n100', 'fx_name': 'mm', 'target': 'aten.mm.default',
                                           'native_dot_reduction_leaves': ['op1'], 'template_leaves': [], 'extern_leaves': []}]}
        with gzip.open(self.path, 'wt') as stream:
            json.dump(self.source, stream)
        self.region = copy.deepcopy(self.original)
        self.region['nodes'][0]['id'] = 'n2'
        self.region['nodes'][1]['id'] = 'n3'
        self.region['edges'] = [{'source': 'n2', 'target': 'n3'}]
        self.mapping = {'n100': 'n2', 'n101': 'n3'}

    def test_replay_preserves_classes_dependencies_and_loop_ir(self):
        projected = project_capture(source_capture=self.path, original_graph=self.region, fx_mapping=self.mapping)
        self.assertEqual(replay_projection(projected), projected)
        for stage in ('lowering', 'fusion'):
            for original, scoped in zip(self.source['stages'][stage]['scheduler_nodes'], projected['stages'][stage]['scheduler_nodes']):
                self.assertEqual(original['class'], scoped['class'])
                self.assertEqual(original['member_operation_names'], scoped['member_operation_names'])
                self.assertEqual(original['reads'], scoped['reads'])
                self.assertEqual(original['writes'], scoped['writes'])
                self.assertEqual(original['leaf_members'][0]['loop_ir'], scoped['leaf_members'][0]['loop_ir'])
        self.assertEqual(projected['stages']['fusion']['scheduler_nodes'][0]['execution_fx_ids'], ['n2'])

    def test_epilogue_contribution_does_not_count_as_new_matmul_core(self):
        projected = project_capture(source_capture=self.path, original_graph=self.region, fx_mapping=self.mapping, fusion_nodes=['op2'])
        lowering, fusion = _regions(projected)
        self.assertEqual(len(lowering), 1)
        self.assertEqual(len(fusion), 1)
        self.assertEqual(fusion[0]['execution_fx_ids'], ['n2', 'n3'])
        self.assertEqual(fusion[0]['matmul_fx_ids'], [])

    def test_replay_rejects_changed_scheduler_facts(self):
        projected = project_capture(source_capture=self.path, original_graph=self.region, fx_mapping=self.mapping)
        projected['stages']['fusion']['scheduler_nodes'][0]['class'] = 'FusedSchedulerNode'
        with self.assertRaisesRegex(ValueError, 'compiler facts differ'):
            replay_projection(projected)

    def test_replay_rejects_changed_source(self):
        projected = project_capture(source_capture=self.path, original_graph=self.region, fx_mapping=self.mapping)
        with gzip.open(self.path, 'wt') as stream:
            json.dump(dict(self.source, extra=True), stream)
        with self.assertRaisesRegex(ValueError, 'compiler capture changed'):
            replay_projection(projected)

    def test_rejects_invalid_ids_targets_geometry_and_scheduler_scope(self):
        cases = [
            {'fx_mapping': {'n100': 'n2', 'n101': 'n2'}},
            {'fx_mapping': {'n999': 'n2'}},
            {'fusion_nodes': ['op999']},
            {'original_graph': {'nodes': [dict(self.region['nodes'][0], target='aten.add.Tensor'), self.region['nodes'][1]], 'edges': self.region['edges']}},
            {'original_graph': {'nodes': [dict(self.region['nodes'][0], geometry={'shape': [2, 16], 'dtype': 'torch.bfloat16'}), self.region['nodes'][1]], 'edges': self.region['edges']}},
        ]
        for changes in cases:
            arguments = dict(source_capture=self.path, original_graph=self.region, fx_mapping=self.mapping)
            arguments.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                project_capture(**arguments)


if __name__ == '__main__':
    unittest.main()
