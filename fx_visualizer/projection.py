"""Project recorded full-graph scheduler memberships onto a recovered FX graph."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

from .execution import EXECUTION_ANNOTATION_VERSION, annotate_scheduler_execution

ID_LISTS = {
    'origin_fx_ids', 'execution_fx_ids', 'indexing_fx_ids', 'unknown_fx_ids',
    'direct_root_fx_ids', 'excluded_ancestry_fx_ids', 'native_original_mm_ids',
    'template_original_mm_ids', 'template_selected_mm_ids', 'matmul_fx_ids', 'matmul_original_mm_ids',
    'native_dot_matmul_ids', 'fx_ids', 'pattern_fx_ids',
}
MM_FIELDS = ('native_dot_reduction_leaves', 'template_leaves', 'extern_leaves')


def _load(path):
    path = Path(path)
    data = path.read_bytes()
    decoded = gzip.decompress(data) if path.suffix == '.gz' else data
    return json.loads(decoded), hashlib.sha256(data).hexdigest()


def _remap(value, mapping, id_lists=ID_LISTS):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in id_lists and isinstance(item, list):
                result[key] = [mapping[i] for i in item if i in mapping]
            elif key == 'original_id':
                result[key] = mapping.get(item)
            else:
                result[key] = _remap(item, mapping, id_lists)
        return result
    if isinstance(value, list):
        return [_remap(item, mapping, id_lists) for item in value]
    return value


def project_capture(*, source_capture, original_graph, fx_mapping, fusion_nodes=None, name=None):
    """Keep actual scheduler classes/IR and remap only references to FX nodes.

    ``fx_mapping`` maps full-graph source IDs to the recovered graph IDs.
    ``fusion_nodes`` optionally selects an exact recorded scheduler scope;
    otherwise select nodes with observed or unresolved scoped memberships.
    Replay verifies the persisted source checksum and rebuilds the projection.
    """
    path = Path(source_capture).resolve()
    source, checksum = _load(path)
    if source.get('source_projection'):
        raise ValueError('A source projection must reference a full compiler capture')
    if source['stages']['lowering'].get('final_ir_values'):
        annotate_scheduler_execution(source)
    return _project_capture(source=source, path=path, source_sha256=checksum, original_graph=original_graph,
                            fx_mapping=fx_mapping, fusion_nodes=fusion_nodes, name=name)


def _project_capture(*, source, path, source_sha256, original_graph, fx_mapping, fusion_nodes=None, name=None,
                     annotation_version=EXECUTION_ANNOTATION_VERSION, id_lists=ID_LISTS):
    mapping = dict(fx_mapping)
    originals = {n['id']: n for n in source['original_graph']['nodes']}
    recovered = {n['id']: n for n in original_graph['nodes']}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('FX source projection must be one-to-one')
    for full_id, local_id in mapping.items():
        if full_id not in originals or local_id not in recovered:
            raise ValueError(f'Unknown FX projection ID: {full_id} / {local_id}')
        full, local = originals[full_id], recovered[local_id]
        if full['target'] != local['target'] or full['fx_op'] != local['fx_op']:
            raise ValueError(f'FX target differs in projection: {full_id} / {local_id}')
        for field in ('shape', 'dtype'):
            full_value = (full.get('geometry') or {}).get(field)
            local_value = (local.get('geometry') or {}).get(field)
            if full_value != local_value:
                raise ValueError(f'FX {field} differs in projection: {full_id} / {local_id}')
    selected = set(mapping)
    all_fused = source['stages']['fusion']['scheduler_nodes']
    if fusion_nodes is None:
        fused = []
        for node in all_fused:
            ids = set(node.get('execution_fx_ids', [])) | set(node.get('indexing_fx_ids', []))
            ids |= set(node.get('execution_evidence', {}).get('unknown_fx_ids', []))
            for leaf in node.get('leaf_members', []):
                root = leaf.get('data_origin_node') or leaf.get('origin_node') or {}
                ids |= set(root.get('origin_fx_ids', []))
            if ids & selected:
                fused.append(node)
    else:
        wanted = set(fusion_nodes)
        fused = [n for n in all_fused if n['id'] in wanted]
        if {n['id'] for n in fused} != wanted:
            raise ValueError('Projection contains a scheduler absent from the full capture')
    operations = {op for n in fused for op in n['member_operation_names']}
    lowered = [n for n in source['stages']['lowering']['scheduler_nodes']
               if operations & set(n['member_operation_names'])]
    matmuls = source.get('matmul_lowering', [])
    result = {
        'name': name or 'full_graph_projection', 'schema_version': 1,
        'original_graph': copy.deepcopy(original_graph), 'compile': copy.deepcopy(source['compile']),
        'original_graph_id': source.get('original_graph_id'), 'stages': {},
        'execution_annotation': {'version': annotation_version, 'context': 'verified_full_graph_projection'},
        'fx_status': {mapping[i]: _remap(status, mapping, id_lists) for i, status in source.get('fx_status', {}).items() if i in mapping},
        'matmul_lowering': [], 'accepted_fusions': [],
    }
    for mode, nodes in (('lowering', lowered), ('fusion', fused)):
        projected = []
        for node in nodes:
            item = _remap(copy.deepcopy(node), mapping, id_lists)
            members = set(node['member_operation_names'])
            item['matmul_fx_ids'] = [mapping[m['fx_id']] for m in matmuls if m['fx_id'] in mapping and
                                    members & {leaf for field in MM_FIELDS for leaf in m.get(field, [])}]
            outside = (set(node.get('execution_fx_ids', [])) | set(node.get('indexing_fx_ids', []))) - selected
            item['source_scope'] = {'outside_fx_ids': sorted(outside), 'full_graph_scheduler': node['id']}
            if outside:
                item.setdefault('execution_evidence', {})['certainty'] = 'partial'
            projected.append(item)
        kept = {n['id'] for n in nodes}
        stage = source['stages'][mode]
        result['stages'][mode] = {
            'stage': mode, 'graph_id': stage['graph_id'], 'nodes': projected, 'scheduler_nodes': projected,
            'scheduler_node_count': len(projected),
            'edges': [copy.deepcopy(e) for e in stage['edges'] if e['source'] in kept and e['target'] in kept],
            'source_stage_reference': {'capture': str(path), 'stage': mode},
        }
    for entry in matmuls:
        if entry['fx_id'] not in mapping:
            continue
        item = _remap(copy.deepcopy(entry), mapping, id_lists)
        item['source_fx_id'] = entry['fx_id']
        item['fx_id'] = mapping[entry['fx_id']]
        item['fx_name'] = recovered[item['fx_id']]['fx_name']
        result['matmul_lowering'].append(item)
    fused_ids = {n['id'] for n in fused}
    result['accepted_fusions'] = [_remap(copy.deepcopy(event), mapping, id_lists) for event in source.get('accepted_fusions', []) if event.get('result') in fused_ids]
    result['source_projection'] = {
        'version': 1, 'source_capture': str(path), 'sha256': source_sha256,
        'fx_mapping': mapping, 'fusion_nodes': [n['id'] for n in fused],
        'full_lowering_count': len(source['stages']['lowering']['scheduler_nodes']),
        'full_fusion_count': len(all_fused), 'source_original_fx_count': len(source['original_graph']['nodes']),
        'method': 'Actual full-graph schedule; only FX references and displayed node scope are projected. No region recompilation.',
    }
    return result


def replay_projection(capture):
    projection = capture['source_projection']
    path = Path(projection['source_capture'])
    raw_source, checksum = _load(path)
    if projection.get('version') != 1 or checksum != projection['sha256']:
        raise ValueError('Source compiler capture changed; regenerate the FX projection')
    if raw_source.get('source_projection'):
        raise ValueError('A source projection must reference a full compiler capture')
    arguments = dict(path=path, source_sha256=checksum, original_graph=capture['original_graph'],
                     fx_mapping=projection['fx_mapping'], fusion_nodes=projection['fusion_nodes'], name=capture.get('name'))
    refreshed = copy.deepcopy(raw_source)
    if refreshed['stages']['lowering'].get('final_ir_values'):
        annotate_scheduler_execution(refreshed)
    result = _project_capture(source=refreshed, **arguments)
    if result == capture:
        return result
    if capture.get('execution_annotation', {}).get('version') == 3:
        # Verify the exact old record before replacing its derived annotations.
        legacy = _project_capture(source=raw_source, **arguments,
                                  annotation_version=3, id_lists=ID_LISTS - {'matmul_original_mm_ids'})
        if legacy == capture:
            return result
    raise ValueError('Projected compiler facts differ from the recorded full-graph source')
