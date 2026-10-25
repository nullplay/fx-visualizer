"""Draw actual lowering and fusion regions over an original, directed FX graph."""

from __future__ import annotations

import base64
import copy
import gzip
import html
import json
import uuid
from pathlib import Path


_STYLE = """
<style>
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
.fx-regions{background:var(--widget-background,#ffffff);color:var(--widget-foreground,#111827);font:14px/1.45 system-ui,sans-serif;max-width:100%}
.fx-regions h3{font-size:17px;margin:0 0 6px}
.fx-regions .heading,.fx-regions .controls,.fx-regions .foot{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.fx-regions .heading{justify-content:space-between;align-items:baseline}
.fx-regions .muted{color:var(--widget-muted,#6b7280);font-size:12px}
.fx-regions .controls{margin:4px 0}
.fx-regions button,.fx-regions select{font:inherit;color:var(--widget-foreground,#111827);background:var(--widget-surface,#f3f4f6);border:1px solid var(--widget-border,#d1d5db);border-radius:7px;padding:3px 8px;max-width:100%}
.fx-regions .controls select{max-width:220px}
.fx-regions button{cursor:pointer}
.fx-regions button:hover,.fx-regions button:focus-visible,.fx-regions select:focus-visible{outline:2px solid var(--widget-accent,#6366f1);outline-offset:2px}
.fx-regions label{display:inline-flex;align-items:center;gap:4px}
.fx-regions input{accent-color:var(--widget-accent,#6366f1)}
.fx-regions .workspace{display:grid;grid-template-columns:minmax(0,1fr);gap:12px}
.fx-regions .viewport{position:relative;box-sizing:border-box;height:clamp(300px,55vh,480px);border:1px solid var(--widget-border,#d1d5db);border-radius:10px;overflow:hidden;background:var(--widget-background,#ffffff)}
.fx-regions .inspector{min-width:0}
.fx-regions svg{display:block;width:100%;height:100%;touch-action:none;cursor:grab;background:var(--widget-background,#ffffff)}
.fx-regions svg.is-dragging{cursor:grabbing}
.fx-regions [data-fx-id]{cursor:pointer}
.fx-regions [data-fx-id].chosen>polygon,.fx-regions [data-fx-id].chosen>rect,.fx-regions [data-fx-id].chosen>ellipse,.fx-regions [data-fx-id].chosen>path{stroke:var(--widget-accent,#6366f1);stroke-width:3.5}
.fx-regions .lowering-boundary,.fx-regions .fusion-boundary{cursor:pointer}
.fx-regions .lowering-boundary path,.fx-regions .fusion-boundary path{pointer-events:stroke;stroke-linejoin:round;stroke-linecap:round}
.fx-regions .lowering-boundary:focus-visible,.fx-regions .fusion-boundary:focus-visible{outline:2px solid var(--widget-accent,#6366f1);outline-offset:3px}
.fx-regions .fx-edge{pointer-events:none}
.fx-regions .chosen-region path{stroke-width:4}
.fx-regions .detail{background:var(--widget-surface,#f3f4f6);border:1px solid var(--widget-border,#d1d5db);border-radius:8px;padding:9px 11px;overflow-wrap:anywhere}
.fx-regions .detail strong{display:block;margin-bottom:3px}
.fx-regions .detail pre{font:12px/1.45 ui-monospace,monospace;margin:5px 0 0;white-space:pre-wrap;max-height:220px;overflow:auto}
.fx-regions .arguments{margin-top:10px;border:1px solid var(--widget-border,#d1d5db);border-radius:8px;padding:8px 10px;background:var(--widget-surface,#f3f4f6)}
.fx-regions .arguments summary{cursor:pointer;overflow-wrap:anywhere}
.fx-regions .arguments details{margin-top:8px}
.fx-regions .arguments pre{font:12px/1.45 ui-monospace,monospace;white-space:pre-wrap;max-height:240px;overflow:auto}
.fx-regions .scheduler-list{margin-top:10px;overflow-x:auto}
.fx-regions .scheduler-list table{width:100%;border-collapse:collapse;font-size:12px}
.fx-regions .scheduler-list th,.fx-regions .scheduler-list td{text-align:left;padding:5px 4px;border-bottom:1px solid var(--widget-border,#d1d5db);overflow-wrap:anywhere}
.fx-regions .scheduler-list button{font-size:12px;text-align:left;overflow-wrap:anywhere}
.fx-regions .scheduler-list button[aria-pressed="true"]{background:var(--widget-accent,#6366f1);color:var(--widget-on-accent,#ffffff)}
.fx-regions .foot{justify-content:space-between;margin-top:7px}
.fx-regions .load-message{padding:16px;color:var(--widget-muted,#6b7280)}
@media(min-width:1000px){.fx-regions .workspace{grid-template-columns:minmax(0,1fr) 320px;height:clamp(340px,calc(100dvh - 200px),700px)}.fx-regions .viewport{height:100%;min-height:0}.fx-regions .inspector{min-height:0;overflow:auto}}
@media(max-width:500px){.fx-regions .controls{gap:6px}.fx-regions .heading .muted{width:100%}}
</style>
"""

_VIEWER_SCRIPT = r"""
function initializeViewer(root,decoded){
 try{
  root.querySelector('.viewport').innerHTML=decoded.svg;
  const data=decoded.data,svg=root.querySelector('svg'),viewport=root.querySelector('.viewport');
  const nodes=new Map(data.nodes.map(n=>[n.id,n]));
  const matmulLowering=new Map((data.matmul_lowering||[]).map(item=>[item.fx_id,item]));
  const nodeElements=new Map([...root.querySelectorAll('[data-fx-id]')].map(e=>[e.dataset.fxId,e]));
  const edgeElements=[...root.querySelectorAll('.fx-edge')];
  const groups=[...data.lowering_groups.map(g=>({...g,stage:'lowering'})),...data.fusion_groups.map(g=>({...g,stage:'fusion'}))];
  const key=g=>g.stage+':'+g.id;
  const groupMap=new Map(groups.map(g=>[key(g),g]));
  const regionElements=new Map();
  const schedulerButtons=new Map();
  for(const stage of ['lowering','fusion'])root.querySelectorAll('.'+stage+'-boundary').forEach(e=>{
   const id=stage+':'+e.dataset.scheduler;
   if(!regionElements.has(id))regionElements.set(id,[]);
   regionElements.get(id).push(e);
  });
  const full=data.view_box.map(Number);
  let view=full.slice(),drag=null,selected=null;
  function applyView(){svg.setAttribute('viewBox',view.join(' '));}
  function screenPoint(event,inverse){const p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;return p.matrixTransform(inverse||svg.getScreenCTM().inverse());}
  function fitBox(box,padding=30){
   const r=viewport.getBoundingClientRect(),aspect=Math.max(r.width,1)/Math.max(r.height,1);
   const w=Math.max(box.width+padding*2,(box.height+padding*2)*aspect),h=w/aspect;
   view=[box.x+box.width/2-w/2,box.y+box.height/2-h/2,w,h];applyView();
  }
  function fitAll(){view=full.slice();applyView();}
  function zoom(factor,anchor){
   const w=Math.max(100,Math.min(full[2]*2,view[2]*factor)),actual=w/view[2];
   const p=anchor||{x:view[0]+view[2]/2,y:view[1]+view[3]/2};
   view=[p.x+(view[0]-p.x)*actual,p.y+(view[1]-p.y)*actual,w,view[3]*actual];applyView();
  }
  function boundaryIds(id){
   const group=groupMap.get(id);
   return group?.stage==='fusion'?[id,...group.lowering_members.map(child=>'lowering:'+child)]:[id];
  }
  function layers(){
   const only=root.querySelector('[data-only]').checked&&selected;
   const spotlight=selected?.kind==='group'&&selected.spotlight&&root.querySelector('[data-layer="'+groupMap.get(selected.id).stage+'"]').checked;
   const members=spotlight?new Set(groupMap.get(selected.id).member_fx_ids):null;
   const visibleBoundaries=spotlight?new Set(boundaryIds(selected.id)):null;
   for(const [id,elements] of regionElements){
    const stage=groupMap.get(id).stage;
    const enabled=root.querySelector('[data-layer="'+stage+'"]').checked;
    elements.forEach(element=>{
     element.style.display=enabled&&(!spotlight||visibleBoundaries.has(id))&&(!only||element.classList.contains('chosen-region'))?'':'none';
     element.setAttribute('aria-pressed',String(Boolean(spotlight&&id===selected.id)));
    });
   }
   nodeElements.forEach((element,id)=>element.style.opacity=spotlight&&!members.has(id)?'0.16':'');
   edgeElements.forEach(element=>element.style.opacity=spotlight&&!(members.has(element.dataset.source)&&members.has(element.dataset.target))?'0.12':'');
   schedulerButtons.forEach((button,id)=>button.setAttribute('aria-pressed',String(selected?.kind==='group'&&selected.id===id)));
  }
  function clear(){
   selected=null;
   nodeElements.forEach(e=>e.classList.remove('chosen'));
   regionElements.forEach(elements=>elements.forEach(e=>e.classList.remove('chosen-region')));
   root.querySelectorAll('[data-group-select]').forEach(e=>e.value='');
  }
  function showDetail(title,summary,extra){
   root.querySelector('.detail-title').textContent=title;
   root.querySelector('.detail-summary').textContent=summary;
   root.querySelector('.detail-extra').textContent=extra||'';
  }
  function initialDetail(){
   showDetail('Original FX and observed scheduler groups',data.summary,
    'Solid: lowering; thick dashed: every scheduler node after fusion\nBroad envelopes; click to highlight exact members\nAfter-fusion selection also highlights its lowering boundaries and labels\nClick the same boundary or label again to restore all groups\nClick a node to inspect computation/indexing; select a group to zoom'+(data.capture_note?'\n'+data.capture_note:'')+(data.partial_count?'\npartial '+data.partial_count+' partial groups: only observed membership is displayed.':''));
  }
  function resetSelection(){clear();initialDetail();layers();}
  function contribution(group,id){
   const computation=group.execution_fx_ids.includes(id),indexing=group.indexing_fx_ids.includes(id);
   return computation&&indexing?'compute + indexing':indexing?'indexing':'compute';
  }
  function selectGroup(id,{spotlight=false,fit=true}={}){
   const group=groupMap.get(id);if(!group)return;
   clear();selected={kind:'group',id,spotlight};
   group.member_fx_ids.forEach(n=>nodeElements.get(n)?.classList.add('chosen'));
   boundaryIds(id).forEach(key=>regionElements.get(key)?.forEach(e=>e.classList.add('chosen-region')));
   root.querySelector('[data-group-select="'+group.stage+'"]').value=id;
   const fused=group.stage==='fusion';
   const heading=(fused?'After fusion: ':'Lowering: ')+group.id;
   const summary=[group.class||group.scheduler_type||'',(group.matmul_fx_ids||[]).length+' MM',group.execution_fx_ids.length+' compute',group.indexing_fx_ids.length+' indexing',group.codegen_backend||'',group.certainty==='partial'?'partial':'',group.geometry_status==='unmapped_or_empty'?'no mapped FX members':'',group.geometry_status==='disconnected'?'shown as multiple boundaries':''].filter(Boolean).join(' · ');
   const extra=group.member_fx_ids.map(n=>n+' · '+(nodes.get(n)?.target||'')+' ['+contribution(group,n)+']');
   if(group.matmul_fx_ids?.length)extra.unshift('MM members: '+group.matmul_fx_ids.join(', '));
   if(fused)extra.unshift('Lowering members: '+group.lowering_members.join(', '));
   if(group.incidental_enclosed_fx_ids?.length)extra.push('Incidental nonmembers inside the envelope: '+group.incidental_enclosed_fx_ids.join(', '));
   if(group.unknown_fx_ids?.length)extra.push('Unresolved membership: '+group.unknown_fx_ids.join(', '));
   showDetail(heading,summary,extra.join('\n'));
   layers();if(fit&&group.bbox)fitBox(group.bbox);
  }
  function toggleBoundary(id){
   if(!groupMap.get(id)?.bbox)return;
   if(selected?.kind==='group'&&selected.id===id&&selected.spotlight)resetSelection();
   else selectGroup(id,{spotlight:true,fit:false});
  }
  function selectNode(id){
   const node=nodes.get(id);if(!node)return;clear();selected={kind:'node',id};
   nodeElements.get(id)?.classList.add('chosen');
   const related=groups.filter(g=>g.member_fx_ids.includes(id));
   related.forEach(g=>regionElements.get(key(g))?.forEach(e=>e.classList.add('chosen-region')));
   const descriptions=related.map(g=>(g.stage==='lowering'?'Lowering':'After fusion')+' '+contribution(g,id)+': '+g.id+(g.certainty==='partial'?' [partial]':''));
   const geometry=node.geometry||node;
   const extra=[node.fx_name?'FX name: '+node.fx_name:'',node.source_name?'source: '+node.source_name:''];
   if(node.inlined_cast_schedulers?.length)extra.push('Inlined cast in lowering schedulers: '+node.inlined_cast_schedulers.join(', '));
   if(node.geometry)extra.push('Geometry: '+JSON.stringify(node.geometry,null,2));
   else if(geometry.shape)extra.push('shape: '+JSON.stringify(geometry.shape));
   if(node.actual_args)extra.push('Arguments: '+JSON.stringify(node.actual_args,null,2));
   if(node.actual_kwargs?.items?.length)extra.push('Keyword arguments: '+JSON.stringify(node.actual_kwargs,null,2));
   if(node.output_ports?.length)extra.push('Returned values: '+JSON.stringify(node.output_ports,null,2));
   const lowering=matmulLowering.get(id);
   if(lowering){
    for(const [field,label] of [['native_dot_reduction_leaves','native dot/reduction'],['template_leaves','GEMM template'],['extern_leaves','extern']]){
     if(lowering[field]?.length)extra.push(label+': '+lowering[field].join(', '));
    }
    if(lowering.unknown)extra.push('matmul lowering membership unresolved');
   }
   showDetail(id+' · '+(node.target||node.label||node.kind),descriptions.join(' · ')||'Original FX boundary or computation without observed membership',extra.join('\n'));
   layers();
  }
  for(const stage of ['lowering','fusion']){
   const select=root.querySelector('[data-group-select="'+stage+'"]');
   for(const group of groups.filter(g=>g.stage===stage)){
    const option=document.createElement('option');option.value=key(group);
    option.textContent=group.id+' · '+(group.class||group.scheduler_type||'')+' · MM '+(group.matmul_fx_ids||[]).length+' · '+group.execution_fx_ids.length+' compute / '+group.indexing_fx_ids.length+' indexing'+(group.certainty==='partial'?' [partial]':'')+(group.codegen_backend?' · '+group.codegen_backend:'');
    select.appendChild(option);
   }
   select.disabled=!groups.some(g=>g.stage===stage);
   select.addEventListener('change',()=>{if(select.value)selectGroup(select.value);else resetSelection();});
  }
  const schedulerList=root.querySelector('[data-scheduler-list]');
  for(const group of groups.filter(g=>g.stage==='fusion')){
   const row=document.createElement('tr'),name=document.createElement('td'),type=document.createElement('td'),mm=document.createElement('td'),button=document.createElement('button');
   button.type='button';button.textContent=group.id;button.setAttribute('aria-pressed','false');
   button.addEventListener('click',()=>selectGroup(key(group),{spotlight:true}));
   schedulerButtons.set(key(group),button);name.appendChild(button);
   type.textContent=group.class||group.scheduler_type||'unspecified';mm.textContent=(group.matmul_fx_ids||[]).length;
   row.append(name,type,mm);schedulerList.appendChild(row);
  }
  root.querySelectorAll('[data-layer],[data-only]').forEach(e=>e.addEventListener('change',layers));
  root.querySelector('[data-action="fit"]').addEventListener('click',fitAll);
  root.querySelector('[data-action="in"]').addEventListener('click',()=>zoom(1/1.25));
  root.querySelector('[data-action="out"]').addEventListener('click',()=>zoom(1.25));
  root.querySelector('[data-action="clear"]').addEventListener('click',resetSelection);
  regionElements.forEach((elements,id)=>{
   if(!groupMap.get(id).bbox)return;
   elements.forEach(element=>{
    element.setAttribute('tabindex','0');element.setAttribute('role','button');
    element.setAttribute('aria-label',(groupMap.get(id).stage==='lowering'?'Lowering':'After fusion')+' '+groupMap.get(id).id+' focus boundary or restore');
    element.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){if(!event.repeat)toggleBoundary(id);event.preventDefault();event.stopPropagation();}});
    element.addEventListener('click',event=>{if(event.detail===0){toggleBoundary(id);event.stopPropagation();}});
   });
  });
  nodeElements.forEach((element,id)=>{
   element.setAttribute('tabindex','0');element.setAttribute('role','button');
   element.setAttribute('aria-label',id+' '+(nodes.get(id)?.target||''));
   element.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){selectNode(id);event.preventDefault();event.stopPropagation();}});
  });
  svg.addEventListener('wheel',event=>{event.preventDefault();zoom(event.deltaY>0?1.12:1/1.12,screenPoint(event));},{passive:false});
  svg.addEventListener('pointerdown',event=>{
   if(event.button!==0)return;
   const inverse=svg.getScreenCTM().inverse(),p=screenPoint(event,inverse);
   const boundary=event.target.closest('.lowering-boundary,.fusion-boundary');
   const boundaryId=boundary?(boundary.classList.contains('lowering-boundary')?'lowering:':'fusion:')+boundary.dataset.scheduler:null;
   drag={pointer:event.pointerId,inverse,p,start:view.slice(),clientX:event.clientX,clientY:event.clientY,node:event.target.closest('[data-fx-id]')?.dataset.fxId,boundary:boundaryId,moved:false};
   svg.setPointerCapture(event.pointerId);svg.classList.add('is-dragging');
  });
  svg.addEventListener('pointermove',event=>{
   if(!drag||drag.pointer!==event.pointerId)return;
   const p=screenPoint(event,drag.inverse);
   drag.moved=drag.moved||Math.hypot(event.clientX-drag.clientX,event.clientY-drag.clientY)>4;
   if(drag.moved){view=[drag.start[0]-(p.x-drag.p.x),drag.start[1]-(p.y-drag.p.y),drag.start[2],drag.start[3]];applyView();}
  });
  function release(event,canceled){
   if(!drag||drag.pointer!==event.pointerId)return;
   const previous=drag;drag=null;svg.classList.remove('is-dragging');
   if(svg.hasPointerCapture(event.pointerId))svg.releasePointerCapture(event.pointerId);
   if(!canceled&&!previous.moved){
    if(previous.node)selectNode(previous.node);
    else if(previous.boundary)toggleBoundary(previous.boundary);
   }
  }
  svg.addEventListener('pointerup',e=>release(e,false));svg.addEventListener('pointercancel',e=>release(e,true));
  svg.addEventListener('lostpointercapture',()=>{drag=null;svg.classList.remove('is-dragging');});
  root.addEventListener('keydown',event=>{
   if(event.key==='Escape'){resetSelection();return;}
   if(event.target.matches('input,select,button,[data-fx-id],[data-scheduler]'))return;
   if(event.key==='+'||event.key==='='){zoom(1/1.25);event.preventDefault();}
   if(event.key==='-'){zoom(1.25);event.preventDefault();}
   if(event.key==='0'){fitAll();event.preventDefault();}
  });
  const argumentPanel=root.querySelector('[data-arguments]');
  if(argumentPanel){
   const rows=data.nodes.filter(node=>node.fx_op==='placeholder'||node.fx_op==='output'||node.kind==='input'||node.kind==='output');
   for(const node of rows){
    const item=document.createElement('details'),label=document.createElement('summary'),body=document.createElement('pre');
    label.textContent=(node.kind==='output'||node.fx_op==='output'?'Output':'Input')+': '+(node.fx_name||node.label||node.id);
    body.textContent=JSON.stringify({geometry:node.geometry,arguments:node.actual_args,returned_values:node.output_ports},null,2);
    item.append(label,body);argumentPanel.appendChild(item);
   }
  }
  root.dataset.ready='true';initialDetail();layers();fitAll();
 }catch(error){
  root.dataset.ready='error';
  root.querySelector('.viewport').textContent='Could not load visualization: '+error.message;
 }
}
"""

_SCRIPT = r"""
<script id="%%LOADER%%" data-fx-payload="%%ENCODED%%">
(async()=>{
%%VIEWER_SCRIPT%%
 const root=document.getElementById('%%ROOT%%');
 try{
  const bytes=Uint8Array.from(atob(document.getElementById('%%LOADER%%').dataset.fxPayload),c=>c.charCodeAt(0));
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  initializeViewer(root,JSON.parse(await new Response(stream).text()));
 }catch(error){
  root.dataset.ready='error';
  root.querySelector('.viewport').textContent='Could not load visualization: '+error.message;
 }
})();
</script>
""".replace("%%VIEWER_SCRIPT%%", _VIEWER_SCRIPT)

_COMPARISON_SCRIPT = r"""
<script id="%%LOADER%%" data-fx-payload="%%ENCODED%%">
(async()=>{
%%VIEWER_SCRIPT%%
 const roots=%%ROOTS%%.map(id=>document.getElementById(id));
 try{
  const bytes=Uint8Array.from(atob(document.getElementById('%%LOADER%%').dataset.fxPayload),c=>c.charCodeAt(0));
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  const decoded=JSON.parse(await new Response(stream).text());
  decoded.viewers.forEach(viewer=>initializeViewer(document.getElementById(viewer.root_id),{
   svg:viewer.svg_lines.map(index=>decoded.svg_lines[index]).join(''),data:viewer.data
  }));
 }catch(error){
  roots.forEach(root=>{
   root.dataset.ready='error';
   root.querySelector('.viewport').textContent='Could not load visualization: '+error.message;
  });
 }
})();
</script>
""".replace("%%VIEWER_SCRIPT%%", _VIEWER_SCRIPT)


def _widget_parts(payload, title, post_count, *, compile_info=None, matmul_lowering=None):
    root_id = "fx-regions-" + uuid.uuid4().hex[:12]
    from .regions import _svg

    svg = _svg(payload, id_prefix=root_id + "-")
    node_fields = ("id", "kind", "fx_name", "fx_op", "target", "label", "source_name", "shape", "stride", "dtype", "geometry", "bbox", "is_matmul", "inlined_cast_schedulers", "actual_args", "actual_kwargs", "output_ports")
    group_fields = ("id", "member_fx_ids", "execution_fx_ids", "indexing_fx_ids", "matmul_fx_ids", "lowering_members", "incidental_enclosed_fx_ids", "incidental_intersected_fx_ids", "bbox", "certainty", "unknown_fx_ids", "geometry_status", "class", "scheduler_type", "codegen_backend")
    lowering = payload["lowering_groups"]
    fused = payload["fusion_groups"]
    partial_count = sum(group.get("certainty") == "partial" for group in lowering)
    matmul_count = sum(bool(node.get("is_matmul") or node.get("kind") == "mm") for node in payload["nodes"])
    summary = f"Matmul {matmul_count} | Lowering {len(lowering)} -> After fusion {post_count} | After-fusion boundaries {len(fused)}"
    capture_note = ""
    if (compile_info or {}).get("mode") == "schedule_only":
        capture_note = "Scheduler preview: untuned candidates and heuristic fusion; no final GPU compilation, autotuning or execution"
    elif (compile_info or {}).get("mode") == "fx_only":
        summary = f"FX nodes {len(payload['nodes'])} | Directed edges {len(payload['edges'])} | Matmul {matmul_count}"
        capture_note = "Original FX only; no compiler capture or real execution. Intermediate geometry is shown when present in node metadata."
        metadata = (compile_info or {}).get("metadata_capture", {})
        if metadata.get("status") == "partial":
            capture_note += " Partial metadata: " + metadata["reason"]
    data = {
        "view_box": payload["view_box"],
        "nodes": [{key: node[key] for key in node_fields if key in node} for node in payload["nodes"]],
        "edges": [{key: edge[key] for key in ("id", "source", "target", "ports", "port", "label") if key in edge} for edge in payload["edges"]],
        "lowering_groups": [{key: group[key] for key in group_fields if key in group} for group in lowering],
        "fusion_groups": [{key: group[key] for key in group_fields if key in group} for group in fused],
        "summary": summary,
        "partial_count": partial_count,
        "capture_note": capture_note,
        "matmul_lowering": matmul_lowering or [],
    }
    safe_title = html.escape(title)
    content = f"""
<section id="{root_id}" class="fx-regions" tabindex="0" aria-label="{safe_title} FX scheduler boundaries">
 <div class="heading"><h3>{safe_title} · Original FX</h3><span class="muted">{html.escape(summary)}</span></div>
 <div class="muted">{html.escape(capture_note)}</div>
 <div class="controls">
  <label><input type="checkbox" data-layer="lowering" checked>Lowering boundaries</label>
  <label><input type="checkbox" data-layer="fusion" checked>After fusion boundaries</label>
  <label><input type="checkbox" data-only>Selected boundaries only</label>
  <button type="button" data-action="in" aria-label="Zoom in">+</button>
  <button type="button" data-action="out" aria-label="Zoom out">-</button>
  <button type="button" data-action="fit">Fit graph</button>
  <button type="button" data-action="clear">Clear selection</button>
 </div>
 <div class="controls">
  <label>Lowering <select data-group-select="lowering"><option value="">Select group / zoom</option></select></label>
  <label>After fusion <select data-group-select="fusion"><option value="">Select group / zoom</option></select></label>
 </div>
 <div class="workspace">
  <div class="viewport"><div class="load-message">Loading graph...</div></div>
  <aside class="inspector" aria-label="FX node and scheduler inspector">
   <div class="detail" aria-live="polite"><strong class="detail-title">Observed scheduler boundaries</strong><div class="detail-summary">{html.escape(summary)}</div><pre class="detail-extra"></pre></div>
   <div class="scheduler-list"><strong>After fusion: {len(fused)} scheduler nodes</strong><table><thead><tr><th scope="col">Scheduler</th><th scope="col">Class</th><th scope="col">MM</th></tr></thead><tbody data-scheduler-list></tbody></table></div>
   <details class="arguments"><summary>FX input and output arguments</summary><div data-arguments></div></details>
  </aside>
 </div>
 <div class="foot muted"><span>Solid: lowering; one thick dashed boundary per scheduler after fusion</span><span>Select a scheduler to highlight exact FX members and child lowering groups; scroll to zoom; drag to pan</span></div>
</section>
"""
    heading = f'<h2 class="sr-only">{safe_title} original FX and lowering/fusion boundaries</h2>'
    return root_id, heading, content, {"svg": svg, "data": data}


def _widget_from_parts(root_id, heading, content, decoded):
    encoded = base64.b64encode(gzip.compress(json.dumps(decoded, ensure_ascii=True, separators=(",", ":")).encode(), mtime=0)).decode()
    script = _SCRIPT.replace("%%ROOT%%", root_id).replace("%%LOADER%%", root_id + "-loader").replace("%%ENCODED%%", encoded)
    return heading + _STYLE + content + script


def _build_widget(payload, title, post_count, *, compile_info=None, matmul_lowering=None):
    parts = _widget_parts(payload, title, post_count, compile_info=compile_info, matmul_lowering=matmul_lowering)
    return _widget_from_parts(*parts)


def _pack_viewers(viewers):
    lines, line_ids, packed = [], {}, []
    for viewer in viewers:
        indexes = []
        for line in viewer["svg"].splitlines(keepends=True):
            index = line_ids.get(line)
            if index is None:
                index = len(lines)
                line_ids[line] = index
                lines.append(line)
            indexes.append(index)
        packed.append({"root_id": viewer["root_id"], "svg_lines": indexes, "data": viewer["data"]})
    return {"svg_lines": lines, "viewers": packed}


def _standalone(widget, title):
    return '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '</title><style>body{margin:0;background:var(--widget-background,#ffffff);color:var(--widget-foreground,#111827)}main{max-width:1600px;margin-inline:auto}main>section{margin-bottom:24px}main>section:last-of-type{margin-bottom:0}</style></head><body><main>' + widget + '</main></body></html>'


def _read_capture(capture):
    if isinstance(capture, (str, Path)):
        capture = json.loads(Path(capture).read_text())
    if not isinstance(capture, dict) or not isinstance(capture.get("original_graph"), dict):
        raise ValueError("Expected structured FX capture data with original_graph")
    if capture.get("compile", {}).get("status") in ("running", "error"):
        raise ValueError("Capture did not complete successfully; inspect its compiler diagnostics")
    for stage in ("lowering", "fusion"):
        if "scheduler_nodes" not in capture.get("stages", {}).get(stage, {}):
            raise ValueError(f"Capture is missing actual {stage} scheduler_nodes")
    return capture


def render_capture(*, capture, output_dir, title=None, clustered=True):
    """Render original FX and recorded scheduler evidence without PyTorch."""
    capture = _read_capture(capture)
    groups = [node for stage in ("lowering", "fusion") for node in capture["stages"][stage]["scheduler_nodes"]]
    saved_schedule = capture.get("compile", {}).get("mode") == "schedule_only" and any(
        "loop_ir" in leaf for node in capture["stages"]["lowering"]["scheduler_nodes"] for leaf in node.get("leaf_members", [])
    )
    stale_annotation = capture.get("execution_annotation", {}).get("version", 0) < 3 and bool(capture["stages"]["lowering"].get("final_ir_values"))
    if saved_schedule or stale_annotation or (not capture.get("execution_annotation") and any("execution_fx_ids" not in node or "indexing_fx_ids" not in node for node in groups)):
        from .execution import annotate_scheduler_execution

        capture = copy.deepcopy(capture)
        annotate_scheduler_execution(capture)
    from .regions import layout_regions

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = layout_regions(capture, output_dir, clustered=clustered)
    svg_path = output_dir / "original_fx_regions.svg"
    title = title or payload.get("model") or capture.get("model") or capture.get("name") or "FX graph"
    post_count = len(capture["stages"]["fusion"]["scheduler_nodes"])
    widget = _build_widget(payload, title, post_count, compile_info=capture.get("compile"), matmul_lowering=capture.get("matmul_lowering"))
    widget_path = output_dir / "original_fx_regions.widget.html"
    html_path = output_dir / "original_fx_regions.html"
    capture_path = output_dir / "source_capture.json"
    widget_path.write_text(widget)
    html_path.write_text(_standalone(widget, title))
    capture_path.write_text(json.dumps(capture, ensure_ascii=True, indent=2) + "\n")
    artifacts = {
        "name": capture.get("name", title), "title": title,
        "svg": str(svg_path), "html": str(html_path), "widget": str(widget_path),
        "payload": str(output_dir / "region_payload.json"), "capture": str(capture_path),
        "lowering_count": len(payload["lowering_groups"]), "after_fusion_count": post_count,
        "fusion_region_count": len(payload["fusion_groups"]),
        "partial_lowering_count": sum(group.get("certainty") == "partial" for group in payload["lowering_groups"]),
        "widget_characters": len(widget),
        "capture_mode": capture.get("compile", {}).get("mode", "unspecified"),
    }
    matmul_ids = {node["id"] for node in payload["nodes"] if node.get("is_matmul")}
    classified_leaves = {
        item["fx_id"]: {leaf for field in ("native_dot_reduction_leaves", "template_leaves", "extern_leaves") for leaf in item.get(field, [])}
        for item in capture.get("matmul_lowering", [])
    }
    multi_matmul = []
    for node in capture["stages"]["fusion"]["scheduler_nodes"]:
        operations = set(node.get("member_operation_names", [node["id"]]))
        sources = {fx_id for fx_id, leaves in classified_leaves.items() if leaves & operations} if classified_leaves else set(node.get("execution_fx_ids", [])) & matmul_ids
        if len(sources) > 1:
            multi_matmul.append({"id": node["id"], "matmul_fx_ids": sorted(sources), "certainty": "observed matmul leaves" if classified_leaves else node.get("execution_evidence", {}).get("certainty")})
    artifacts["multi_matmul_source_schedulers"] = multi_matmul
    artifacts["matmul_lowering_counts"] = {
        field: sum(bool(item.get(field)) for item in capture.get("matmul_lowering", []))
        for field in ("native_dot_reduction_leaves", "template_leaves", "extern_leaves", "unknown")
    }
    (output_dir / "visualization.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    return artifacts


def _visualize_scheduler(*, gm, example_inputs, output_dir, config=None, triton_matmul=False, title=None, clustered=True):
    """Draw actual scheduler regions, stopping before final codegen and tuning.

    Inputs keep their device, dtype, shape, strides, storage offsets and aliases.
    Use a dedicated process when another thread might compile concurrently:
    collection patches internal hooks.
    """
    from .capture import capture_scheduler_stages

    output_dir = Path(output_dir).resolve()
    capture = capture_scheduler_stages(
        gm, example_inputs, output_dir / "compiler", config=config,
        triton_matmul=triton_matmul, schedule_only=True,
    )
    capture = dict(capture, name=output_dir.name)
    capture["original_graph"] = dict(capture["original_graph"], name=output_dir.name)
    Path(capture["capture_path"]).write_text(json.dumps(capture, ensure_ascii=True, indent=2) + "\n")
    return render_capture(capture=capture, output_dir=output_dir, title=title or output_dir.name, clustered=clustered)


_MATMUL_MODES = {
    "template": ("Template GEMM", "Lowering with Triton GEMM templates"),
    "native": ("native_matmul=True", "Eligible shapes use broadcast dot/reduction IR; size-one dimensions may retain their existing path"),
}


def _validate_comparison(captures):
    reference = captures[0]
    original = {key: reference["original_graph"].get(key) for key in ("nodes", "edges")}
    for capture in captures[1:]:
        other = {key: capture["original_graph"].get(key) for key in ("nodes", "edges")}
        if other != original:
            raise ValueError("Comparison requires the same original FX nodes, arguments, geometry and directed edges")
        for key in ("inputs", "input_aliases", "input_storage_groups"):
            if capture.get("compile", {}).get(key) != reference.get("compile", {}).get(key):
                raise ValueError(f"Comparison input {key} differs between captures")


def _write_matmul_comparison(variants, output_dir, title):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root_id = "fx-matmul-" + uuid.uuid4().hex[:12]
    payloads = {mode: json.loads(Path(artifact["payload"]).read_text()) for mode, artifact in variants.items()}
    reference = next(iter(payloads.values()))
    positions = [{key: node[key] for key in ("id", "center", "bbox")} for node in reference["nodes"]]
    routes = [{key: edge[key] for key in ("source", "target", "paths", "arrow_polygons")} for edge in reference["edges"]]
    for payload in payloads.values():
        if [{key: node[key] for key in ("id", "center", "bbox")} for node in payload["nodes"]] != positions or [{key: edge[key] for key in ("source", "target", "paths", "arrow_polygons")} for edge in payload["edges"]] != routes:
            raise ValueError("Matmul comparison requires identical original DAG positions and edge routes")
    boxes = [payload["view_box"] for payload in payloads.values()]
    left, top = min(box[0] for box in boxes), min(box[1] for box in boxes)
    view_box = [left, top, max(box[0] + box[2] for box in boxes) - left, max(box[1] + box[3] for box in boxes) - top]
    buttons, panels, viewers = [], [], []
    for mode, artifact in variants.items():
        payload = payloads[mode]
        payload["view_box"] = view_box
        Path(artifact["payload"]).write_text(json.dumps(payload, indent=2))
        svg_path = Path(artifact["svg"])
        from .regions import _svg

        svg_path.write_text(_svg(payload))
        capture = _read_capture(artifact["capture"])
        parts = _widget_parts(payload, artifact["title"], artifact["after_fusion_count"], compile_info=capture.get("compile"), matmul_lowering=capture.get("matmul_lowering"))
        widget = _widget_from_parts(*parts)
        viewer_id, heading, content, decoded = parts
        viewers.append({"root_id": viewer_id, **decoded})
        Path(artifact["widget"]).write_text(widget)
        Path(artifact["html"]).write_text(_standalone(widget, artifact["title"]))
        artifact["widget_characters"] = len(widget)
        (svg_path.parent / "visualization.json").write_text(json.dumps(artifact, indent=2) + "\n")
        label, description = _MATMUL_MODES[mode]
        counts = f"{artifact['lowering_count']} → {artifact['after_fusion_count']} schedulers"
        multi_mm = len(artifact.get("multi_matmul_source_schedulers", []))
        buttons.append(f'<button type="button" data-matmul-mode="{mode}" aria-controls="{root_id}-{mode}" aria-pressed="{str(mode == "template").lower()}">{html.escape(label)}<span>{html.escape(counts)}</span><span>Schedulers containing multiple matmuls: {multi_mm}</span></button>')
        hidden = "" if mode == "template" else " hidden"
        panels.append(f'<div id="{root_id}-{mode}" data-matmul-panel="{mode}"{hidden}><p class="mode-description">{html.escape(description)}</p>{heading}{content}</div>')
    safe_title = html.escape(title)
    style = """<style>
.fx-matmul-comparison{background:var(--widget-background,#ffffff);color:var(--widget-foreground,#111827);font:14px/1.45 system-ui,sans-serif;max-width:100%}
.fx-matmul-comparison>.comparison-title{font-size:17px;margin:0}
.fx-matmul-comparison>.matmul-buttons{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.fx-matmul-comparison [data-matmul-mode]{font:inherit;cursor:pointer;text-align:left;padding:8px 12px;border-radius:8px;border:1px solid var(--widget-border,#d1d5db);background:var(--widget-surface,#f3f4f6);color:var(--widget-foreground,#111827)}
.fx-matmul-comparison [data-matmul-mode] span{display:block;font-size:12px}
.fx-matmul-comparison [data-matmul-mode][aria-pressed="true"]{background:var(--widget-accent,#6366f1);color:var(--widget-on-accent,#ffffff)}
.fx-matmul-comparison [data-matmul-mode]:focus-visible{outline:2px solid var(--widget-accent,#6366f1);outline-offset:3px}
.fx-matmul-comparison [data-matmul-panel][hidden]{display:none}
.fx-matmul-comparison .mode-description{color:var(--widget-muted,#6b7280);margin:8px 0}
</style>"""
    script = """<script>
(()=>{
 const root=document.getElementById('%%ROOT%%');
 const buttons=[...root.querySelectorAll('[data-matmul-mode]')];
 const panels=[...root.querySelectorAll('[data-matmul-panel]')];
 function activate(mode){
  if(!panels.some(panel=>panel.dataset.matmulPanel===mode))return;
  panels.forEach(panel=>panel.hidden=panel.dataset.matmulPanel!==mode);
  buttons.forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.matmulMode===mode)));
  root.dataset.matmulMode=mode;
 }
 buttons.forEach(button=>button.addEventListener('click',()=>activate(button.dataset.matmulMode)));
 activate('template');
})();
</script>""".replace("%%ROOT%%", root_id)
    packed = _pack_viewers(viewers)
    encoded = base64.b64encode(gzip.compress(json.dumps(packed, ensure_ascii=True, separators=(",", ":")).encode(), mtime=0)).decode()
    viewer_script = _COMPARISON_SCRIPT.replace("%%ROOTS%%", json.dumps([viewer["root_id"] for viewer in viewers])).replace("%%LOADER%%", root_id + "-loader").replace("%%ENCODED%%", encoded)
    widget = f'<h2 class="sr-only">{safe_title} matmul lowering comparison</h2>{style}{_STYLE}<section id="{root_id}" class="fx-matmul-comparison" aria-label="{safe_title} matmul lowering comparison"><h3 class="comparison-title">{safe_title}</h3><div class="matmul-buttons" role="group" aria-label="Matmul lowering mode">' + "".join(buttons) + '</div>' + "".join(panels) + '</section>' + viewer_script + script
    widget_path = output_dir / "matmul_comparison.widget.html"
    html_path = output_dir / "matmul_comparison.html"
    manifest_path = output_dir / "matmul_comparison.json"
    widget_path.write_text(widget)
    html_path.write_text(_standalone(widget, title))
    artifact = {"name": output_dir.name, "title": title, "html": str(html_path), "widget": str(widget_path), "manifest": str(manifest_path), "variants": variants, "widget_characters": len(widget)}
    manifest_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def render_matmul_comparison(*, template_capture, native_capture, output_dir, title=None):
    """Switch actual scheduler contours over one identical original FX graph."""
    captures = {"template": _read_capture(template_capture), "native": _read_capture(native_capture)}
    _validate_comparison(list(captures.values()))
    output_dir = Path(output_dir).resolve()
    title = title or output_dir.name
    variants = {mode: render_capture(capture=capture, output_dir=output_dir / mode, title=_MATMUL_MODES[mode][0], clustered=False) for mode, capture in captures.items()}
    return _write_matmul_comparison(variants, output_dir, title)


def visualize_matmul_comparison(*, gm, example_inputs, output_dir, config=None, title=None):
    """Capture template and native dot schedules for one FX graph.

    ``gm`` and ``example_inputs`` follow ``visualize_graph``. ``output_dir``
    receives a combined HTML viewer and separate template/native captures.
    ``config`` supplies shared Inductor options; the comparison itself controls
    ``triton.native_matmul``. ``title`` labels the combined viewer.

    Both captures stop after fusion, before final codegen, autotuning and
    compiled execution. Each retains the original FX, shapes, strides and input
    aliases, using the compiler's existing template and native lowering paths.
    """
    config = dict(config or {})
    if "triton.native_matmul" in config or "triton.native-matmul" in config:
        raise ValueError("Matmul comparison controls triton.native_matmul; remove it from config")
    output_dir = Path(output_dir).resolve()
    variants = {}
    for mode in _MATMUL_MODES:
        mode_config = dict(config, **{"triton.native_matmul": mode == "native"})
        variants[mode] = _visualize_scheduler(
            gm=gm, example_inputs=example_inputs, output_dir=output_dir / mode,
            config=mode_config, triton_matmul=True, title=_MATMUL_MODES[mode][0],
            clustered=False,
        )
    _validate_comparison([_read_capture(artifact["capture"]) for artifact in variants.values()])
    return _write_matmul_comparison(variants, output_dir, title or output_dir.name)


def write_index(*, artifacts, output_dir, title="FX graph scheduler regions"):
    """Combine standalone viewers and record their artifact paths."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    widgets = [Path(artifact["widget"]).read_text() for artifact in artifacts]
    path = output_dir / "index.html"
    path.write_text(_standalone("\n".join(widgets), title))
    (output_dir / "manifest.json").write_text(json.dumps({"title": title, "html": str(path), "graphs": artifacts}, indent=2) + "\n")
    return path
