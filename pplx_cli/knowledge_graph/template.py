# Constants for D3.js force graph template
D3_JS_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Knowledge Graph - {title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; overflow: hidden; }}
#container {{ width: 100vw; height: 100vh; position: relative; }}
svg {{ width: 100%; height: 100%; }}
#info-panel {{ position: absolute; top: 16px; right: 16px; background: rgba(30,30,60,0.92); color: #e0e0e0; padding: 14px 18px; border-radius: 10px; font-size: 13px; pointer-events: none; border: 1px solid rgba(100,100,180,0.25); backdrop-filter: blur(8px); max-width: 260px; }}
#info-panel h3 {{ margin-bottom: 6px; font-size: 14px; color: #7eb8ff; }}
#info-panel .stat {{ margin: 2px 0; }}
#legend {{ position: absolute; bottom: 20px; left: 20px; background: rgba(30,30,60,0.92); color: #ccc; padding: 10px 14px; border-radius: 8px; font-size: 12px; border: 1px solid rgba(100,100,180,0.25); backdrop-filter: blur(8px); }}
.legend-item {{ display: flex; align-items: center; margin: 3px 0; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}
.tooltip {{ position: absolute; background: rgba(20,20,50,0.95); color: #eee; padding: 8px 12px; border-radius: 6px; font-size: 12px; pointer-events: none; opacity: 0; transition: opacity 0.15s; border: 1px solid rgba(100,100,180,0.3); max-width: 300px; word-wrap: break-word; }}
</style>
</head>
<body>
<div id="container">
  <svg></svg>
  <div id="info-panel">
    <h3>{title}</h3>
    <div class="stat">Files: <strong>{node_count}</strong></div>
    <div class="stat">Links: <strong>{edge_count}</strong></div>
  </div>
  <div class="legend">
    <div class="legend-item"><span class="legend-dot" style="background:#5b9bd5;"></span> Markdown File</div>
    <div class="legend-item"><span class="legend-dot" style="background:#ff6b6b;"></span> External / Absent</div>
  </div>
  <div class="tooltip" id="tooltip"></div>
</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const graphData = {graph_json};

const container = document.getElementById('container');
const svg = d3.select('svg');
const tooltip = d3.select('#tooltip');
const width = window.innerWidth;
const height = window.innerHeight;

const color = d3.scaleOrdinal()
  .domain(['internal','external'])
  .range(['#5b9bd5','#ff6b6b']);

const nodes = graphData.nodes.map(d => ({{...d}}));
const links = graphData.edges.map(d => ({{...d}}));
const nodeMap = new Map(nodes.map(n => [n.id, n]));

const link = svg.append('g')
  .attr('stroke', 'rgba(140,140,180,0.3)')
  .attr('stroke-width', 1.2)
  .selectAll('line')
  .data(links)
  .join('line');

const node = svg.append('g')
  .attr('stroke', 'rgba(255,255,255,0.15)')
  .attr('stroke-width', 1.5)
  .selectAll('circle')
  .data(nodes)
  .join('circle')
  .attr('r', d => d.weight ? 4 + d.weight * 3 : 5)
  .attr('fill', d => color(d.type))
  .call(d3.drag()
    .on('start', (event, d) => {{
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }})
    .on('drag', (event, d) => {{
      d.fx = event.x;
      d.fy = event.y;
    }})
    .on('end', (event, d) => {{
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }}));

node.on('mouseover', (event, d) => {{
  const connected = new Set();
  links.forEach(l => {{
    if (l.source.id === d.id || (l.target.id && l.target.id === d.id)) {{
      connected.add(l.source.id === d.id ? (l.target.id || l.target) : l.source.id);
    }}
  }});
  link.attr('stroke', l => connected.has(l.source.id) || connected.has(l.target.id || l.target) ? 'rgba(200,200,255,0.6)' : 'rgba(140,140,180,0.08)');
  node.attr('opacity', n => n.id === d.id || connected.has(n.id) ? 1 : 0.2);

  const pos = d3.pointer(event, container.node());
  tooltip.style('opacity', 1)
    .style('left', (pos[0] + 14) + 'px')
    .style('top', (pos[1] - 10) + 'px')
    .html(`<strong>${{d.name}}</strong><br>${{d.type === 'external' ? 'External link' : (d.path || '')}}`);
}}).on('mouseout', () => {{
  link.attr('stroke', 'rgba(140,140,180,0.3)');
  node.attr('opacity', 1);
  tooltip.style('opacity', 0);
}});

const label = svg.append('g')
  .selectAll('text')
  .data(nodes)
  .join('text')
  .text(d => d.name.length > 28 ? d.name.slice(0, 26) + '..' : d.name)
  .attr('font-size', 10)
  .attr('fill', '#b0b8d0')
  .attr('text-anchor', 'middle')
  .attr('dy', -10)
  .style('pointer-events', 'none')
  .style('user-select', 'none');

const simulation = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links).id(d => d.id).distance(100))
  .force('charge', d3.forceManyBody().strength(-200))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('collision', d3.forceCollide().radius(20))
  .on('tick', () => {{
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);
    node
      .attr('cx', d => d.x)
      .attr('cy', d => d.y);
    label
      .attr('x', d => d.x)
      .attr('y', d => d.y);
  }});

window.addEventListener('resize', () => {{
  const w = window.innerWidth;
  const h = window.innerHeight;
  simulation.force('center', d3.forceCenter(w / 2, h / 2));
  simulation.alpha(0.3).restart();
}});

const zoom = d3.zoom()
  .scaleExtent([0.1, 6])
  .on('zoom', (event) => {{
    svg.selectAll('g').attr('transform', event.transform);
  }});
svg.call(zoom);
</script>
</body>
</html>"""
