import { useState, useCallback } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import CustomNode from './components/CustomNode';
import StartEndNode from './components/StartEndNode';
import ParallelGroup from './components/ParallelGroup';
import RoutePanel from './components/RoutePanel';
import NodeDetail from './components/NodeDetail';
import { initialNodes, initialEdges, routes } from './data/graphData';
import { savedLayout } from './data/savedLayout';
import type { NodeData, GraphNode } from './types';
import type { Node, NodeMouseHandler } from '@xyflow/react';

const nodeTypes = {
  custom: CustomNode,
  startEnd: StartEndNode,
  parallelGroup: ParallelGroup,
};

// --- Apply saved layout to nodes ---
const layoutedNodes: GraphNode[] = initialNodes.map(node => {
  const saved = savedLayout[node.id];
  if (saved) {
    const updatedNode = {
      ...node,
      position: { x: saved.x, y: saved.y },
    };
    if (saved.width && saved.height) {
      updatedNode.style = {
        ...node.style,
        width: saved.width,
        height: saved.height,
      };
    }
    return updatedNode;
  }
  return node;
});

// --- Build parallel group from saved positions ---
const parallelIds = ['symbolicCalculator', 'domainRAG'];
const PAD = 20;
const parallelNodes = layoutedNodes.filter(n => parallelIds.includes(n.id));
const gMinX = Math.min(...parallelNodes.map(n => n.position.x));
const gMinY = Math.min(...parallelNodes.map(n => n.position.y));
const gMaxX = Math.max(...parallelNodes.map(n => n.position.x + (savedLayout[n.id]?.width || 200)));
const gMaxY = Math.max(...parallelNodes.map(n => n.position.y + (savedLayout[n.id]?.height || 70)));

const groupNodes: GraphNode[] = [{
  id: 'parallelGroup_0',
  type: 'parallelGroup',
  position: { x: gMinX - PAD, y: gMinY - PAD },
  data: {
    label: 'Parallel Execution',
    category: 'expert' as const,
    description: '',
    inputs: [],
    outputs: [],
    processingLogic: '',
  },
  style: {
    width: gMaxX - gMinX + PAD * 2,
    height: gMaxY - gMinY + PAD * 2,
    zIndex: -1,
  },
}];

const allNodes = [...groupNodes, ...layoutedNodes];

// --- Edges (unchanged) ---
const layoutedEdges = initialEdges;

// --- Color maps ---
const nodeColorMap: Record<string, string> = {
  start_end: '#6366f1', input: '#3b82f6', router: '#eab308',
  processor: '#22c55e', expert: '#a855f7', evaluator: '#f97316',
  output: '#64748b', clarification: '#d97706',
};
const miniMapNodeColor = (node: Node) => {
  const cat = (node.data as Record<string, unknown>)?.category;
  return nodeColorMap[cat as string] || '#94a3b8';
};

// --- SVG Arrow Markers ---
const markerDefs = (
  <svg style={{ position: 'absolute', width: 0, height: 0 }}>
    <defs>
      <marker id="arrowDefault" viewBox="0 0 20 20" refX="20" refY="10" markerWidth="12" markerHeight="12" orient="auto-start-reverse">
        <path d="M 0 0 L 20 10 L 0 20 z" fill="#334155" />
      </marker>
      <marker id="arrowRed" viewBox="0 0 20 20" refX="20" refY="10" markerWidth="12" markerHeight="12" orient="auto-start-reverse">
        <path d="M 0 0 L 20 10 L 0 20 z" fill="#ef4444" />
      </marker>
      <marker id="arrowGreen" viewBox="0 0 20 20" refX="20" refY="10" markerWidth="12" markerHeight="12" orient="auto-start-reverse">
        <path d="M 0 0 L 20 10 L 0 20 z" fill="#22c55e" />
      </marker>
      <marker id="arrowAmber" viewBox="0 0 20 20" refX="20" refY="10" markerWidth="12" markerHeight="12" orient="auto-start-reverse">
        <path d="M 0 0 L 20 10 L 0 20 z" fill="#f59e0b" />
      </marker>
    </defs>
  </svg>
);

// --- Component ---
export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState(allNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges);
  const [activeRoute, setActiveRoute] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<NodeData | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleSelectRoute = useCallback((routeId: string | null) => {
    setActiveRoute(routeId);
    if (!routeId) {
      setNodes(allNodes.map(n => ({ ...n, style: { ...n.style, opacity: 1 } })));
      setEdges(layoutedEdges.map(e => ({ ...e, style: { ...e.style, opacity: 1 } })));
      return;
    }
    const route = routes.find(r => r.id === routeId);
    if (!route) return;
    const activeNodeIdSet = new Set(route.nodeIds);
    const activeEdgeIdSet = new Set(route.edgeIds);
    const usesParallel = route.nodeIds.includes('symbolicCalculator') && route.nodeIds.includes('domainRAG');
    setNodes(allNodes.map(n => {
      const isGroup = n.type === 'parallelGroup';
      const isActive = isGroup ? usesParallel : activeNodeIdSet.has(n.id);
      return { ...n, style: { ...n.style, opacity: isActive ? 1 : 0.08, filter: isActive ? 'none' : 'grayscale(100%)' } };
    }));
    setEdges(layoutedEdges.map(e => ({ ...e, style: { ...e.style, opacity: activeEdgeIdSet.has(e.id) ? 1 : 0.06 } })));
  }, [setNodes, setEdges]);

  const onNodeClick: NodeMouseHandler = useCallback((_, node) => {
    if (node.type === 'parallelGroup') return;
    setSelectedNode(node.data as unknown as NodeData);
    setSelectedNodeId(node.id);
  }, []);
  const onPaneClick = useCallback(() => { setSelectedNode(null); setSelectedNodeId(null); }, []);

  // --- Export layout to clipboard ---
  const handleExportLayout = useCallback(() => {
    const positions: Record<string, { x: number; y: number; width?: number; height?: number }> = {};
    nodes.forEach(n => {
      if (n.type !== 'parallelGroup') {
        positions[n.id] = {
          x: Math.round(n.position.x),
          y: Math.round(n.position.y),
          width: n.measured?.width,
          height: n.measured?.height,
        };
      }
    });
    const json = JSON.stringify(positions, null, 2);
    navigator.clipboard.writeText(json).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [nodes]);

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      {markerDefs}

      {/* Title bar */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 48,
        background: 'white', borderBottom: '1px solid #e2e8f0',
        display: 'flex', alignItems: 'center', paddingLeft: 288, zIndex: 20,
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      }}>
        <span style={{ fontSize: 16, fontWeight: 700, color: '#0f172a' }}>YHJ LangGraph Architecture</span>
        <span style={{ fontSize: 11, color: '#94a3b8', marginLeft: 12 }}>
          Drag nodes to adjust layout | Click node for details | Resize by selecting a node
        </span>
      </div>

      <ReactFlow
        nodes={nodes} edges={edges}
        onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick} onPaneClick={onPaneClick}
        nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.3 }}
        minZoom={0.1} maxZoom={2} proOptions={{ hideAttribution: true }}
        style={{ background: '#f8fafc', paddingTop: 48 }}
        defaultEdgeOptions={{
          type: 'smoothstep',
          style: { strokeWidth: 1.5 },
          markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16, color: '#334155' },
        }}
      >
        <Background color="#e2e8f0" gap={20} size={1} />
        <Controls position="bottom-right" style={{ background: 'white', borderRadius: 8, boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }} />
        <MiniMap nodeColor={miniMapNodeColor} nodeStrokeWidth={0} maskColor="rgba(0,0,0,0.05)"
          style={{ position: 'absolute', bottom: 16, right: 80, background: 'white', borderRadius: 8, boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }} />
      </ReactFlow>

      {/* Export button */}
      <button
        onClick={handleExportLayout}
        style={{
          position: 'absolute', top: 60, right: 16, zIndex: 20,
          background: copied ? '#22c55e' : '#6366f1',
          color: 'white', border: 'none', borderRadius: 8,
          padding: '8px 16px', fontSize: 12, fontWeight: 600,
          cursor: 'pointer', boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          transition: 'background 0.2s',
        }}
      >
        {copied ? 'Copied!' : 'Export Layout'}
      </button>

      <RoutePanel routes={routes} activeRoute={activeRoute} onSelectRoute={handleSelectRoute} />
      <NodeDetail nodeData={selectedNode} nodeId={selectedNodeId ?? undefined} onClose={() => { setSelectedNode(null); setSelectedNodeId(null); }} />
    </div>
  );
}
