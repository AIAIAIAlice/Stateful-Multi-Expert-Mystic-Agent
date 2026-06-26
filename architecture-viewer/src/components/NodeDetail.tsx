import { useState } from 'react';
import type { NodeData, RouteOption, AgentInfo, EvalCriteria, ExecMode, KeyMetric } from '../types';

const categoryColors: Record<string, string> = {
  start_end: '#6366f1', input: '#3b82f6', router: '#eab308',
  processor: '#22c55e', expert: '#a855f7', evaluator: '#f97316',
  output: '#64748b', clarification: '#d97706',
};

const categoryLabels: Record<string, string> = {
  start_end: '\u8d77\u6b62\u8282\u70b9', input: '\u8f93\u5165\u5904\u7406',
  router: '\u8def\u7531\u5668', processor: '\u5904\u7406\u5668',
  expert: '\u4e13\u5bb6\u63a8\u7406', evaluator: '\u8bc4\u4f30\u5668',
  output: '\u8f93\u51fa\u5904\u7406', clarification: '\u6f84\u6e05\u8282\u70b9',
};

interface NodeDetailProps { nodeData: NodeData | null; nodeId?: string; onClose: () => void; }

function StepsPanel({ steps }: { steps: string[] }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {steps.map((step, i) => (
        <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
          <div style={{ minWidth: 20, height: 20, borderRadius: 10, background: '#6366f1', color: 'white', fontSize: 10, fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>{i + 1}</div>
          <div style={{ fontSize: 11, color: '#334155', lineHeight: 1.4, paddingTop: 2 }}>{step}</div>
        </div>
      ))}
    </div>
  );
}

function RoutesTable({ routes }: { routes: RouteOption[] }) {
  return (
    <div style={{ fontSize: 11 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: '4px 8px', alignItems: 'center' }}>
        <div style={{ fontWeight: 700, color: '#475569', borderBottom: '1px solid #e2e8f0', paddingBottom: 4 }}>{'\u6761\u4ef6'}</div>
        <div style={{ fontWeight: 700, color: '#475569', borderBottom: '1px solid #e2e8f0', paddingBottom: 4 }}>{'\u76ee\u6807'}</div>
        <div style={{ fontWeight: 700, color: '#475569', borderBottom: '1px solid #e2e8f0', paddingBottom: 4 }}>{'\u6807\u7b7e'}</div>
        {routes.map((r, i) => (
          <div key={i} style={{ display: 'contents' }}>
            <div style={{ color: '#334155', padding: '3px 0' }}>{r.condition}</div>
            <div style={{ color: '#6366f1', fontWeight: 600, padding: '3px 0' }}>{r.target}</div>
            <div style={{ background: '#f1f5f9', borderRadius: 4, padding: '2px 6px', fontSize: 10, fontWeight: 600, color: '#475569', textAlign: 'center' }}>{r.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ExecutionModesPanel({ modes }: { modes: ExecMode[] }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {modes.map((m, i) => (
        <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: '8px 10px', border: '1px solid #e2e8f0' }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: '#6366f1', marginBottom: 4 }}>{m.mode}</div>
          <div style={{ fontSize: 10, color: '#64748b', marginBottom: 4 }}>{m.condition}</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
            {m.activeNodes.map((n, j) => <span key={j} style={{ fontSize: 9, background: '#eef2ff', color: '#4338ca', padding: '1px 6px', borderRadius: 4, fontWeight: 600 }}>{n}</span>)}
          </div>
        </div>
      ))}
    </div>
  );
}

function AgentsPanel({ agents }: { agents: AgentInfo[] }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {agents.map((a, i) => (
        <div key={i} style={{ background: '#faf5ff', borderRadius: 8, padding: '8px 10px', border: '1px solid #e9d5ff' }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: '#7e22ce', marginBottom: 2 }}>{a.name}</div>
          <div style={{ fontSize: 10, color: '#6b21a8', fontStyle: 'italic', marginBottom: 4 }}>{a.role}</div>
          <div style={{ fontSize: 10, color: '#334155', lineHeight: 1.4, marginBottom: 4 }}>{a.systemPrompt}</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
            {a.contextKeys.map((k, j) => <span key={j} style={{ fontSize: 9, background: '#f3e8ff', color: '#7e22ce', padding: '1px 6px', borderRadius: 4 }}>{k}</span>)}
          </div>
        </div>
      ))}
    </div>
  );
}

function EvalCriteriaPanel({ criteria }: { criteria: EvalCriteria[] }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {criteria.map((c, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 6px', background: '#f8fafc', borderRadius: 6, border: '1px solid #f1f5f9' }}>
          <div style={{ minWidth: 28, fontSize: 9, fontWeight: 700, color: '#f97316', background: '#fff7ed', borderRadius: 4, padding: '1px 4px', textAlign: 'center' }}>{c.id}</div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: '#334155' }}>{c.name}</div>
            <div style={{ fontSize: 9, color: '#94a3b8' }}>{c.description}</div>
          </div>
          <div style={{ fontSize: 9, color: '#94a3b8', textAlign: 'right' }}>{'>'}{c.passThreshold}</div>
        </div>
      ))}
    </div>
  );
}

function KeyMetricsPanel({ metrics }: { metrics: KeyMetric[] }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
      {metrics.map((m, i) => (
        <div key={i} style={{ background: '#f8fafc', borderRadius: 6, padding: '6px 8px', border: '1px solid #f1f5f9' }}>
          <div style={{ fontSize: 9, color: '#94a3b8', marginBottom: 2 }}>{m.label}</div>
          <div style={{ fontSize: 11, fontWeight: 600, color: '#334155' }}>{m.value}</div>
        </div>
      ))}
    </div>
  );
}

function SchemaBlock({ label, content }: { label: string; content: string }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: '#475569', marginBottom: 4 }}>{label}</div>
      <pre style={{ fontSize: 10, color: '#334155', background: '#f8fafc', padding: 10, borderRadius: 6, overflowX: 'auto', lineHeight: 1.4, margin: 0, fontFamily: 'Consolas, Monaco, monospace' }}>{content}</pre>
    </div>
  );
}

function NodeSpecificPanel({ nodeId, data }: { nodeId: string; data: NodeData }) {
  switch (nodeId) {
    case 'inputNormalizer': return <StepsPanel steps={data.steps || []} />;
    case 'intentRouter': return <RoutesTable routes={data.routes || []} />;
    case 'plannerExecutor': return <ExecutionModesPanel modes={data.executionModes || []} />;
    case 'specialistSubgraph': return <AgentsPanel agents={data.agents || []} />;
    case 'criticEvaluator': return <EvalCriteriaPanel criteria={data.evalCriteria || []} />;
    default: return data.steps ? <StepsPanel steps={data.steps} /> : null;
  }
}

export default function NodeDetail({ nodeData, nodeId, onClose }: NodeDetailProps) {
  const [showDetails, setShowDetails] = useState(false);
  if (!nodeData) return null;
  const color = categoryColors[nodeData.category] || '#64748b';
  const categoryLabel = categoryLabels[nodeData.category] || '\u672a\u77e5';
  const hasDetail = !!(nodeData.inputSchema || nodeData.outputSchema || nodeData.promptTemplate || nodeData.exampleData);

  return (
    <div style={{ position: 'absolute', top: 16, right: 16, zIndex: 10, background: 'white', borderRadius: 12, boxShadow: '0 4px 24px rgba(0,0,0,0.12)', width: 380, maxHeight: 'calc(100vh - 32px)', overflowY: 'auto', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', animation: 'slideIn 0.2s ease-out' }}>
      <div style={{ padding: '16px 16px 12px', borderBottom: '1px solid #f1f5f9' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 8 }}>
          <div style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 4, background: color + '15', color: color, fontSize: 11, fontWeight: 600 }}>{categoryLabel}</div>
          <button onClick={onClose} style={{ background: '#f1f5f9', border: 'none', borderRadius: 6, width: 28, height: 28, fontSize: 16, cursor: 'pointer', color: '#64748b', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>X</button>
        </div>
        <div style={{ fontWeight: 700, fontSize: 16, color: '#0f172a', lineHeight: 1.3 }}>{nodeData.label.split('\n')[0]}</div>
        {nodeData.label.split('\n')[1] && <div style={{ fontSize: 13, color: '#64748b', marginTop: 2 }}>{nodeData.label.split('\n')[1]}</div>}
        {nodeData.techStack && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 4 }}>{nodeData.techStack}</div>}
        {nodeData.execOrder && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{nodeData.execOrder}</div>}
      </div>

      <div style={{ padding: '12px 16px' }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 6 }}>{'\u{1F4CB}'} {'\u529f\u80fd\u63cf\u8ff0'}</div>
        <div style={{ fontSize: 12, color: '#334155', lineHeight: 1.5, background: '#f8fafc', padding: 10, borderRadius: 8 }}>{nodeData.description}</div>
      </div>

      {nodeId && (
        <div style={{ padding: '0 16px 12px' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 6 }}>{'\u2699\uFE0F'} {'\u6838\u5fc3\u4fe1\u606f'}</div>
          <NodeSpecificPanel nodeId={nodeId} data={nodeData} />
        </div>
      )}

      <div style={{ padding: '0 16px 12px' }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 6 }}>{'\u{1F4E5}'} {'\u8f93\u5165'} {'\u2192'} {'\u{1F4E4}'} {'\u8f93\u51fa'}</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <div>
            {nodeData.inputs.map((input, i) => <div key={i} style={{ fontSize: 10, color: '#334155', padding: '4px 6px', background: '#eff6ff', borderRadius: 4, marginBottom: 3, lineHeight: 1.3 }}>{input}</div>)}
            {nodeData.inputs.length === 0 && <div style={{ fontSize: 10, color: '#94a3b8', fontStyle: 'italic' }}>{'\u65e0\u8f93\u5165'}</div>}
          </div>
          <div>
            {nodeData.outputs.map((output, i) => <div key={i} style={{ fontSize: 10, color: '#334155', padding: '4px 6px', background: '#f0fdf4', borderRadius: 4, marginBottom: 3, lineHeight: 1.3 }}>{output}</div>)}
            {nodeData.outputs.length === 0 && <div style={{ fontSize: 10, color: '#94a3b8', fontStyle: 'italic' }}>{'\u65e0\u8f93\u51fa'}</div>}
          </div>
        </div>
      </div>

      {nodeData.keyMetrics && nodeData.keyMetrics.length > 0 && (
        <div style={{ padding: '0 16px 12px' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 6 }}>{'\u{1F4CA}'} {'\u5173\u952e\u6307\u6807'}</div>
          <KeyMetricsPanel metrics={nodeData.keyMetrics} />
        </div>
      )}

      <div style={{ padding: '0 16px 12px' }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 6 }}>{'\u{1F527}'} {'\u5904\u7406\u903b\u8f91'}</div>
        <div style={{ fontSize: 11, color: '#334155', padding: 10, background: '#faf5ff', borderRadius: 8, lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>{nodeData.processingLogic}</div>
      </div>

      {hasDetail && (
        <div style={{ padding: '0 16px 16px' }}>
          <button onClick={() => setShowDetails(!showDetails)} style={{ width: '100%', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '8px 12px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 600, color: '#475569' }}>
            <span style={{ transform: showDetails ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.2s', display: 'inline-block' }}>{'\u25B6'}</span> {'\u8be6\u7ec6\u4fe1\u606f'}
          </button>
          {showDetails && (
            <div style={{ paddingTop: 12 }}>
              {nodeData.inputSchema && <SchemaBlock label={'\u8f93\u5165 Schema'} content={nodeData.inputSchema} />}
              {nodeData.outputSchema && <SchemaBlock label={'\u8f93\u51fa Schema'} content={nodeData.outputSchema} />}
              {nodeData.promptTemplate && <SchemaBlock label={'Prompt \u6a21\u677f'} content={nodeData.promptTemplate} />}
              {nodeData.exampleData && <SchemaBlock label={'\u793a\u4f8b\u6570\u636e'} content={nodeData.exampleData} />}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
