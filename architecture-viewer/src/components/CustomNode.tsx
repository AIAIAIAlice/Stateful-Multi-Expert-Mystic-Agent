import { Handle, NodeResizer, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { NodeData } from '../types';

const categoryColors: Record<string, { bg: string; border: string; text: string }> = {
  start_end: { bg: '#f0f4ff', border: '#6366f1', text: '#4338ca' },
  input: { bg: '#eff6ff', border: '#3b82f6', text: '#1d4ed8' },
  router: { bg: '#fefce8', border: '#eab308', text: '#a16207' },
  processor: { bg: '#f0fdf4', border: '#22c55e', text: '#15803d' },
  expert: { bg: '#faf5ff', border: '#a855f7', text: '#7e22ce' },
  evaluator: { bg: '#fff7ed', border: '#f97316', text: '#c2410c' },
  output: { bg: '#f8fafc', border: '#64748b', text: '#334155' },
  clarification: { bg: '#fef3c7', border: '#d97706', text: '#92400e' },
};

const categoryIcons: Record<string, string> = {
  start_end: '\u25cf',
  input: '\u2b07',
  router: '\u2195',
  processor: '\u2699',
  expert: '\u2728',
  evaluator: '\u2315',
  output: '\u2b06',
  clarification: '\u2753',
};

export default function CustomNode({ data, selected }: NodeProps) {
  const nodeData = data as unknown as NodeData;
  const colors = categoryColors[nodeData.category] || categoryColors.processor;
  const icon = categoryIcons[nodeData.category] || '\u25a3';

  return (
    <div
      style={{
        background: colors.bg,
        border: '2px solid ' + colors.border,
        borderRadius: 10,
        padding: '10px 14px',
        width: '100%',
        height: '100%',
        boxShadow: selected
          ? '0 0 0 2px ' + colors.border + ', 0 4px 16px rgba(0,0,0,0.12)'
          : '0 2px 8px rgba(0,0,0,0.08)',
        cursor: 'pointer',
        transition: 'box-shadow 0.2s ease',
        position: 'relative',
        overflow: 'visible',
      }}
      className="custom-node"
    >
      <NodeResizer
        isVisible={selected}
        minWidth={160}
        minHeight={60}
        maxWidth={400}
        maxHeight={200}
        handleStyle={{
          width: 8,
          height: 8,
          background: colors.border,
          border: '2px solid white',
          borderRadius: 2,
        }}
        color={colors.border}
      />
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: colors.border, width: 8, height: 8, border: '2px solid white' }}
      />
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <span style={{ fontSize: 16 }}>{icon}</span>
        <div style={{ fontWeight: 700, fontSize: 13, color: colors.text, lineHeight: 1.2 }}>
          {nodeData.label.split('\n').map((line: string, i: number) => (
            <span key={i}>
              {line}
              {i < nodeData.label.split('\n').length - 1 && <br />}
            </span>
          ))}
        </div>
      </div>
      <div style={{ fontSize: 10, color: '#64748b', lineHeight: 1.3, wordBreak: 'break-word' }}>
        {nodeData.description}
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: colors.border, width: 8, height: 8, border: '2px solid white' }}
      />
    </div>
  );
}
