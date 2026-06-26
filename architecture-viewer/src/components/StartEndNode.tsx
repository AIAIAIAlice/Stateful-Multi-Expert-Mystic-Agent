import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { NodeData } from '../types';

export default function StartEndNode({ data }: NodeProps) {
  const nodeData = data as unknown as NodeData;
  const isStart = nodeData.label === 'BEGIN';

  return (
    <div
      style={{
        width: 56,
        height: 56,
        borderRadius: '50%',
        background: isStart ? 'linear-gradient(135deg, #6366f1, #818cf8)' : 'linear-gradient(135deg, #64748b, #94a3b8)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
        fontWeight: 800,
        fontSize: 11,
        letterSpacing: 0.5,
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        cursor: 'pointer',
      }}
      className="custom-node"
    >
      {!isStart && <Handle type="target" position={Position.Top} style={{ background: '#64748b', width: 8, height: 8, border: '2px solid white' }} />}
      <span style={{ marginTop: isStart ? 0 : -2 }}>{nodeData.label}</span>
      {isStart && <Handle type="source" position={Position.Bottom} style={{ background: '#6366f1', width: 8, height: 8, border: '2px solid white' }} />}
      {!isStart && <Handle type="source" position={Position.Bottom} style={{ background: '#64748b', width: 8, height: 8, border: '2px solid white' }} />}
    </div>
  );
}
