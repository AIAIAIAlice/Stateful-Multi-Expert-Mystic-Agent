import type { NodeProps } from '@xyflow/react';
import type { NodeData } from '../types';

export default function ParallelGroup({ data }: NodeProps) {
  const nodeData = data as unknown as NodeData;

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        background: 'rgba(168, 85, 247, 0.03)',
        border: '2px dashed rgba(168, 85, 247, 0.4)',
        borderRadius: 16,
        position: 'relative',
        pointerEvents: 'none',
      }}
    >
      {/* Top-left label with icon */}
      <div style={{
        position: 'absolute',
        top: -14,
        left: 16,
        background: 'white',
        padding: '3px 12px',
        borderRadius: 8,
        fontSize: 11,
        fontWeight: 700,
        color: '#7e22ce',
        border: '1.5px solid rgba(168, 85, 247, 0.3)',
        whiteSpace: 'nowrap',
        lineHeight: '18px',
        display: 'flex',
        alignItems: 'center',
        gap: 5,
        boxShadow: '0 1px 4px rgba(168, 85, 247, 0.1)',
      }}>
        <span style={{ fontSize: 12 }}>◆</span>
        {nodeData.label}
      </div>

      {/* Bottom-right annotation */}
      <div style={{
        position: 'absolute',
        bottom: 8,
        right: 12,
        fontSize: 9,
        color: 'rgba(126, 34, 206, 0.5)',
        fontStyle: 'italic',
      }}>
        concurrent execution
      </div>
    </div>
  );
}
