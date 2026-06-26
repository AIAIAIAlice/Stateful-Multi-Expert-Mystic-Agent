import type { Node, Edge } from '@xyflow/react';

export type NodeCategory =
  | 'start_end'
  | 'input'
  | 'router'
  | 'processor'
  | 'expert'
  | 'evaluator'
  | 'output'
  | 'clarification';

export interface RouteOption {
  condition: string;
  target: string;
  label: string;
}

export interface AgentInfo {
  name: string;
  role: string;
  systemPrompt: string;
  contextKeys: string[];
}

export interface EvalCriteria {
  id: string;
  name: string;
  description: string;
  passThreshold: number;
}

export interface ExecMode {
  mode: string;
  condition: string;
  activeNodes: string[];
}

export interface KeyMetric {
  label: string;
  value: string;
}

export interface NodeData extends Record<string, unknown> {
  label: string;
  category: NodeCategory;
  description: string;
  inputs: string[];
  outputs: string[];
  processingLogic: string;
  techStack?: string;
  execOrder?: string;
  steps?: string[];
  routes?: RouteOption[];
  agents?: AgentInfo[];
  evalCriteria?: EvalCriteria[];
  executionModes?: ExecMode[];
  inputSchema?: string;
  outputSchema?: string;
  promptTemplate?: string;
  exampleData?: string;
  keyMetrics?: KeyMetric[];
}

export type GraphNode = Node<NodeData>;
export type GraphEdge = Edge;

export interface Route {
  id: string;
  name: string;
  nameEn: string;
  description: string;
  nodeIds: string[];
  edgeIds: string[];
}
