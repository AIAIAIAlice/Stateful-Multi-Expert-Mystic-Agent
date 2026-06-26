import type { Route } from '../types';

interface RoutePanelProps {
  routes: Route[];
  activeRoute: string | null;
  onSelectRoute: (routeId: string | null) => void;
}

export default function RoutePanel({ routes, activeRoute, onSelectRoute }: RoutePanelProps) {
  return (
    <div style={{
      position: 'absolute',
      top: 16,
      left: 16,
      zIndex: 10,
      background: 'white',
      borderRadius: 12,
      boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
      padding: '12px',
      width: 260,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: 16 }}>🛤️</span>
        路由路径切换
      </div>

      <button
        onClick={() => onSelectRoute(null)}
        style={{
          width: '100%',
          padding: '8px 12px',
          marginBottom: 4,
          borderRadius: 8,
          border: !activeRoute ? '2px solid #6366f1' : '1px solid #e2e8f0',
          background: !activeRoute ? '#eef2ff' : 'white',
          color: '#1e293b',
          fontWeight: !activeRoute ? 600 : 400,
          fontSize: 12,
          cursor: 'pointer',
          textAlign: 'left',
          transition: 'all 0.15s ease',
        }}
      >
        🏠 显示全部
      </button>

      {routes.map((route) => (
        <button
          key={route.id}
          onClick={() => onSelectRoute(route.id)}
          style={{
            width: '100%',
            padding: '8px 12px',
            marginBottom: 4,
            borderRadius: 8,
            border: activeRoute === route.id ? '2px solid #6366f1' : '1px solid #e2e8f0',
            background: activeRoute === route.id ? '#eef2ff' : 'white',
            color: '#1e293b',
            fontWeight: activeRoute === route.id ? 600 : 400,
            fontSize: 12,
            cursor: 'pointer',
            textAlign: 'left',
            transition: 'all 0.15s ease',
          }}
          title={route.description}
        >
          <div>{route.name}</div>
          <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{route.nameEn}</div>
        </button>
      ))}

      {activeRoute && (
        <div style={{
          marginTop: 8,
          padding: '8px 10px',
          background: '#f8fafc',
          borderRadius: 8,
          fontSize: 11,
          color: '#64748b',
          lineHeight: 1.4,
        }}>
          {routes.find(r => r.id === activeRoute)?.description}
        </div>
      )}
    </div>
  );
}
