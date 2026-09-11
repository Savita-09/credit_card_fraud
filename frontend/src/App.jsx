import { useEffect, useRef, useState } from 'react';
import { Link, NavLink, Route, Routes, useLocation } from 'react-router-dom';
import { Activity, ArrowUpRight, BookOpen, ChartNoAxesCombined, CheckCircle2, ChevronRight, FileStack, LayoutDashboard, Menu, ScanLine, ShieldCheck, X } from 'lucide-react';
import { api, API_BASE } from './services/api';
import { useResource } from './hooks/useResource';
import { ErrorState, Loading } from './components/ui';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import Batch from './pages/Batch';
import Performance from './pages/Performance';

const navigation = [{ path: '/', label: 'Overview', icon: LayoutDashboard }, { path: '/predict', label: 'Single prediction', icon: ScanLine }, { path: '/batch', label: 'Batch analysis', icon: FileStack }, { path: '/performance', label: 'Model performance', icon: ChartNoAxesCombined }];

export default function App() {
  const resource = useResource('/model-info');
  const [healthy, setHealthy] = useState(null), [mobileOpen, setMobileOpen] = useState(false), [toast, setToast] = useState(null);
  const timer = useRef(null);
  const location = useLocation();
  const current = navigation.find(item => item.path === location.pathname)?.label || 'Workspace';
  useEffect(() => { let active = true; const check = () => api('/health').then(data => { if (active) setHealthy(data.status === 'healthy'); }).catch(() => { if (active) setHealthy(false); }); check(); const interval = setInterval(check, 30000); return () => { active = false; clearInterval(interval); clearTimeout(timer.current); }; }, []);
  useEffect(() => { setMobileOpen(false); window.scrollTo(0, 0); }, [location.pathname]);
  function notify(message) { clearTimeout(timer.current); setToast(message); timer.current = setTimeout(() => setToast(null), 5000); }
  return <div className="app-shell"><a className="skip-link" href="#main">Skip to content</a>{mobileOpen && <button className="sidebar-backdrop" aria-label="Close navigation" onClick={() => setMobileOpen(false)}/>}
    <aside id="main-sidebar" className={`sidebar ${mobileOpen ? 'open' : ''}`}><Link to="/" className="brand"><span className="brand-icon"><ShieldCheck size={25}/></span><span>sentinel<span className="brand-period">.</span><small>FRAUD INTELLIGENCE</small></span></Link><div className="workspace-selector"><span className="workspace-avatar">S</span><div><strong>Sentinel workspace</strong><small>Credit card fraud detection</small></div><ChevronRight size={14}/></div><div className="nav-label">WORKSPACE</div><nav aria-label="Main navigation">{navigation.map(({ path, label, icon: Icon }) => <NavLink key={path} to={path} end={path === '/'}><Icon size={19}/><span>{label}</span>{path === '/' && <span className="nav-active-dot"/>}</NavLink>)}</nav><div className="nav-divider"/><div className="nav-label">RESOURCES</div><a className="resource-link" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer"><BookOpen size={18}/> API documentation <ArrowUpRight size={14}/></a><div className="sidebar-bottom"><div className="model-status-card"><div><span className={`status-dot ${healthy === false ? 'offline' : ''}`}/><strong>{healthy === null ? 'Checking model…' : healthy ? 'Model is ready' : 'API unavailable'}</strong></div><p>{resource.data?.model_name || 'Connecting to inference service'}</p><Link to="/performance">View performance <ArrowUpRight size={13}/></Link></div><div className="workspace-footer"><span className="workspace-footer-icon"><ShieldCheck size={18}/></span><div><strong>Research workspace</strong><small>{resource.data?.dataset?.dataset_id ? 'Hugging Face benchmark' : 'Transaction benchmark'}</small></div></div></div></aside>
    <div className="main-shell"><header className="topbar"><div className="breadcrumbs"><button className="mobile-menu icon-button" aria-label="Open navigation" aria-controls="main-sidebar" aria-expanded={mobileOpen} onClick={() => setMobileOpen(true)}><Menu size={21}/></button><span>Workspace</span><ChevronRight size={13}/><strong>{current}</strong></div><div className="topbar-right"><span className={`health-pill ${healthy === false ? 'unhealthy' : ''}`}><span className="status-dot"/>{healthy === null ? 'Connecting' : healthy ? 'System healthy' : 'API offline'}</span><div className="topbar-divider"/><span className="topbar-date">{new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', year: 'numeric' }).format(new Date())}</span><span className="avatar">SL</span></div></header>
    <main id="main" className="main-content">{resource.loading ? <Loading/> : resource.error ? <><h1>Let’s connect your model.</h1><p className="lead">Start the FastAPI backend and verify that the trained model is ready.</p><ErrorState error={resource.error} retry={resource.refresh}/><code className="startup-code">python -m uvicorn backend.main:app --port 8000</code></> : <Routes><Route path="/" element={<Dashboard model={resource.data}/>}/><Route path="/predict" element={<Predict model={resource.data} notify={notify}/>}/><Route path="/batch" element={<Batch model={resource.data} notify={notify}/>}/><Route path="/performance" element={<Performance model={resource.data}/>}/><Route path="*" element={<div className="state"><h1>Page not found</h1><Link className="button primary" to="/">Return to overview</Link></div>}/></Routes>}</main></div>
    {toast && <div className="toast" role="status"><CheckCircle2 size={19}/><span>{toast}</span><button className="icon-button" aria-label="Dismiss notification" onClick={() => setToast(null)}><X size={16}/></button></div>}
  </div>;
}
