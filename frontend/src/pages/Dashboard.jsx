import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, ArrowRight, ArrowUpRight, CheckCheck, Database, Download, Layers3, RefreshCw, ShieldAlert, Sparkles } from 'lucide-react';
import { useResource } from '../hooks/useResource';
import { Card, Empty, ErrorState, Loading, PageHeading, PredictionTable, Stat } from '../components/ui';
import { ActivityChart, ClassificationChart, DistributionChart } from '../components/charts';
import { downloadText, number, percent } from '../services/api';

export default function Dashboard({ model }) {
  const [scope, setScope] = useState('evaluation');
  const [distribution, setDistribution] = useState('probability');
  const resource = useResource(`/statistics?scope=${scope}`);
  const data = resource.data;
  return <>
    <PageHeading eyebrow="YOUR RISK WORKSPACE" title="Fraud, in focus." text="Understand transaction risk. Make more informed decisions.">
      <button className="button secondary" disabled={!data || resource.loading} onClick={() => downloadText(JSON.stringify(data, null, 2), `sentinel-${scope}-report.json`, 'application/json')}><Download size={16}/> Export report</button>
      <Link className="button primary" to="/batch"><Sparkles size={16}/> Analyze transactions</Link>
    </PageHeading>
    <div className="view-toolbar"><div className="segmented" aria-label="Data source"><button className={scope === 'evaluation' ? 'active' : ''} onClick={() => setScope('evaluation')}><Database size={14}/> Evaluation snapshot</button><button className={scope === 'live' ? 'active' : ''} onClick={() => setScope('live')}><Activity size={14}/> Live predictions</button></div><span className="source-caption">{scope === 'evaluation' ? 'Held-out test data · no live traffic' : 'Stored API predictions · current model'}<button className="icon-button" aria-label="Refresh statistics" onClick={resource.refresh}><RefreshCw size={14}/></button></span></div>
    {resource.loading ? <Loading text="Loading risk analytics…"/> : resource.error ? <ErrorState error={resource.error} retry={resource.refresh}/> : data && <>
      <div className="stats-grid"><Stat label="Transactions analyzed" value={number(data.summary.total)} detail={scope === 'evaluation' ? 'Unseen test transactions' : 'Scored through the API'} icon={Layers3}/><Stat label="Fraud flags" value={number(data.summary.fraud)} detail="Flagged for further review" icon={ShieldAlert} variant="rose"/><Stat label="Legitimate predictions" value={number(data.summary.legitimate)} detail="Below classification threshold" icon={CheckCheck} variant="green"/><Stat label="Flag rate" value={`${data.summary.fraud_percentage.toFixed(2)}%`} detail="Share predicted as fraud" icon={Activity}/></div>
      {!data.summary.total ? <Card title="Your live workspace is ready"><Empty action={<Link className="button primary" to="/predict">Score your first transaction <ArrowRight size={16}/></Link>}/></Card> : <>
        <div className="overview-grid"><Card title={scope === 'evaluation' ? 'Transaction activity' : 'Fraud score distribution'} subtitle={scope === 'evaluation' ? (model.dataset.time_origin ? 'Held-out transaction volume by source date' : 'Held-out transaction volume across elapsed hours') : 'All stored scores for the active model'} action={<span className="chart-key"><i/>Transactions</span>}>{scope === 'evaluation' && data.timeline.length ? <ActivityChart data={data.timeline}/> : <DistributionChart data={data.probability_distribution}/>}</Card><Card title="Classification overview" subtitle="Predicted decisions, not confirmed labels"><ClassificationChart summary={data.summary}/></Card></div>
        <div className="analytics-grid"><Card title="Score & amount distribution" subtitle={distribution === 'probability' ? 'Model score from 0 (low) to 1 (high)' : `Transaction amounts in ${model.dataset.amount_unit || 'dataset units'}`} action={<select aria-label="Distribution type" className="small-select" value={distribution} onChange={e => setDistribution(e.target.value)}><option value="probability">Fraud score</option><option value="amount">Amount</option></select>}><DistributionChart kind={distribution} data={data[`${distribution}_distribution`]}/></Card><Card title="Risk bands" subtitle={`Low < ${data.risk_bands?.medium ?? .3} · High ≥ ${data.risk_bands?.high ?? .7}`}><DistributionChart kind="risk" data={data.risk_distribution}/></Card><div className="model-spotlight"><div className="spotlight-icon"><Sparkles size={20}/></div><span className="eyebrow">BUILT ON MEASURED RESULTS</span><h2>A clearer signal.<br/>A stronger defense.</h2><p>{model.model_name} selected by validation {model.selection_metric === 'f2' ? 'F2 to prioritize fraud recall' : 'average precision'}.</p><div className="spotlight-metric"><strong>{percent(model.selection_metric === 'f2' ? model.metrics.recall : model.metrics.pr_auc, 1)}</strong><span>{model.selection_metric === 'f2' ? 'Test fraud recall' : 'Test PR-AUC'}<br/>{model.selection_metric === 'f2' ? '(fraud cases detected)' : '(average precision)'}</span></div><Link to="/performance">Explore model performance <ArrowUpRight size={16}/></Link></div></div>
        <Card title={scope === 'evaluation' ? 'A closer look at predictions' : 'Recent predictions'} subtitle={scope === 'evaluation' ? '12 sample rows from the held-out test set · actual labels shown for comparison' : 'Latest scores for the active model · labels are predictions'} action={<Link className="text-button" to="/batch">Analyze a batch <ArrowRight size={14}/></Link>} className="table-card"><PredictionTable currency={model.dataset.currency} rows={data.recent} evaluation={scope === 'evaluation'}/></Card>
      </>}
      <div className="footnote"><span><Database size={13}/> {number(model.dataset.rows)} source transactions · {model.dataset.name || 'Transaction benchmark'}</span><span>Model flags support review. They do not establish fraud.</span></div>
    </>}
  </>;
}
