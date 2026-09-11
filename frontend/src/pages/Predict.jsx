import { useState } from 'react';
import { CheckCheck, ChevronDown, FlaskConical, LoaderCircle, Play, RotateCcw, ShieldAlert, SlidersHorizontal } from 'lucide-react';
import { api, percent } from '../services/api';
import { Badge, Card, Empty, ErrorState, PageHeading, Threshold } from '../components/ui';

export default function Predict({ model, notify }) {
  const defaults = Object.fromEntries(model.features.map(f => [f.name, f.default]));
  const [values, setValues] = useState(defaults);
  const [threshold, setThreshold] = useState(model.active_threshold);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [example, setExample] = useState(null);
  const [loadingExample, setLoadingExample] = useState(false);
  const standard = model.features.filter(f => ['Amount', 'Time'].includes(f.name));
  const remaining = model.features.filter(f => !['Amount', 'Time'].includes(f.name));
  function change(name, value) { setValues(previous => ({ ...previous, [name]: value })); setResult(null); setExample(null); }
  async function loadExample(kind) { setError(null); setLoadingExample(true); try { const data = await api('/examples'); setValues(data.transactions[kind]); setExample(kind); setResult(null); notify('Training example loaded. Review the values before scoring.'); } catch (error) { setError(error.message); } finally { setLoadingExample(false); } }
  async function submit(event) {
    event.preventDefault(); setBusy(true); setError(null); setResult(null);
    try {
      const features = Object.fromEntries(model.features.map(f => [f.name, values[f.name] === '' ? null : f.type === 'number' ? Number(values[f.name]) : values[f.name]]));
      const data = await api('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ features, threshold, explain: true }) });
      setResult(data); notify('Transaction analyzed and added to live predictions.');
    } catch (error) { setError(error.message); } finally { setBusy(false); }
  }
  const featureLabels = { category: 'Merchant category', state: 'State', city_pop: 'City population', lat: 'Cardholder latitude', long: 'Cardholder longitude', merch_lat: 'Merchant latitude', merch_long: 'Merchant longitude' };
  const field = feature => <label className="field" key={feature.name}><span>{feature.label || (feature.name === 'Amount' ? `Transaction amount (${model.dataset.amount_unit || 'dataset units'})` : feature.name === 'Time' ? 'Elapsed time (seconds)' : featureLabels[feature.name] || feature.name)}</span><input disabled={busy || loadingExample} type={feature.type === 'number' ? 'number' : 'text'} step="any" min={['Amount', 'Time', 'city_pop'].includes(feature.name) ? '0' : undefined} value={values[feature.name] ?? ''} onChange={event => change(feature.name, event.target.value)} placeholder="Empty = training imputation" list={feature.type === 'category' ? `values-${feature.name}` : undefined} maxLength={feature.type === 'category' ? 200 : undefined}/>{feature.type === 'category' && <datalist id={`values-${feature.name}`}>{feature.categories.map(value => <option key={value} value={value}/>)}</datalist>}{feature.name === 'Time' && <small>{model.dataset.time_origin ? `Seconds since ${model.dataset.time_origin.replace('T', ' ')} in the source timestamps. Timezone is unspecified.` : 'Seconds since the dataset’s first transaction.'}</small>}</label>;
  return <><PageHeading eyebrow="TRANSACTION INTELLIGENCE" title="Every transaction tells a story." text="Score a transaction using the same transformations as the trained model."/>
    <div className="prediction-layout"><div><form onSubmit={submit}><Card title="Transaction details" subtitle={`${model.features.length} input features · generated from the trained schema`} action={<SlidersHorizontal size={18} className="muted"/>}>
      <div className="example-bar"><FlaskConical size={17}/><span>Try a benchmark example</span><button type="button" disabled={loadingExample || busy} onClick={() => loadExample('legitimate')}>Legitimate</button><button type="button" disabled={loadingExample || busy} onClick={() => loadExample('fraud')}>Fraud</button></div>
      {example && <div className="notice">Loaded a labeled {example} example from training. This is a demonstration, not an independent test.</div>}
      <div className="form-grid">{standard.map(field)}</div>
      <details className="feature-details" open><summary>Transaction context <span>{remaining.length} features <ChevronDown size={14}/></span></summary><p className="caption">{model.dataset.feature_note || 'Enter the feature values required by the saved model.'}</p><div className="feature-grid">{remaining.map(field)}</div></details>
      <div className="form-actions"><button type="button" className="button secondary" disabled={busy} onClick={() => { setValues(defaults); setExample(null); setResult(null); setError(null); }}><RotateCcw size={15}/> Reset values</button><button disabled={busy || loadingExample} type="submit" className="button primary">{busy ? <LoaderCircle size={16} className="spin"/> : <Play size={15}/>} {busy ? 'Analyzing…' : 'Analyze transaction'}</button></div>
    </Card></form><Card title="Decision threshold" subtitle="Control how sensitive the fraud flag is"><Threshold disabled={busy || loadingExample} value={threshold} onChange={value => { setThreshold(value); setResult(null); }} defaultValue={model.active_threshold}/></Card></div>
    <div className="prediction-result-column">{error && <ErrorState error={error}/>}{result ? <section className={`result-card ${result.prediction === 'Fraud' ? 'fraud-result' : ''}`} aria-live="polite"><div className="result-symbol">{result.prediction === 'Fraud' ? <ShieldAlert size={29}/> : <CheckCheck size={29}/>}</div><span className="eyebrow">MODEL DECISION</span><h2>{result.prediction === 'Fraud' ? 'Fraud flag' : 'Legitimate'}</h2><Badge value={`${result.risk_level}`}/><div className="result-probability"><strong>{percent(result.fraud_probability)}</strong><span>Fraud probability score</span></div><div className="result-track"><i style={{ width: `${result.fraud_probability * 100}%` }}/><b style={{ left: `${result.threshold * 100}%` }}/></div><p className="caption">Classification threshold: {result.threshold.toFixed(4)}</p><div className="recommendation"><strong>Recommended next step</strong><p>{result.recommendation}</p></div><dl className="result-meta"><div><dt>Model</dt><dd>{result.model_name}</dd></div><div><dt>Model version</dt><dd>{result.model_version}</dd></div></dl><div className="signals"><h3>What influenced this score?</h3><p className="caption">Score changes when each feature is replaced with its training median or mode. Positive signals raised this score relative to that reference.</p>{result.signals.map(signal => <div className="signal-row" key={signal.feature}><span>{signal.feature}</span><div className="signal-track"><i style={{ width: `${Math.min(100, Math.abs(signal.score_delta) * 100)}%`, background: signal.score_delta > 0 ? '#e3909c' : '#9980db' }}/></div><strong>{signal.score_delta >= 0 ? '+' : ''}{(signal.score_delta * 100).toFixed(2)} pp</strong></div>)}<p className="caption">Sensitivity signals are non-additive and non-causal. They are not proof of fraud.</p></div></section> : <Card title="Prediction result"><Empty title="Ready when you are" text="Enter transaction features or load a benchmark example, then analyze the transaction."/></Card>}
    <div className="info-note"><h3>A score, with context.</h3><p>{model.probability_note}</p><div className="risk-guide"><span><Badge value="Low"/> &lt; {model.risk_bands.medium}</span><span><Badge value="Medium"/> {model.risk_bands.medium}–{model.risk_bands.high}</span><span><Badge value="High"/> ≥ {model.risk_bands.high}</span></div></div></div></div>
  </>;
}
