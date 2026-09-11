"""Build and execute notebooks for the dataset recorded in the active model metadata."""
import argparse
import json
import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
kernel_root = ROOT / ".tmp" / "jupyter"
kernel = kernel_root / "kernels" / "sentinel"
kernel.mkdir(parents=True, exist_ok=True)
(kernel / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Sentinel (.venv)", "language": "python"}))
os.environ["JUPYTER_PATH"] = str(kernel_root)
os.environ["IPYTHONDIR"] = str(ROOT / ".tmp" / "ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".tmp" / "jupyter-runtime")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".tmp" / "matplotlib")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--only", nargs="*", default=[])
selected = parser.parse_args().only


def build(name, cells):
    if selected and name not in selected:
        return
    notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(value) if kind == "md" else nbf.v4.new_code_cell(value) for kind, value in cells])
    notebook.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(notebook, timeout=240, kernel_name="sentinel", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(notebook, ROOT / "notebooks" / name)
    print("EXECUTED", name, flush=True)


COMMON = """from pathlib import Path
import json, numpy as np, pandas as pd
from IPython.display import display, Image, Markdown
from src.dataset import load_training_frame
from src.data_processing import inspect_dataset, split_dataset, make_schema, validate_features
metadata = json.loads(Path('models/model_metadata.json').read_text())
frame, target, source = load_training_frame(Path('data/raw') / metadata['dataset']['file'])
train, validation, test = split_dataset(frame, target, metadata['training']['seed'], metadata['dataset'].get('split_strategy', 'stratified'))
features = [f['name'] for f in metadata['features']]
"""

build("01_eda.ipynb", [
("md", "# 01 · Dataset provenance and exploratory analysis\n\nThis notebook uses the source recorded in the active model. The Hugging Face dataset is a public benchmark; its minimal card does not document how transactions were collected. Names, card numbers and identifiers are excluded. Source counts describe integrity; feature plots use training rows only."),
("code", COMMON + """
profile = inspect_dataset(frame, target)
print('Dataset:', source.get('dataset_id', source['name']))
print('Revision:', source.get('revision'))
print('SHA-256:', metadata['dataset']['sha256'])
print({key: profile[key] for key in ['rows', 'columns', 'class_counts', 'fraud_percentage', 'duplicate_rows']})
print('Original columns:', metadata['dataset'].get('raw_columns', list(frame)))
print('Excluded columns:', metadata['dataset'].get('excluded_columns', []))
display(pd.DataFrame({'dtype': frame.dtypes.astype(str), 'missing': frame.isna().sum()}))
"""),
("md", "## Class imbalance\nAn always-legitimate classifier can appear accurate while detecting no fraud. Report fraud-class recall, precision, F2, average precision and confusion counts."),
("code", """print('Always-legitimate accuracy:', 1 - frame[target].mean())
display(Image(filename='reports/eda/class_distribution.png'))
display(pd.DataFrame([{'partition': key, **value} for key, value in metadata['dataset']['splits'].items()]))
"""),
("md", "## Amount and time\nAmounts use source units; a currency is not assumed when the dataset card does not specify one. Time is measured from the recorded origin, without assuming a verified local timezone. Class densities are normalized."),
("code", """print('Amount unit:', metadata['dataset'].get('amount_unit'))
print('Time origin:', metadata['dataset'].get('time_origin'))
display(train.groupby(target)['Amount'].describe())
display(Image(filename='reports/eda/amount_distributions.png'))
display(Image(filename='reports/eda/time_distribution.png'))
"""),
("md", "## Correlations and numeric feature distributions\nAssociations are not causal explanations. All displayed feature distributions come from the training partition."),
("code", """display(Image(filename='reports/eda/correlations.png'))
for path in sorted(Path('reports/eda').glob('feature_distributions_*.png')):
    display(Image(filename=str(path)))
"""),
("md", "## Outliers and limitations\nOutliers are retained because unusual transactions may carry fraud signal. Chronological evaluation measures later transactions, not unseen customers: customers can recur across windows. This benchmark does not establish real-world payment performance."),
("code", """eda = json.loads(Path('reports/eda/eda.json').read_text())
display(pd.Series(eda['iqr_outlier_counts'], name='IQR outliers').sort_values(ascending=False).to_frame())
display(Markdown(Path('reports/eda/findings.md').read_text()))
"""),
])

build("02_preprocessing.ipynb", [
("md", "# 02 · Source adaptation, splitting and training-only preprocessing\n\nThe adapter renames amt to Amount and converts transaction timestamps to elapsed seconds. It keeps category, state, population and coordinates. Raw names, card numbers, identifiers, dates of birth and unix_time are not model inputs. The same feature engineering is used at inference."),
("code", COMMON + """
schema = metadata['features']
assert set(train.index).isdisjoint(validation.index)
assert set(train.index).isdisjoint(test.index)
assert set(validation.index).isdisjoint(test.index)
if metadata['dataset']['split_strategy'] == 'chronological':
    assert train.Time.max() < validation.Time.min()
    assert validation.Time.max() < test.Time.min()
display(pd.DataFrame([{'partition': name, 'rows': len(part), 'fraud': int(part[target].sum())} for name, part in [('train', train), ('validation', validation), ('test', test)]]))
display(pd.DataFrame(schema))
"""),
("md", "## Shared transformations\nMedian imputation, robust scaling and float32 arrays are fitted on training rows. Categorical values use unknown-safe one-hot encoding. The geographic schema adds great-circle merchant distance; elapsed time adds daily and weekly cyclic features. Log amount is derived inside the pipeline."),
("code", """from src.feature_engineering import preprocessing
pipeline = preprocessing(schema).fit(validate_features(train[features], schema))
transformed = pipeline.transform(validation[features].head(5))
print('Input:', len(features), 'Transformed:', transformed.shape[1], 'dtype:', transformed.dtype)
print(pipeline.get_feature_names_out())
missing = validation[features].head(1).copy()
missing.loc[:, 'Amount'] = np.nan
assert np.isfinite(pipeline.transform(missing)).all()
"""),
("md", "## Oversampling\nRandom oversampling duplicates complete minority-class training rows and preserves categorical combinations. It runs inside each training CV fold, after training-fitted preprocessing. Validation/test rows retain their natural prevalence. Oversampled classifiers do not also apply class weights."),
("code", """display(pd.DataFrame([{'model': row['name'], 'strategy': row['imbalance_strategy'], 'parameters': row['parameters']} for row in metadata['comparison']]))
display(pd.DataFrame(metadata['training_class_counts']).rename(index={'0': 'Legitimate', '1': 'Fraud'}))
print(metadata['training']['cv_strategy'])
"""),
("md", "## Saved-model parity\nSampling is skipped at prediction time. The standalone preprocessor must produce the same probabilities as the complete saved pipeline."),
("code", """import joblib
from backend.services.model_service import ModelService
model = ModelService(Path('models'))
saved_preprocessor = joblib.load('models/preprocessor.joblib')
probe = test[features].head(20)
direct = model.pipeline.named_steps['classifier'].predict_proba(saved_preprocessor.transform(probe))[:, 1]
np.testing.assert_allclose(direct, model.predict_proba(probe), rtol=1e-7)
print('Saved preprocessor and inference pipeline agree on 20 later test examples')
"""),
])

build("03_model_training.ipynb", [
("md", "# 03 · Cross-validation, recall selection and evaluation\n\nResults below are read from measured artifacts. The Hugging Face workflow compares XGBoost with and without random oversampling, using expanding chronological CV windows on a 60,000-row training sample and full-partition refits. Model depth and sampling ratio are tuned by average precision. Validation F2 selects the model and threshold before test evaluation."),
("code", """from pathlib import Path
import json, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
metadata = json.loads(Path('models/model_metadata.json').read_text())
RUN_TRAINING = False
if RUN_TRAINING:
    from src.train import train
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        train(Path('data/raw') / metadata['dataset']['file'], Path('models'), target=metadata['dataset']['target'],
              seed=metadata['training']['seed'], profile_name=metadata['training']['profile'],
              minimum_recall=metadata.get('minimum_validation_recall'), candidate_names=metadata['training']['requested_candidates'],
              split_strategy=metadata['dataset']['split_strategy'])
    metadata = json.loads(Path('models/model_metadata.json').read_text())
print(metadata['selection_rule'])
print(metadata['training'])
"""),
("md", "## CV and validation\nTimestamps tied at boundaries never appear on both sides of a chronological training/validation boundary. Preprocessing and oversampling are fitted in training only. Few positive examples and recurring customers still limit generalization claims."),
("code", """display(pd.DataFrame([{'model': row['name'], 'CV AP': row['cv_pr_auc'], 'CV std': row['cv_std'],
                      'validation AP': row['validation']['pr_auc'], 'validation recall': row['validation']['recall'],
                      'validation precision': row['validation']['precision'], 'validation F2': row['validation']['f2'],
                      'threshold': row['threshold'], 'selected': row['selected']} for row in metadata['comparison']]))
"""),
("md", "## Frozen selection, then later test transactions\nThere is no refit on validation or test. Average precision is reported as PR-AUC; it is not trapezoidal integration. Results from the older ULB experiment belong to a different dataset and are not directly comparable."),
("code", """print('Selected:', metadata['model_name'], metadata['model_version'])
print('Threshold:', metadata['selected_threshold'])
display(pd.DataFrame([{'model': row['name'], **{key:value for key,value in row['test'].items() if key != 'confusion_matrix'}} for row in metadata['comparison']]))
display(pd.DataFrame(metadata['metrics']['confusion_matrix'], index=['Actual legitimate','Actual fraud'], columns=['Predicted legitimate','Predicted fraud']))
"""),
("md", "## Precision–recall, ROC and threshold trade-offs\nLower thresholds can catch more fraud and create more reviews. F2 weights recall more than precision. A configurable minimum validation recall is available, but does not guarantee future recall. Risk bands are separate presentation categories."),
("code", """fig, axes = plt.subplots(1, 3, figsize=(15, 4))
roc = pd.DataFrame(metadata['curves']['roc'])
pr = pd.DataFrame(metadata['curves']['precision_recall'])
thresholds = pd.DataFrame(metadata['threshold_curve'])
axes[0].plot(roc.fpr, roc.tpr)
axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Later test ROC')
axes[1].plot(pr.recall, pr.precision)
axes[1].set(xlabel='Recall', ylabel='Precision', title='Later test precision–recall')
axes[2].plot(thresholds.threshold, thresholds.precision, label='Precision')
axes[2].plot(thresholds.threshold, thresholds.recall, label='Recall')
axes[2].axvline(metadata['selected_threshold'], linestyle='--', color='grey')
axes[2].legend()
axes[2].set(xlabel='Threshold', title='Validation trade-off')
plt.tight_layout()
plt.show()
"""),
("md", "## Explainability and limits\nGlobal importance and local sensitivity describe associations, not reasons that prove fraud. Scores are uncalibrated. Validation and test customers may recur, and the dataset card provides limited collection provenance."),
("code", """print(metadata['importance_method'])
display(pd.DataFrame(metadata['feature_importance']).head(15))
print(metadata['probability_note'])
print(metadata['dataset'].get('provenance_note'))
"""),
])
