export const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');

export function errorMessage(data, fallback = 'The request could not be completed.') {
  if (typeof data?.detail === 'string') return data.detail;
  if (Array.isArray(data?.detail)) return data.detail.map(e => `${(e.location || e.loc || []).join('.')}: ${e.message || e.msg}`).join('; ');
  return fallback;
}

export async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(data, `API returned ${response.status}. Check that the backend is ready.`));
  return data;
}

export function upload(path, file, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', `${API_BASE}${path}`);
    request.timeout = 120000;
    request.upload.onprogress = event => { if (event.lengthComputable) onProgress?.(Math.round(event.loaded / event.total * 100)); };
    request.onload = () => {
      let data;
      try { data = JSON.parse(request.responseText); } catch { reject(new Error('The server returned an invalid response.')); return; }
      if (request.status >= 200 && request.status < 300) resolve(data);
      else reject(new Error(errorMessage(data)));
    };
    request.onerror = () => reject(new Error('Cannot reach the API. Check your connection and backend.'));
    request.ontimeout = () => reject(new Error('The upload timed out. Try a smaller batch.'));
    const body = new FormData(); body.append('file', file); request.send(body);
  });
}

export function downloadText(text, filename, type = 'text/csv;charset=utf-8') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename;
  document.body.appendChild(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const number = value => new Intl.NumberFormat('en-US').format(value ?? 0);
export const percent = (value, digits = 2) => `${((value ?? 0) * 100).toFixed(digits)}%`;
export const money = (value, currency = null) => value == null ? '—' : new Intl.NumberFormat('en-US', currency ? { style: 'currency', currency } : { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);
