import { useCallback, useEffect, useState } from 'react';
import { api } from '../services/api';

export function useResource(path) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision(n => n + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    setState(previous => ({ ...previous, error: null, loading: true }));
    api(path, { signal: controller.signal }).then(data => setState({ data, loading: false, error: null }))
      .catch(error => { if (error.name !== 'AbortError') setState({ data: null, loading: false, error: error.message }); });
    return () => controller.abort();
  }, [path, revision]);
  return { ...state, refresh };
}
