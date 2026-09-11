import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Threshold, PredictionTable, ErrorState } from './ui';
import { errorMessage, money } from '../services/api';

describe('risk and result interactions', () => {
  it('resets the threshold to the exact active default', () => {
    const change = vi.fn();
    render(<Threshold value={0.7} onChange={change} defaultValue={0.183746}/>);
    fireEvent.click(screen.getByText(/Reset to active default/));
    expect(change).toHaveBeenCalledWith(0.183746);
  });
  it('shows decisions and risk with text, not only color', () => {
    render(<PredictionTable rows={[{ id: 'abc', amount: 19.99, prediction: 'Fraud', risk_level: 'Low', fraud_probability: 0.1 }]}/>);
    expect(screen.getByText('Low')).toBeVisible();
    expect(screen.getByText('⚑ Fraud flag')).toBeVisible();
    expect(screen.getByText('10.00%')).toBeVisible();
  });
  it('can retry an API failure', () => {
    const retry = vi.fn(); render(<ErrorState error="Backend unavailable" retry={retry}/>);
    fireEvent.click(screen.getByRole('button', {name: 'Try again'}));
    expect(retry).toHaveBeenCalledOnce();
  });
  it('renders backend validation details without hiding the field', () => {
    expect(errorMessage({detail: [{location: ['body','features','Amount'], message: 'Invalid number'}]})).toContain('Amount: Invalid number');
  });
  it('does not assign a currency when the dataset does not specify one', () => {
    expect(money(1234.5, null)).toBe('1,234.50');
    expect(money(1234.5, 'EUR')).toContain('€');
  });
});
