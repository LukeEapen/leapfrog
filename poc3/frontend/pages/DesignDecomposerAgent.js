import React from 'react';
import Layout from '../components/Layout';
import theme from '../components/Theme';

const cardStyle = {
  background: theme.colors.agentCore,
  color: '#fff',
  borderRadius: theme.borderRadius,
  padding: theme.spacing,
  margin: theme.spacing,
  boxShadow: '0 2px 8px rgba(231,76,60,0.08)'
};

export default function DesignDecomposerAgent({ onNext }) {
  return (
    <Layout>
      <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
        <div style={cardStyle}>
          <h2>Design Decomposer Agent</h2>
          <p>Decompose functions into design elements, referencing standards.</p>
          <div style={{margin: '16px 0'}}>
            <span style={{background: theme.colors.shared, color: theme.colors.text, borderRadius: 6, padding: '4px 12px', marginRight: 8}}>Open API</span>
            <span style={{background: theme.colors.shared, color: theme.colors.text, borderRadius: 6, padding: '4px 12px', marginRight: 8}}>Bian</span>
            <span style={{background: theme.colors.shared, color: theme.colors.text, borderRadius: 6, padding: '4px 12px'}}>ISO</span>
          </div>
          <button style={{background: theme.colors.agent, color: theme.colors.text, border: 'none', borderRadius: 8, padding: '10px 28px', fontSize: 16, cursor: 'pointer'}} onClick={onNext}>
            Next
          </button>
        </div>
      </div>
    </Layout>
  );
}
