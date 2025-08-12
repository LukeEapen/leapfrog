import React from 'react';
import Layout from '../components/Layout';
import theme from '../components/Theme';

const cardStyle = {
  background: theme.colors.agent,
  color: theme.colors.text,
  borderRadius: theme.borderRadius,
  padding: theme.spacing,
  margin: theme.spacing,
  boxShadow: '0 2px 8px rgba(246,194,62,0.08)'
};

export default function ServiceBuilderAgent({ onNext }) {
  return (
    <Layout>
      <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
        <div style={cardStyle}>
          <h2>Service Builder Agent</h2>
          <p>Build the modernized service based on the finalized scope and design.</p>
          <button style={{background: theme.colors.agentCore, color: '#fff', border: 'none', borderRadius: 8, padding: '10px 28px', fontSize: 16, cursor: 'pointer'}} onClick={onNext}>
            Next
          </button>
        </div>
      </div>
    </Layout>
  );
}
