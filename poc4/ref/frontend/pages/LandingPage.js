import React from 'react';
import Layout from '../components/Layout';
import theme from '../components/Theme';

const cardStyle = {
  background: theme.colors.artifact,
  color: '#fff',
  borderRadius: theme.borderRadius,
  padding: theme.spacing,
  margin: theme.spacing,
  boxShadow: '0 2px 8px rgba(44,166,164,0.08)'
};

export default function LandingPage({ onStart }) {
  return (
    <Layout>
      <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
        <div style={cardStyle}>
          <h2>Welcome to Service Builder</h2>
          <p>Start a new workflow to modernize your services using legacy code and user stories.</p>
          <button style={{background: theme.colors.agentCore, color: '#fff', border: 'none', borderRadius: 8, padding: '12px 32px', fontSize: 18, cursor: 'pointer'}} onClick={onStart}>
            Start Workflow
          </button>
        </div>
      </div>
    </Layout>
  );
}
