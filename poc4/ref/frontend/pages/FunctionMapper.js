import React from 'react';
import Layout from '../components/Layout';
import theme from '../components/Theme';

const cardStyle = {
  background: theme.colors.shared,
  color: theme.colors.text,
  borderRadius: theme.borderRadius,
  padding: theme.spacing,
  margin: theme.spacing,
  boxShadow: '0 2px 8px rgba(178,223,219,0.08)'
};

export default function FunctionMapper({ onNext }) {
  return (
    <Layout>
      <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
        <div style={cardStyle}>
          <h2>Function Mapper</h2>
          <p>Map synthesized functions to standards and prepare for design decomposition.</p>
          <div style={{margin: '16px 0'}}>
            <span style={{background: theme.colors.approval, color: '#fff', borderRadius: 6, padding: '4px 12px', marginRight: 8}}>Approval</span>
            <span style={{background: theme.colors.chat, color: '#fff', borderRadius: 6, padding: '4px 12px'}}>Chat</span>
          </div>
          <button style={{background: theme.colors.agentCore, color: '#fff', border: 'none', borderRadius: 8, padding: '10px 28px', fontSize: 16, cursor: 'pointer'}} onClick={onNext}>
            Next
          </button>
        </div>
      </div>
    </Layout>
  );
}
