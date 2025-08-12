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

export default function UserStoryDecomposer({ onNext }) {
  return (
    <Layout>
      <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
        <div style={cardStyle}>
          <h2>User Story Decomposer</h2>
          <p>Decompose user stories into actionable requirements and functions.</p>
          <button style={{background: theme.colors.agentCore, color: '#fff', border: 'none', borderRadius: 8, padding: '10px 28px', fontSize: 16, cursor: 'pointer'}} onClick={onNext}>
            Next
          </button>
        </div>
      </div>
    </Layout>
  );
}
