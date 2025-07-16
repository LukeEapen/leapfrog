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

export default function ModernizationValidatorAgent() {
  return (
    <Layout>
      <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
        <div style={cardStyle}>
          <h2>Modernization Validator Agent</h2>
          <p>Validate the modernized service for compliance and readiness.</p>
        </div>
      </div>
    </Layout>
  );
}
