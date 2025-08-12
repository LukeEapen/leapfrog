import React from 'react';
import Layout from '../components/Layout';
import theme from '../components/Theme';

const horizontalStyle = {
  background: theme.colors.shared,
  color: theme.colors.text,
  borderRadius: theme.borderRadius,
  padding: theme.spacing,
  margin: theme.spacing,
  boxShadow: '0 2px 8px rgba(178,223,219,0.08)'
};

export default function Horizontals() {
  return (
    <Layout>
      <div style={{display: 'flex', flexWrap: 'wrap', justifyContent: 'center'}}>
        <div style={horizontalStyle}>
          <h3>Traceability</h3>
        </div>
        <div style={horizontalStyle}>
          <h3>Shared Memory Store</h3>
        </div>
        <div style={horizontalStyle}>
          <h3>Domain Ontology Agent</h3>
        </div>
        <div style={horizontalStyle}>
          <h3>Observability Agent</h3>
        </div>
        <div style={horizontalStyle}>
          <h3>Compliance Evaluator Agent</h3>
        </div>
      </div>
    </Layout>
  );
}
