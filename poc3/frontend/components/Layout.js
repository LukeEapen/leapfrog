import React from 'react';
import theme from './Theme';

const Layout = ({ children }) => (
  <div style={{
    minHeight: '100vh',
    background: theme.colors.background,
    fontFamily: theme.fontFamily,
    color: theme.colors.text,
  }}>
    <header style={{padding: theme.spacing, background: theme.colors.agentCore, color: '#fff'}}>
      <h1 style={{margin: 0}}>Service Builder</h1>
    </header>
    <main style={{padding: theme.spacing}}>
      {children}
    </main>
  </div>
);

export default Layout;
