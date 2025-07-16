import React from 'react';

function AgentCard({ name, description }) {
  return (
    <div style={{border: '1px solid #ccc', borderRadius: 8, padding: 16, margin: 8, background: '#f9f9f9'}}>
      <h3>{name}</h3>
      <p>{description}</p>
    </div>
  );
}

export default AgentCard;
