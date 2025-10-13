import React from 'react';
import { Link } from 'react-router-dom';

const LoginPage = () => {
  return (
    <Link to="/chat" className="login-link">
      <main className="login-card">
        <div className="logo-container">
          <div className="logo-inner">
            <span className="logo-text">D</span>
          </div>
        </div>

        <h1>DemoPlan</h1>
        <p className="tagline">Generate your offer with AI</p>
        <p className="prompt">Click to enter</p>

        <div className="footer">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
          <span>Powered by Gemini</span>
        </div>
      </main>
    </Link>
  );
};

export default LoginPage;