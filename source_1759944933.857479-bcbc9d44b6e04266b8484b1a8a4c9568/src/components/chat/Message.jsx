import React from 'react';

// This is a simple user icon.
const UserIcon = () => <div className="message-icon user-icon">U</div>;
// This is a simple AI icon.
const AiIcon = () => <div className="message-icon ai-icon">A</div>;

const Message = ({ text, sender }) => {
  const isUser = sender === 'user';

  return (
    <div className={`message-wrapper ${isUser ? 'user' : 'ai'}`}>
      {isUser ? <UserIcon /> : <AiIcon />}
      <div className="message-text">
        {text}
      </div>
    </div>
  );
};

export default Message;