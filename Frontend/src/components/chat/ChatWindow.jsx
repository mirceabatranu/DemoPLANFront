import React, { useEffect, useRef } from 'react';
import Message from './Message';
import TypingIndicator from './TypingIndicator'; // Import the new component

// The component now accepts 'messages' and 'isLoading'
const ChatWindow = ({ messages, isLoading }) => {
  const chatContainerRef = useRef(null); // Create a ref for the container div

  // useEffect will run every time the 'messages' or 'isLoading' state changes
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, isLoading]); // Dependency array ensures this runs on updates

  return (
    // Attach the ref to the main div
    <div className="chat-window" ref={chatContainerRef}>
      {messages.map((msg) => (
        <Message key={msg.id} text={msg.text} sender={msg.sender} />
      ))}
      
      {/* Conditionally render the typing indicator */}
      {isLoading && <TypingIndicator />}
    </div>
  );
};

export default ChatWindow;