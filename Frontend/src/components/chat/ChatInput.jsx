import React, { useState, useRef, useEffect } from 'react';

const PaperclipIcon = () => <svg stroke="currentColor" fill="none" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>;
const SendIcon = () => <svg stroke="currentColor" fill="none" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>;

const ChatInput = ({ onSendMessage, onFileUpload, isLoading }) => {
    const [message, setMessage] = useState('');
    const fileInputRef = useRef(null);
    const textareaRef = useRef(null);

    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            const maxHeight = 200; 

            textarea.style.height = 'auto';
            textarea.style.overflowY = 'hidden';

            const scrollHeight = textarea.scrollHeight;

            if (scrollHeight > maxHeight) {
                textarea.style.height = `${maxHeight}px`;
                textarea.style.overflowY = 'auto';
            } else {
                textarea.style.height = `${scrollHeight}px`;
            }
        }
    }, [message]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (message.trim() && !isLoading) {
            onSendMessage(message);
            setMessage('');
        }
    };

    const handleAttachClick = () => {
        fileInputRef.current.click();
    };

    const handleFileChange = (e) => {
        if (e.target.files.length > 0) {
            onFileUpload(e.target.files);
        }
    };

    return (
        <div className="chat-input-container">
            <input
                type="file"
                multiple
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }}
            />

            {/* This is the corrected line */}
            <form className="chat-input-form" onSubmit={handleSubmit}>
                <textarea
                    ref={textareaRef}
                    className="chat-textarea"
                    placeholder={isLoading ? "Se procesează..." : "Trimite un mesaj..."}
                    rows="1"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    disabled={isLoading}
                />
                <div className="input-buttons">
                    <button
                        type="button"
                        className="input-icon-button"
                        title="Atașează fișier"
                        disabled={isLoading}
                        onClick={handleAttachClick}
                    >
                        <PaperclipIcon />
                    </button>
                    <button type="submit" className="send-button" title="Trimite mesaj" disabled={isLoading}>
                        <SendIcon />
                    </button>
                </div>
            </form>
        </div>
    );
};

export default ChatInput;