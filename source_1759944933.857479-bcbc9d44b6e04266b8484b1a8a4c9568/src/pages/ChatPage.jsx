import React, { useState } from 'react';
import {
    startNewSession, sendChatMessage, uploadFiles,
    getSessionMessages, getSessionFiles
} from '../services/apiClient';
import HistorySidebar from '../components/chat/HistorySidebar';
import ChatWindow from '../components/chat/ChatWindow';
import FilesSidebar from '../components/chat/FilesSidebar';
import ChatInput from '../components/chat/ChatInput';
import AgentSelector from '../components/chat/AgentSelector'; // <-- Import new component
import '../styles/ChatPage.css';

const ChatPage = () => {
    const [sessionId, setSessionId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [uploadedFiles, setUploadedFiles] = useState([]);
    // New state for sidebar visibility
    const [isHistoryCollapsed, setIsHistoryCollapsed] = useState(false);
    const [isFilesCollapsed, setIsFilesCollapsed] = useState(false);

    const handleNewChat = async () => { /* ... existing code ... */ };
    const handleSendMessage = async (newMessageText) => { /* ... existing code ... */ };
    const handleFileUpload = async (files) => { /* ... existing code ... */ };
    const handleLoadSession = async (sid) => { /* ... existing code ... */ };

    return (
        <div className="chat-page">
            <HistorySidebar
                onNewChat={handleNewChat}
                onLoadSession={handleLoadSession}
                isCollapsed={isHistoryCollapsed}
                onToggle={() => setIsHistoryCollapsed(!isHistoryCollapsed)}
            />
            <main className="main-content">
                <AgentSelector /> {/* <-- Add new component here */}
                <ChatWindow messages={messages} isLoading={isLoading} />
                <ChatInput
                    onSendMessage={handleSendMessage}
                    onFileUpload={handleFileUpload}
                    isLoading={isLoading}
                />
            </main>
            <FilesSidebar
                files={uploadedFiles}
                isCollapsed={isFilesCollapsed}
                onToggle={() => setIsFilesCollapsed(!isFilesCollapsed)}
            />
        </div>
    );
};

// Paste your full handler functions back here
// For brevity, I've omitted them, but you must include them for the page to work.
const FullChatPage = () => {
    const [sessionId, setSessionId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [isHistoryCollapsed, setIsHistoryCollapsed] = useState(false);
    const [isFilesCollapsed, setIsFilesCollapsed] = useState(false);

    const handleNewChat = async () => {
        setIsLoading(true);
        setMessages([]);
        setUploadedFiles([]);
        try {
            const newSessionId = await startNewSession();
            setSessionId(newSessionId);
            setMessages([{ id: 1, text: "New session started. Please upload your files or ask a question.", sender: 'ai' }]);
        } catch (error) {
            console.error("Failed to start new session:", error);
            setMessages([{ id: 1, text: "Error: Could not start a new session. Please try again.", sender: 'ai' }]);
        }
        setIsLoading(false);
    };

    const handleSendMessage = async (newMessageText) => {
        if (!sessionId) {
            alert("Please start a new session first by clicking '+ New Chat'.");
            return;
        }
        const userMessage = { id: Date.now(), text: newMessageText, sender: 'user' };
        setMessages(prevMessages => [...prevMessages, userMessage]);
        setIsLoading(true);
        try {
            const aiResponse = await sendChatMessage(sessionId, newMessageText);
            const aiMessage = { id: Date.now() + 1, text: aiResponse.response, sender: 'ai' };
            setMessages(prevMessages => [...prevMessages, aiMessage]);
        } catch (error) {
            console.error("Failed to send message:", error);
            const errorMessage = { id: Date.now() + 1, text: "Sorry, I couldn't get a response.", sender: 'ai' };
            setMessages(prevMessages => [...prevMessages, errorMessage]);
        }
        setIsLoading(false);
    };

    const handleFileUpload = async (files) => {
        if (!sessionId) {
            alert("Please start a new session first to upload files.");
            return;
        }
        setIsLoading(true);
        try {
            const response = await uploadFiles(sessionId, files);
            const analysisMessage = {
                id: Date.now(),
                text: response.ai_response || "Files uploaded and analyzed successfully.",
                sender: 'ai',
            };
            setMessages(prev => [...prev, analysisMessage]);
            const newFiles = Array.from(files);
            setUploadedFiles(prev => [...prev, ...newFiles]);
        } catch (error) {
            console.error("File upload failed:", error);
            const errorMessage = {
                id: Date.now(),
                text: "There was an error uploading your files. Please try again.",
                sender: 'ai',
            };
            setMessages(prev => [...prev, errorMessage]);
        }
        setIsLoading(false);
    };

    const handleLoadSession = async (sid) => {
        setIsLoading(true);
        try {
            const [loadedMessages, loadedFiles] = await Promise.all([
                getSessionMessages(sid),
                getSessionFiles(sid)
            ]);
            setSessionId(sid);
            const formattedMessages = loadedMessages.map((msg, index) => ({
                id: index,
                sender: msg.type === 'assistant' ? 'ai' : 'user',
                text: msg.content
            }));
            setMessages(formattedMessages);
            const formattedFiles = loadedFiles.map(file => ({ name: file.filename }));
            setUploadedFiles(formattedFiles);
        } catch (error) {
            console.error("Failed to load session:", error);
            setMessages([{ id: 1, text: "Error: Could not load the selected session.", sender: 'ai' }]);
        }
        setIsLoading(false);
    };

    return (
        <div className="chat-page">
            <HistorySidebar
                onNewChat={handleNewChat}
                onLoadSession={handleLoadSession}
                isCollapsed={isHistoryCollapsed}
                onToggle={() => setIsHistoryCollapsed(!isHistoryCollapsed)}
            />
            <main className="main-content">
                <AgentSelector />
                <ChatWindow messages={messages} isLoading={isLoading} />
                <ChatInput
                    onSendMessage={handleSendMessage}
                    onFileUpload={handleFileUpload}
                    isLoading={isLoading}
                />
            </main>
            <FilesSidebar
                files={uploadedFiles}
                isCollapsed={isFilesCollapsed}
                onToggle={() => setIsFilesCollapsed(!isFilesCollapsed)}
            />
        </div>
    );
};

export default FullChatPage;