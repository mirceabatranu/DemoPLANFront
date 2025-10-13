// Keep all your existing imports at the top of the file
import React, { useState } from 'react';
import {
    startNewSession, sendChatMessage, uploadFiles,
    getSessionMessages, getSessionFiles
} from '../services/apiClient';
import HistorySidebar from '../components/chat/HistorySidebar';
import ChatWindow from '../components/chat/ChatWindow';
import FilesSidebar from '../components/chat/FilesSidebar';
import ChatInput from '../components/chat/ChatInput';
import AgentSelector from '../components/chat/AgentSelector';
import '../styles/ChatPage.css';

// You can define these icons here or import them
const HistoryIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M11 17L6 12L11 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M18 17L13 12L18 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);
const FilesIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M13 17L18 12L13 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M6 17L11 12L6 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);


const FullChatPage = () => {
    const [sessionId, setSessionId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [uploadedFiles, setUploadedFiles] = useState([]);
    // FIX 3: Set initial state to true to be collapsed by default
    const [isHistoryCollapsed, setIsHistoryCollapsed] = useState(true);
    const [isFilesCollapsed, setIsFilesCollapsed] = useState(true);

    // ... (keep all your handler functions: handleNewChat, handleSendMessage, etc.)
    const handleNewChat = async () => {
        setIsLoading(true);
        setMessages([]);
        setUploadedFiles([]);
        try {
            const newSessionId = await startNewSession();
            setSessionId(newSessionId);
            setMessages([{ id: 1, text: "Sesiune nouă a început. Vă rugăm să încărcați fișierele sau să adresați o întrebare.", sender: 'ai' }]);
        } catch (error) {
            console.error("Failed to start new session:", error);
            setMessages([{ id: 1, text: "Eroare: Nu s-a putut porni o sesiune nouă. Vă rugăm să încercați din nou.", sender: 'ai' }]);
        }
        setIsLoading(false);
    };

    const handleSendMessage = async (newMessageText) => {
        if (!sessionId) {
            alert("Vă rugăm să începeți o nouă sesiune dând clic pe '+ Chat Nou'.");
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
            const errorMessage = { id: Date.now() + 1, text: "Ne pare rău, nu am putut obține un răspuns.", sender: 'ai' };
            setMessages(prevMessages => [...prevMessages, errorMessage]);
        }
        setIsLoading(false);
    };

    const handleFileUpload = async (files) => {
        if (!sessionId) {
            alert("Vă rugăm să începeți o nouă sesiune pentru a încărca fișiere.");
            return;
        }
        setIsLoading(true);
        try {
            const response = await uploadFiles(sessionId, files);
            const analysisMessage = {
                id: Date.now(),
                text: response.ai_response || "Fișierele au fost încărcate și analizate cu succes.",
                sender: 'ai',
            };
            setMessages(prev => [...prev, analysisMessage]);
            const newFiles = Array.from(files);
            setUploadedFiles(prev => [...prev, ...newFiles]);
        } catch (error) {
            console.error("File upload failed:", error);
            const errorMessage = {
                id: Date.now(),
                text: "A apărut o eroare la încărcarea fișierelor. Vă rugăm să încercați din nou.",
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
            setMessages([{ id: 1, text: "Eroare: Nu s-a putut încărca sesiunea selectată.", sender: 'ai' }]);
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
                {/* FIX 3: New header for agent selector and mobile buttons */}
                <div className="main-header">
                    <button
                        className="mobile-toggle-button history-toggle-mobile"
                        onClick={() => setIsHistoryCollapsed(!isHistoryCollapsed)}
                    >
                        <HistoryIcon />
                    </button>
                    <AgentSelector />
                    <button
                        className="mobile-toggle-button files-toggle-mobile"
                        onClick={() => setIsFilesCollapsed(!isFilesCollapsed)}
                    >
                        <FilesIcon />
                    </button>
                </div>

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