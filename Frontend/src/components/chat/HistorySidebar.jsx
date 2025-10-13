import React, { useState, useEffect } from 'react';
import { getSessionHistory } from '../../services/apiClient';

const CollapseIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M11 17L6 12L11 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M18 17L13 12L18 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);

const HistorySidebar = ({ onNewChat, onLoadSession, isCollapsed, onToggle }) => {
    const [history, setHistory] = useState([]);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const sessionHistory = await getSessionHistory();
                setHistory(sessionHistory);
            } catch (error) {
                console.error("Could not fetch session history:", error);
            }
        };
        fetchHistory();
    }, []);

    return (
        <aside className={`history-sidebar ${isCollapsed ? 'collapsed' : ''}`}>
            <div className="sidebar-header">
                {!isCollapsed && <span>Istoric</span>}
                {/* Desktop-only collapse button */}
                <button onClick={onToggle} className="sidebar-toggle-button">
                    <CollapseIcon />
                </button>
                {/* CHANGE: Add a mobile-only close button */}
                <button onClick={onToggle} className="mobile-sidebar-close-button">
                    <CollapseIcon />
                </button>
            </div>
            {!isCollapsed && (
                <>
                    <button className="new-chat-button" onClick={onNewChat}>
                        + Chat Nou
                    </button>
                    <div className="history-list">
                        {history.length > 0 ? (
                            history.map(session => (
                                <button
                                    key={session.session_id}
                                    className="history-item-button"
                                    onClick={() => onLoadSession(session.session_id)}
                                >
                                    {session.title || session.session_id}
                                </button>
                            ))
                        ) : (
                            <p className="no-sessions-text">Nicio sesiune anterioară găsită.</p>
                        )}
                    </div>
                </>
            )}
        </aside>
    );
};

export default HistorySidebar;