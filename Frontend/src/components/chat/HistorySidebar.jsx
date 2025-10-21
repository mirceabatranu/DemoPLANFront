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
    const [error, setError] = useState(null);
    const [displayLimit, setDisplayLimit] = useState(20); // Show 20 sessions initially

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const sessionHistory = await getSessionHistory();
                console.log("Raw session history from API:", sessionHistory);
                
                // Process and format the session history
                const formattedHistory = sessionHistory.map(session => {
                    // Generate a display title with shorter format
                    let displayTitle = session.title;
                    
                    if (!displayTitle) {
                        // If no title, create one from created_at date
                        try {
                            const createdDate = new Date(session.created_at);
                            if (!isNaN(createdDate.getTime())) {
                                // Shorter format: "Oct 13, 2025" or "Oct 13" (current year)
                                const currentYear = new Date().getFullYear();
                                const sessionYear = createdDate.getFullYear();
                                
                                if (sessionYear === currentYear) {
                                    // Same year: show "Chat - Oct 13"
                                    displayTitle = `Chat - ${createdDate.toLocaleDateString('en-US', {
                                        month: 'short',
                                        day: 'numeric'
                                    })}`;
                                } else {
                                    // Different year: show "Chat - Oct 13, 2025"
                                    displayTitle = `Chat - ${createdDate.toLocaleDateString('en-US', {
                                        month: 'short',
                                        day: 'numeric',
                                        year: 'numeric'
                                    })}`;
                                }
                            } else {
                                displayTitle = `Session ${session.session_id.substring(0, 8)}`;
                            }
                        } catch (e) {
                            console.error("Error formatting date for session:", session.session_id, e);
                            displayTitle = `Session ${session.session_id.substring(0, 8)}`;
                        }
                    }
                    
                    return {
                        ...session,
                        displayTitle
                    };
                });
                
                // Sort by created_at descending (newest first)
                formattedHistory.sort((a, b) => {
                    try {
                        const dateA = new Date(a.created_at);
                        const dateB = new Date(b.created_at);
                        return dateB - dateA;
                    } catch (e) {
                        return 0;
                    }
                });
                
                console.log("Formatted session history:", formattedHistory);
                setHistory(formattedHistory);
                setError(null);
            } catch (error) {
                console.error("Could not fetch session history:", error);
                setError("Nu s-a putut încărca istoricul");
            }
        };
        
        if (!isCollapsed) {
            fetchHistory();
        }
    }, [isCollapsed]);

    const handleLoadMore = () => {
        setDisplayLimit(prev => prev + 20);
    };

    const displayedHistory = history.slice(0, displayLimit);
    const hasMore = history.length > displayLimit;

    return (
        <aside className={`history-sidebar ${isCollapsed ? 'collapsed' : ''}`}>
            <div className="sidebar-header">
                {!isCollapsed && <span>Istoric</span>}
                {/* Desktop-only collapse button */}
                <button onClick={onToggle} className="sidebar-toggle-button" aria-label="Toggle sidebar">
                    <CollapseIcon />
                </button>
                {/* Mobile-only close button */}
                <button onClick={onToggle} className="mobile-sidebar-close-button" aria-label="Close sidebar">
                    <CollapseIcon />
                </button>
            </div>
            {!isCollapsed && (
                <>
                    <button className="new-chat-button" onClick={onNewChat}>
                        + Chat Nou
                    </button>
                    <div className="history-list">
                        {error ? (
                            <p className="error-text">{error}</p>
                        ) : displayedHistory.length > 0 ? (
                            <>
                                {displayedHistory.map(session => (
                                    <button
                                        key={session.session_id}
                                        className="history-item-button"
                                        onClick={() => onLoadSession(session.session_id)}
                                        title={session.displayTitle}
                                    >
                                        {session.displayTitle}
                                    </button>
                                ))}
                                {hasMore && (
                                    <button 
                                        className="load-more-button" 
                                        onClick={handleLoadMore}
                                    >
                                        Încarcă mai multe ({history.length - displayLimit} rămase)
                                    </button>
                                )}
                            </>
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