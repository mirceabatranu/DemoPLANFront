import React from 'react';

const FileIcon = () => <svg stroke="currentColor" fill="none" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>;
const CollapseIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M13 17L18 12L13 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M6 17L11 12L6 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);

const FilesSidebar = ({ files, isCollapsed, onToggle }) => {
    return (
        <aside className={`files-sidebar ${isCollapsed ? 'collapsed' : ''}`}>
            <div className="sidebar-header">
                {/* CHANGE: Add a mobile-only close button (on the left) */}
                <button onClick={onToggle} className="mobile-sidebar-close-button">
                    <CollapseIcon />
                </button>
                
                {/* Desktop-only collapse button */}
                <button onClick={onToggle} className="sidebar-toggle-button">
                    <CollapseIcon />
                </button>

                {!isCollapsed && <span>Fișiere Atașate</span>}
            </div>
            {!isCollapsed && (
                <>
                    {files.length === 0 ? (
                        <p className="no-files-text">
                            Fișierele încărcate vor apărea aici.
                        </p>
                    ) : (
                        <ul className="files-list">
                            {files.map((file, index) => (
                                <li key={index} className="file-item">
                                    <FileIcon />
                                    <span>{file.name}</span>
                                </li>
                            ))}
                        </ul>
                    )}
                </>
            )}
        </aside>
    );
};

export default FilesSidebar;