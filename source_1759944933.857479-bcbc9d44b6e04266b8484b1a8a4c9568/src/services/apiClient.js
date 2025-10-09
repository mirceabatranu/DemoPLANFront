// A simple API client to communicate with your backend
const API_BASE_URL = "https://demoplan-unified-1041867695241.europe-west1.run.app"; // Replace with your actual backend URL

/**
 * Starts a new chat session.
 * @returns {Promise<string>} The new session ID.
 */
export const startNewSession = async () => {
    const response = await fetch(`${API_BASE_URL}/start-session`, {
        method: 'POST',
    });
    if (!response.ok) {
        throw new Error('Failed to start a new session');
    }
    const data = await response.json();
    return data.session_id;
};

/**
 * Uploads files to a specific session.
 * @param {string} sessionId The ID of the session.
 * @param {FileList} files The files to upload from an input element.
 * @returns {Promise<any>} The analysis result from the backend.
 */
export const uploadFiles = async (sessionId, files) => {
    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }

    const response = await fetch(`${API_BASE_URL}/session/${sessionId}/upload`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error('File upload failed');
    }
    return response.json();
};

/**
 * Sends a chat message to a specific session.
 * @param {string} sessionId The ID of the session.
 * @param {string} message The message text.
 * @returns {Promise<any>} The AI's response.
 */
export const sendChatMessage = async (sessionId, message) => {
    const formData = new FormData();
    formData.append('message', message);

    const response = await fetch(`${API_BASE_URL}/session/${sessionId}/chat`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error('Failed to send message');
    }
    return response.json();
};

/**
 * Fetches the complete status and data for a session, including messages and files.
 * This function will make multiple calls to get all session data.
 * @param {string} sessionId The ID of the session.
 * @returns {Promise<object>} An object containing session details, messages, and files.
 */
export const getSessionData = async (sessionId) => {
    // This part of the function is not implemented as the backend does not
    // have an endpoint to get all messages for a session.
};

/**
 * Fetches the list of all past chat sessions.
 * @returns {Promise<Array>} A list of session objects.
 */
export const getSessionHistory = async () => {
    const response = await fetch(`${API_BASE_URL}/sessions`);
    if (!response.ok) {
        throw new Error('Failed to fetch session history');
    }
    return response.json();
};

/**
 * Fetches all messages for a specific session.
 * @param {string} sessionId The ID of the session.
 * @returns {Promise<Array>} A list of message objects.
 */
export const getSessionMessages = async (sessionId) => {
    const response = await fetch(`${API_BASE_URL}/session/${sessionId}/messages`);
    if (!response.ok) {
        throw new Error('Failed to fetch session messages');
    }
    return response.json();
};

/**
 * Fetches all file analyses for a specific session.
 * NOTE: Your backend already has this endpoint, so we'll just create the client function for it.
 * @param {string} sessionId The ID of the session.
 * @returns {Promise<Array>} A list of file analysis objects.
 */
export const getSessionFiles = async (sessionId) => {
    const response = await fetch(`${API_BASE_URL}/session/${sessionId}/files`);
    if (!response.ok) {
        throw new Error('Failed to fetch session files');
    }
    const data = await response.json();
    return data.files; // The endpoint wraps files in a "files" key
};