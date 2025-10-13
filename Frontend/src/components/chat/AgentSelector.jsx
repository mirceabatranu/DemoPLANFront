import React, { useState } from 'react';

const AgentSelector = () => {
    const [selectedAgent, setSelectedAgent] = useState('evaluator');

    const handleAgentChange = (event) => {
        setSelectedAgent(event.target.value);
        // TODO: Add logic here to handle agent switching,
        // like starting a new session or informing the parent component.
    };

    return (
        <div className="agent-selector-wrapper">
            <select
                id="agent-selector"
                value={selectedAgent}
                onChange={handleAgentChange}
                className="agent-selector-dropdown"
            >
                <option value="evaluator">Agent estimări si evaluări planuri</option>
                <option value="generator">Agent generare planuri</option>
            </select>
        </div>
    );
};

export default AgentSelector;