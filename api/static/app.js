/**
 * app.js — Frontend logic for FCA Compliance RAG UI.
 * Handles: RBAC simulation, Chat interactions, Evidence visualization.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let currentRole = 'admin';
    let chatHistory = [];
    let currentEvidence = [];

    // --- DOM Elements ---
    const chatMessages = document.getElementById('chat-messages');
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const roleBtns = document.querySelectorAll('.role-btn');
    const evidenceSidebar = document.getElementById('evidence-sidebar');
    const evidenceList = document.getElementById('evidence-list');
    const closeEvidence = document.getElementById('close-evidence');
    const clearChatBtn = document.getElementById('clear-chat');
    const sendBtn = document.getElementById('send-btn');
    const messageTemplate = document.getElementById('message-template');
    const sourceCardTemplate = document.getElementById('source-card-template');

    // --- Initialization ---
    autoResizeTextarea();

    // --- Event Listeners ---

    // Role Selection
    roleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            roleBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentRole = btn.dataset.role;
            addSystemMessage(`User role switched to: <strong>${currentRole.toUpperCase()}</strong>`);
        });
    });

    // Form Submission
    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        await handleQuery(query);
    });

    // Enter to submit (Shift+Enter for newline)
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            queryForm.dispatchEvent(new Event('submit'));
        }
    });

    // Sidebar Close
    closeEvidence.addEventListener('click', () => {
        evidenceSidebar.classList.add('collapsed');
    });

    // Clear Chat
    clearChatBtn.addEventListener('click', () => {
        chatMessages.innerHTML = '';
        addSystemMessage('Chat session cleared.');
    });

    // --- Functions ---

    async function handleQuery(query) {
        // 1. Add User Message
        addMessage(query, 'user');
        queryInput.value = '';
        queryInput.style.height = 'auto';

        // 2. Add Loading Indicator
        const loadingMsg = addMessage('Processing query...', 'assistant', true);
        sendBtn.disabled = true;

        try {
            // 3. API Call
            // NOTE: In Phase 4, /query requires a JWT. 
            // We simulate the 'auth' layer by passing the role in the header 
            // or as a mock token that the backend would decode.
            // For this UI demo, we'll use a standard Fetch call.

            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // Simulate a JWT by passing a mock header that the backend wrapper can intercept
                    'Authorization': `Bearer MOCK_TOKEN_ROLE_${currentRole.toUpperCase()}`
                },
                body: JSON.stringify({
                    question: query,
                    top_k: 5,
                    use_hybrid: true,
                    use_reranker: true
                })
            });

            const data = await response.json();

            // 4. Handle Response
            loadingMsg.remove();

            if (response.ok) {
                renderAssistantResponse(data);
            } else {
                addMessage(`Error: ${data.detail || 'Failed to get response'}`, 'assistant', false, 'error');
            }
        } catch (err) {
            loadingMsg.remove();
            addMessage(`Connectivity Error: ${err.message}`, 'assistant', false, 'error');
        } finally {
            sendBtn.disabled = false;
        }
    }

    function renderAssistantResponse(data) {
        const messageDiv = addMessage(data.answer, 'assistant');

        // Store evidence for this message
        if (data.citations && data.citations.length > 0) {
            const metaDiv = messageDiv.querySelector('.message-meta');
            const evidenceBtn = document.createElement('button');
            evidenceBtn.className = 'view-evidence-btn';
            evidenceBtn.innerHTML = '<i class="fa-solid fa-file-contract"></i> View Evidence';
            evidenceBtn.onclick = () => showEvidence(data.citations);
            metaDiv.appendChild(evidenceBtn);

            // Add click listener to citation tags if present in text
            const contentDiv = messageDiv.querySelector('.message-content');
            contentDiv.innerHTML = contentDiv.innerHTML.replace(
                /\[Source: (.*?), Page (\d+)\]/g,
                '<span class="citation-tag" onclick="window.highlightSource(\'$1\', \'$2\')">[Source: $1, Page $2]</span>'
            );
        }
    }

    function showEvidence(citations) {
        evidenceList.innerHTML = '';
        evidenceSidebar.classList.remove('collapsed');

        citations.forEach(src => {
            const clone = sourceCardTemplate.content.cloneNode(true);
            clone.querySelector('.source-file').textContent = src.source_file;
            clone.querySelector('.source-page').textContent = `Page ${src.page_number}`;
            clone.querySelector('.source-text').textContent = src.content || src.page_content || "Evidence text unavailable.";
            clone.querySelector('.doc-type').textContent = src.doc_type || 'Regulatory Doc';
            evidenceList.appendChild(clone);
        });
    }

    // Global helper for citation clicks
    window.highlightSource = (filename, page) => {
        evidenceSidebar.classList.remove('collapsed');
        // Simple scroll into view for the relevant card
        const cards = evidenceList.querySelectorAll('.source-card');
        for (let card of cards) {
            if (card.querySelector('.source-file').textContent === filename &&
                card.querySelector('.source-page').textContent === `Page ${page}`) {
                card.scrollIntoView({ behavior: 'smooth' });
                card.style.borderColor = 'var(--accent-blue)';
                setTimeout(() => card.style.borderColor = '', 2000);
                break;
            }
        }
    };

    function addMessage(text, side, isLoading = false, type = '') {
        const clone = messageTemplate.content.cloneNode(true);
        const div = clone.querySelector('.message');
        div.classList.add(side);
        if (type) div.classList.add(type);

        const content = div.querySelector('.message-content');
        if (isLoading) {
            content.innerHTML = `<span class="typing-indicator"><span></span><span></span><span></span></span> ${text}`;
        } else {
            // Basic markdown-ish formatting for bold
            content.innerHTML = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        }

        const meta = div.querySelector('.message-meta');
        meta.textContent = side === 'user' ? 'You' : 'FCA Assistant';

        chatMessages.appendChild(clone);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return chatMessages.lastElementChild;
    }

    function addSystemMessage(text) {
        const div = document.createElement('div');
        div.className = 'system-log';
        div.innerHTML = `<i class="fa-solid fa-circle-info"></i> ${text}`;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function autoResizeTextarea() {
        queryInput.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }
});
