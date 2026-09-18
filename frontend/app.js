const API_URL = 'http://localhost:8000';
let sessionId = null;

const locSelect = document.getElementById('location-select');
const customCoords = document.getElementById('custom-coords');
const customLat = document.getElementById('custom-lat');
const customLon = document.getElementById('custom-lon');
const sessionList = document.getElementById('session-list');
const btnNewChat = document.getElementById('btn-new-chat');
const btnToggleJson = document.getElementById('btn-toggle-json');
const jsonContainer = document.getElementById('json-input-container');
const jsonInput = document.getElementById('json-input');
const verifyToggle = document.getElementById('verify-toggle');
const chatHistory = document.getElementById('chat-history');

// Sidebar toggle elements
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebar-overlay');
const btnMenu = document.getElementById('btn-menu');
const btnCloseMenu = document.getElementById('btn-close-menu');

// Toggle Sidebar functions
function openSidebar() {
    sidebar.classList.remove('-translate-x-full');
    sidebarOverlay.classList.remove('hidden');
}
function closeSidebar() {
    sidebar.classList.add('-translate-x-full');
    sidebarOverlay.classList.add('hidden');
}

btnMenu.addEventListener('click', openSidebar);
btnCloseMenu.addEventListener('click', closeSidebar);
sidebarOverlay.addEventListener('click', closeSidebar);

// On load, fetch sessions
window.addEventListener('DOMContentLoaded', fetchSessions);

// Sidebar: Fetch and render sessions
async function fetchSessions() {
    try {
        const res = await fetch(`${API_URL}/sessions`);
        const sessions = await res.json();
        sessionList.innerHTML = '';
        if (sessions.length === 0) {
            sessionList.innerHTML = '<div class="text-gray-400 text-sm p-2 text-center italic">No history yet.</div>';
            return;
        }
        sessions.forEach(s => {
            const btn = document.createElement('button');
            btn.className = 'w-full text-left p-3 rounded-lg text-sm mb-1 hover:bg-gray-800 transition-colors border border-transparent';
            if (s.id === sessionId) {
                btn.classList.add('bg-gray-800', 'border-gray-700');
            }
            btn.innerHTML = `<div class="font-semibold text-gray-200 truncate">${s.title}</div><div class="text-xs text-gray-500 mt-1">${new Date(s.updated_at).toLocaleString()}</div>`;
            btn.onclick = () => loadSession(s.id);
            sessionList.appendChild(btn);
        });
    } catch (err) {
        console.error('Failed to load sessions', err);
    }
}

// Load a specific session
async function loadSession(id) {
    try {
        document.getElementById('loading-overlay').classList.remove('hidden');
        const res = await fetch(`${API_URL}/sessions/${id}`);
        const data = await res.json();
        
        sessionId = data.id;
        document.getElementById('session-badge').textContent = 'Active Session';
        
        // Clear chat UI
        chatHistory.innerHTML = '';
        data.history.forEach(msg => {
            appendMessage(msg.role, msg.content);
        });
        
        // Hide right panels
        document.getElementById('recommendation-panel').classList.add('hidden');
        document.getElementById('verification-panel').classList.add('hidden');
        document.getElementById('evidence-panel').classList.add('hidden');
        document.getElementById('env-metrics').innerHTML = '<div class="text-sm text-gray-500 italic">History loaded. Send a query to see metrics.</div>';
        
        fetchSessions(); // re-render sidebar to highlight active
    } catch (err) {
        console.error('Failed to load session', err);
    } finally {
        document.getElementById('loading-overlay').classList.add('hidden');
    }
}

// New Chat
btnNewChat.addEventListener('click', () => {
    sessionId = null;
    document.getElementById('session-badge').textContent = 'New Session';
    chatHistory.innerHTML = `
        <div class="flex items-start welcome-msg">
            <div class="bg-green-100 text-green-900 rounded-lg p-3 max-w-[80%] shadow-sm">
                Hello! I am the Darukaa.Earth Biodiversity Assistant. Select a location above and ask me how to improve biodiversity, assess soil health, or restore a habitat.
            </div>
        </div>
    `;
    document.getElementById('recommendation-panel').classList.add('hidden');
    document.getElementById('verification-panel').classList.add('hidden');
    document.getElementById('evidence-panel').classList.add('hidden');
    document.getElementById('env-metrics').innerHTML = '<div class="text-sm text-gray-500 italic">No data yet. Send a query with a location to analyze.</div>';
    fetchSessions(); // unhighlight sidebar
});

// JSON Toggle
btnToggleJson.addEventListener('click', (e) => {
    e.preventDefault();
    if (jsonContainer.classList.contains('hidden')) {
        jsonContainer.classList.remove('hidden');
        btnToggleJson.innerHTML = '<i class="fas fa-times mr-1"></i>Hide JSON Input';
    } else {
        jsonContainer.classList.add('hidden');
        jsonInput.value = '';
        btnToggleJson.innerHTML = '<i class="fas fa-code mr-1"></i>Structured JSON Input';
    }
});

// Location Select Logic
locSelect.addEventListener('change', (e) => {
    if (e.target.value === 'custom') {
        customCoords.classList.remove('hidden');
    } else {
        customCoords.classList.add('hidden');
    }
});

// Chat Form Submit
document.getElementById('chat-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;

    // Parse JSON override
    let envOverride = null;
    if (jsonInput.value.trim()) {
        try {
            envOverride = JSON.parse(jsonInput.value.trim());
        } catch (err) {
            alert('Invalid JSON in structured input box!');
            return;
        }
    }

    input.value = '';
    appendMessage('user', msg);

    // Parse Location
    let lat = null;
    let lon = null;
    
    if (locSelect.value === 'custom') {
        lat = parseFloat(customLat.value);
        lon = parseFloat(customLon.value);
    } else if (locSelect.value) {
        [lat, lon] = locSelect.value.split(',').map(Number);
    }

    if (lat === null || lon === null || isNaN(lat) || isNaN(lon)) {
        appendMessage('assistant', 'Please select a location or enter valid custom coordinates from the top menu before proceeding.');
        return;
    }

    document.getElementById('loading-overlay').classList.remove('hidden');

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                latitude: lat,
                longitude: lon,
                conversation_id: sessionId,
                env_override: envOverride,
                enable_verification: verifyToggle.checked
            })
        });

        if (!response.ok) throw new Error('API Error');

        const data = await response.json();
        sessionId = data.conversation_id;
        
        appendMessage('assistant', data.answer);
        
        if (!data.is_clarification) {
            updateDashboard(data);
        }
        
        // Refresh sidebar to show the updated session list
        fetchSessions();
        
    } catch (err) {
        console.error(err);
        appendMessage('assistant', 'Sorry, an error occurred while processing your request.');
    } finally {
        document.getElementById('loading-overlay').classList.add('hidden');
    }
});

function appendMessage(role, text) {
    const div = document.createElement('div');
    div.className = 'flex items-start fade-in ' + (role === 'user' ? 'justify-end' : '');
    
    const inner = document.createElement('div');
    inner.className = role === 'user' 
        ? 'bg-blue-600 text-white rounded-lg p-3 max-w-[80%] shadow-sm'
        : 'bg-green-100 text-green-900 rounded-lg p-3 max-w-[80%] shadow-sm whitespace-pre-wrap';
    inner.textContent = text;
    
    div.appendChild(inner);
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function updateDashboard(data) {
    // Environment
    const envBox = document.getElementById('env-metrics');
    envBox.innerHTML = '';
    
    const renderMetric = (label, value, icon, unit='') => {
        let valStr = value !== null && value !== undefined ? `${value} ${unit}` : 'Data unavailable';
        return `
            <div class="bg-gray-50 rounded p-3 border border-gray-200">
                <div class="text-xs text-gray-500 uppercase font-semibold mb-1"><i class="${icon} mr-1"></i>${label}</div>
                <div class="text-gray-800 font-bold ${value===null?'text-sm italic':''}">${valStr}</div>
            </div>
        `;
    };

    let html = '';
    if (data.environment.soil) {
        const s = data.environment.soil;
        html += renderMetric('Soil pH', s.ph, 'fas fa-vial');
        html += renderMetric('Organic Carbon', s.soc, 'fas fa-leaf', 'g/kg');
        html += renderMetric('Nitrogen', s.nitrogen, 'fas fa-atom', 'g/kg');
        html += renderMetric('Bulk Density', s.bulk_density, 'fas fa-weight-hanging', 'g/cm³');
    }
    if (data.environment.land_cover) {
        html += renderMetric('Land Cover', data.environment.land_cover.class_name, 'fas fa-layer-group');
    }
    if (data.environment.climate) {
        html += renderMetric('Ann. Temp', data.environment.climate.temperature_annual_mean, 'fas fa-thermometer-half', '°C');
        html += renderMetric('Ann. Rainfall', data.environment.climate.precipitation_annual_sum, 'fas fa-cloud-rain', 'mm');
    }
    envBox.innerHTML = html || '<div class="col-span-2 text-sm text-gray-500 italic">No environmental data available for these coordinates.</div>';

    // Recommendation
    document.getElementById('recommendation-panel').classList.remove('hidden');
    document.getElementById('rec-text').textContent = data.recommendation;
    document.getElementById('rec-time').textContent = data.time_horizon;
    
    // Upgraded Confidence UI
    const conf = (data.confidence || '').toLowerCase();
    const confContainer = document.getElementById('rec-conf-container');
    const confIcon = document.getElementById('rec-conf-icon');
    
    // Base classes for perfect alignment (h-8 ensures it matches the time pill exactly)
    const baseClasses = 'flex items-center px-3 py-1.5 rounded-md text-xs font-bold border shadow-sm h-8 transition-colors duration-200 ';
    
    if (conf.includes('high')) {
        confContainer.className = baseClasses + 'bg-emerald-50 text-emerald-700 border-emerald-200';
        confIcon.className = 'fas fa-check-circle mr-1.5';
    } else if (conf.includes('medium')) {
        confContainer.className = baseClasses + 'bg-yellow-50 text-yellow-700 border-yellow-200';
        confIcon.className = 'fas fa-exclamation-circle mr-1.5';
    } else {
        confContainer.className = baseClasses + 'bg-red-50 text-red-700 border-red-200';
        confIcon.className = 'fas fa-times-circle mr-1.5';
    }
    document.getElementById('rec-conf').textContent = data.confidence;
    
    document.getElementById('rec-reasoning').textContent = data.scientific_reasoning;
    
    const mContainer = document.getElementById('rec-metrics');
    mContainer.innerHTML = '';
    data.impacted_metrics.forEach(m => {
        mContainer.innerHTML += `<span class="bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full border border-green-200">${m}</span>`;
    });

    // Verification Node Logic
    const vPanel = document.getElementById('verification-panel');
    if (data.verification.disabled) {
        vPanel.classList.add('hidden');
    } else {
        vPanel.classList.remove('hidden');
        const vStatus = document.getElementById('verification-status');
        const vFlags = document.getElementById('verification-flags');
        vFlags.innerHTML = '';
        
        if (data.verification.is_supported) {
            vStatus.innerHTML = `<i class="fas fa-check-circle text-green-500 text-xl"></i><span class="font-bold text-green-700">Verified & Grounded</span>`;
            vFlags.innerHTML = `<li>All scientific claims align with retrieved evidence.</li><li>No fabricated statistics detected.</li>`;
        } else {
            vStatus.innerHTML = `<i class="fas fa-exclamation-triangle text-orange-500 text-xl"></i><span class="font-bold text-orange-700">Issues Detected & Revised</span>`;
            data.verification.flags.forEach(f => {
                vFlags.innerHTML += `<li>${f}</li>`;
            });
        }
    }

    // Evidence
    document.getElementById('evidence-panel').classList.remove('hidden');
    const evList = document.getElementById('evidence-list');
    evList.innerHTML = '';
    data.evidence.forEach((ev, idx) => {
        evList.innerHTML += `
            <div class="bg-yellow-50 rounded p-3 border border-yellow-200 text-sm">
                <div class="font-bold text-yellow-800 mb-1">[Source ${idx+1}] ${ev.document_title || 'Research Document'}</div>
                <div class="text-gray-700 italic">"${ev.text.substring(0, 200)}..."</div>
            </div>
        `;
    });
}
