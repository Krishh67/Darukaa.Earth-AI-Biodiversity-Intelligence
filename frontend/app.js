const API_URL = 'http://localhost:8000';
let sessionId = null;

const locSelect = document.getElementById('location-select');
const customCoords = document.getElementById('custom-coords');
const customLat = document.getElementById('custom-lat');
const customLon = document.getElementById('custom-lon');

locSelect.addEventListener('change', (e) => {
    if (e.target.value === 'custom') {
        customCoords.classList.remove('hidden');
    } else {
        customCoords.classList.add('hidden');
    }
});

document.getElementById('chat-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;

    input.value = '';
    appendMessage('user', msg);

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
                conversation_id: sessionId
            })
        });

        if (!response.ok) throw new Error('API Error');

        const data = await response.json();
        sessionId = data.conversation_id;
        
        appendMessage('assistant', data.answer);
        updateDashboard(data);
        
    } catch (err) {
        console.error(err);
        appendMessage('assistant', 'Sorry, an error occurred while processing your request.');
    } finally {
        document.getElementById('loading-overlay').classList.add('hidden');
    }
});

function appendMessage(role, text) {
    const container = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = 'flex items-start fade-in ' + (role === 'user' ? 'justify-end' : '');
    
    const inner = document.createElement('div');
    inner.className = role === 'user' 
        ? 'bg-blue-600 text-white rounded-lg p-3 max-w-[80%] shadow-sm'
        : 'bg-green-100 text-green-900 rounded-lg p-3 max-w-[80%] shadow-sm';
    inner.textContent = text;
    
    div.appendChild(inner);
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
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
    document.getElementById('rec-conf').textContent = data.confidence;
    document.getElementById('rec-reasoning').textContent = data.scientific_reasoning;
    
    const mContainer = document.getElementById('rec-metrics');
    mContainer.innerHTML = '';
    data.impacted_metrics.forEach(m => {
        mContainer.innerHTML += `<span class="bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full border border-green-200">${m}</span>`;
    });

    // Verification
    document.getElementById('verification-panel').classList.remove('hidden');
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

