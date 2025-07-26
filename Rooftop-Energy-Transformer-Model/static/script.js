// Initialize map with satellite layer
var map = L.map('map', {
    center: [20, 0],
    zoom: 2,
    minZoom: 2,
    maxZoom: 20,
    zoomSnap: 0.1,  // Allow fine-grained zoom levels
    zoomDelta: 0.1  // Allow fine-grained zoom control
});

// Add Google Satellite layer with higher resolution
L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
    maxZoom: 20,
    tileSize: 256,
    zoomOffset: 0,
    subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
    attribution: '© Google',
    detectRetina: true
}).addTo(map);

// Initialize drawing controls
var drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

var drawControl = new L.Control.Draw({
    draw: {
        polygon: false,
        circle: false,
        circlemarker: false,
        polyline: false,
        marker: false,
        rectangle: {
            shapeOptions: {
                color: '#0083b0',
                weight: 2
            }
        }
    },
    edit: {
        featureGroup: drawnItems,
        remove: true
    }
});
map.addControl(drawControl);

// Update file input display
document.getElementById('file-input').addEventListener('change', function(e) {
    const fileName = e.target.files[0]?.name || 'Choose File';
    document.getElementById('file-name').textContent = fileName;
});

// Handle drawing events
var currentROI = null;
map.on('draw:created', function(e) {
    drawnItems.clearLayers();
    currentROI = e.layer;
    drawnItems.addLayer(currentROI);
    document.getElementById('analyzeROI').disabled = false;
});

map.on('draw:deleted', function(e) {
    currentROI = null;
    document.getElementById('analyzeROI').disabled = true;
});

function formatEnergyUnit(value) {
    if (value >= 1000000) {
        return `${(value/1000000).toFixed(2)} GWh/year`;
    } else if (value >= 1000) {
        return `${(value/1000).toFixed(2)} MWh/year`;
    }
    return `${Math.round(value)} kWh/year`;
}

function formatArea(value) {
    if (value >= 1000000) {
        return `${(value/1000000).toFixed(2)} km²`;
    }
    return `${Math.round(value)} m²`;
}

// Handle form submission
document.getElementById('uploadForm').onsubmit = function(e) {
    e.preventDefault();
    var formData = new FormData(this);
    submitAnalysis(formData, 'process');
};

// Handle ROI analysis
document.getElementById('analyzeROI').onclick = function() {
    if (!currentROI) return;
    
    var bounds = currentROI.getBounds();
    var data = {
        north: bounds.getNorth(),
        south: bounds.getSouth(),
        east: bounds.getEast(),
        west: bounds.getWest()
    };
    submitAnalysis(data, 'process_roi');
};

function submitAnalysis(data, endpoint) {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    document.getElementById('results').innerHTML = '';

    fetch('/' + endpoint, {
        method: 'POST',
        body: data instanceof FormData ? data : JSON.stringify(data),
        headers: data instanceof FormData ? {} : {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        if (data.error) {
            document.getElementById('results').innerHTML = `
                <div class="error">
                    <i class="fas fa-exclamation-circle"></i>
                    ${data.error}
                </div>`;
            return;
        }

        // Ensure panel_info exists
        data.panel_info = data.panel_info || {
            type: 'Monocrystalline PERC',
            efficiency: 18,
            lifespan: 25
        };

        // Display results
        var resultsHtml = `
            <h2 class="section-title">
                <i class="fas fa-chart-pie"></i>
                Analysis Results
            </h2>
            <div class="results-grid">
                <div>
                    <h3 class="section-title">Input Image</h3>
                    <img src="/results/${data.input_file}" alt="Input Image" class="result-image">
                </div>
                <div>
                    <h3 class="section-title">Detection Mask</h3>
                    <img src="/results/${data.mask_file}" alt="Detection Mask" class="result-image">
                </div>
            </div>
            <div class="solar-potential">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                    <div class="metric-card">
                        <i class="fas fa-solar-panel" style="color: #0083b0; font-size: 2em;"></i>
                        <div class="metric-value">${formatEnergyUnit(data.total_generation)}</div>
                        <div class="metric-label">Annual Energy Generation</div>
                    </div>
                    <div class="metric-card">
                        <i class="fas fa-expand" style="color: #0083b0; font-size: 2em;"></i>
                        <div class="metric-value">${data.total_area}</div>
                        <div class="metric-label">Total Rooftop Area</div>
                    </div>
                </div>
                <div class="info-card" style="margin-top: 20px; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);">
                    <h3 class="section-title" style="margin-bottom: 15px;">
                        <i class="fas fa-info-circle"></i>
                        Technical Information
                    </h3>
                    <div>
                        <strong>Solar Panel Specifications:</strong>
                        <ul style="list-style: none; padding-left: 20px; margin-top: 5px;">
                            <li><i class="fas fa-check" style="color: #0083b0; margin-right: 8px;"></i>Type: ${data.panel_info.type}</li>
                            <li><i class="fas fa-check" style="color: #0083b0; margin-right: 8px;"></i>Efficiency: ${data.panel_info.efficiency}%</li>
                            <li><i class="fas fa-check" style="color: #0083b0; margin-right: 8px;"></i>Expected Lifespan: ${data.panel_info.lifespan} years</li>
                        </ul>
                    </div>
                </div>
            </div>
        `;
                document.getElementById('results').innerHTML = resultsHtml;
                document.getElementById('results').style.display = 'block';
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('results').style.display = 'block';
        document.getElementById('results').innerHTML = `
            <div class="error">
                <i class="fas fa-exclamation-circle"></i>
                Error: ${error.message}
            </div>`;
    });
}

// Try to get user's location
if ("geolocation" in navigator) {
    navigator.geolocation.getCurrentPosition(function(position) {
        map.setView([position.coords.latitude, position.coords.longitude], 15);
    });
}

// Theme switcher logic
const themeSwitcher = document.querySelector('.theme-switcher i');
const body = document.body;

themeSwitcher.addEventListener('click', () => {
    body.classList.toggle('dark-mode');
    const isDarkMode = body.classList.contains('dark-mode');
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
    updateThemeIcon(isDarkMode);
});

function updateThemeIcon(isDarkMode) {
    if (isDarkMode) {
        themeSwitcher.classList.remove('fa-sun');
        themeSwitcher.classList.add('fa-moon');
    } else {
        themeSwitcher.classList.remove('fa-moon');
        themeSwitcher.classList.add('fa-sun');
    }
}

// Apply saved theme on load
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'dark') {
    body.classList.add('dark-mode');
    updateThemeIcon(true);
} else {
    updateThemeIcon(false);
}
