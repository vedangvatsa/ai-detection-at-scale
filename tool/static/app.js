// --- DOM Elements ---
const textInput = document.getElementById('text-input');
const wordCountEl = document.getElementById('word-count');
const registerOverride = document.getElementById('register-override');
const analyzeBtn = document.getElementById('analyze-btn');
const errorBanner = document.getElementById('error-banner');
const errorMessage = document.getElementById('error-message');
const resultsSection = document.getElementById('results-section');

// Gauge elements
const progressRingBar = document.getElementById('progress-ring-bar');
const probabilityVal = document.getElementById('probability-val');
const classificationBadge = document.getElementById('classification-badge');

// Diagnostics elements
const detectedRegisterEl = document.getElementById('detected-register');
const confidenceRow = document.getElementById('confidence-row');
const routerConfidenceEl = document.getElementById('router-confidence');
const processingTimeEl = document.getElementById('processing-time');
const featuresGrid = document.getElementById('features-grid');

// SVG Ring Initialization
const radius = 90;
const circumference = 2 * Math.PI * radius;
progressRingBar.style.strokeDasharray = `${circumference} ${circumference}`;
progressRingBar.style.strokeDashoffset = circumference;

// --- Feature Meta Mapping for Nice Display Names & Progress Bar Scales ---
const FEATURE_META = {
    'mtld': { label: 'Lexical Diversity (MTLD)', max: 150, format: (v) => v.toFixed(1) },
    'sent_cv': { label: 'Sentence Length Variation (CV)', max: 1.5, format: (v) => v.toFixed(2) },
    'char_entropy': { label: 'Character N-gram Entropy', max: 6.0, format: (v) => v.toFixed(2) },
    'rep_rate': { label: 'Within-Doc Word Repetition Rate', max: 1.0, format: (v) => `${(v * 100).toFixed(0)}%` },
    'punct_entropy': { label: 'Punctuation Entropy', max: 4.0, format: (v) => v.toFixed(2) },
    'self_mention_density': { label: 'First-Person Mentions (per 1k words)', max: 50, format: (v) => v.toFixed(1) },
    'connector_density': { label: 'Connector Words (per 1k words)', max: 40, format: (v) => v.toFixed(1) },
    'hedge_density': { label: 'Hedges / Qualifiers (per 1k words)', max: 30, format: (v) => v.toFixed(1) },
    'boost_density': { label: 'Boosters / Assertions (per 1k words)', max: 20, format: (v) => v.toFixed(1) },
    'mean_sent_len': { label: 'Mean Sentence Length (words)', max: 40, format: (v) => v.toFixed(1) },
    'opener_ratio': { label: 'Sentence-Opener Connector Ratio', max: 0.5, format: (v) => `${(v * 100).toFixed(0)}%` }
};

// --- Word Counter Handler ---
textInput.addEventListener('input', () => {
    const text = textInput.value.trim();
    if (!text) {
        wordCountEl.textContent = 0;
        return;
    }
    const words = text.split(/\s+/).filter(w => w.length > 0);
    wordCountEl.textContent = words.length;
});

// --- Gauge Ring Progress Animator ---
function setProgress(percent, colorVar, glowVar) {
    const offset = circumference - (percent / 100) * circumference;
    progressRingBar.style.strokeDashoffset = offset;
    progressRingBar.style.stroke = `var(${colorVar})`;
    progressRingBar.style.filter = `drop-shadow(0 0 8px var(${glowVar}))`;
}

// --- Submit & API Call ---
analyzeBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    const register = registerOverride.value;

    if (!text) {
        showError('Please paste some text to analyze.');
        return;
    }

    const words = text.split(/\s+/).filter(w => w.length > 0);
    if (words.length < 5) {
        showError('Text is too short. Please provide at least 5 words.');
        return;
    }

    // Reset UI
    clearError();
    resultsSection.classList.add('hidden');
    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                register: register || null,
                return_features: true
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'API request failed');
        }

        const data = await response.json();
        renderResults(data);
    } catch (err) {
        showError(err.message || 'An error occurred during analysis.');
    } finally {
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;
    }
});

// --- UI Rendering of Results ---
function renderResults(data) {
    resultsSection.classList.remove('hidden');

    // 1. Animate Gauge & Badges
    const pct = Math.round(data.ai_probability * 100);
    probabilityVal.textContent = `${pct}%`;

    let colorVar, glowVar, badgeText, badgeClass;
    if (data.ai_probability >= 0.7) {
        colorVar = '--color-ai';
        glowVar = '--color-ai-glow';
        badgeText = 'AI GENERATED';
        badgeClass = 'bg-ai';
    } else if (data.ai_probability >= 0.35) {
        colorVar = '--color-mixed';
        glowVar = '--color-mixed-glow';
        badgeText = 'MIXED / UNCERTAIN';
        badgeClass = 'bg-mixed';
    } else {
        colorVar = '--color-human';
        glowVar = '--color-human-glow';
        badgeText = 'HUMAN WRITTEN';
        badgeClass = 'bg-human';
    }

    classificationBadge.textContent = badgeText;
    classificationBadge.className = 'classification-badge';
    classificationBadge.style.backgroundColor = `var(${colorVar})`;
    classificationBadge.style.color = '#ffffff';
    classificationBadge.style.boxShadow = `0 4px 10px var(${glowVar})`;

    // Delay slightly to trigger transition animation
    setTimeout(() => {
        setProgress(pct, colorVar, glowVar);
    }, 50);

    // 2. Render Diagnostics
    detectedRegisterEl.textContent = data.register.charAt(0).toUpperCase() + data.register.slice(1);
    
    if (data.register_confidence !== null && data.register_confidence !== undefined) {
        confidenceRow.classList.remove('hidden');
        routerConfidenceEl.textContent = `${Math.round(data.register_confidence * 100)}%`;
    } else {
        confidenceRow.classList.add('hidden');
    }

    processingTimeEl.textContent = `${data.processing_time_ms.toFixed(1)} ms`;

    // 3. Render Features Grid
    featuresGrid.innerHTML = '';
    if (data.features) {
        Object.entries(FEATURE_META).forEach(([key, meta]) => {
            const val = data.features[key];
            if (val === undefined || val === null) return;

            const displayVal = meta.format(val);
            const percentage = Math.min((val / meta.max) * 100, 100);

            const item = document.createElement('div');
            item.className = 'feat-item';
            item.innerHTML = `
                <div class="feat-header">
                    <span class="feat-name">${meta.label}</span>
                    <span class="feat-val">${displayVal}</span>
                </div>
                <div class="feat-track">
                    <div class="feat-bar" style="width: ${percentage}%"></div>
                </div>
            `;
            featuresGrid.appendChild(item);
        });
    }
    
    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// --- Helper Utilities ---
function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.classList.remove('hidden');
}

function clearError() {
    errorBanner.classList.add('hidden');
}
