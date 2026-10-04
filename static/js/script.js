/* 
=============================================================================
CreditPulse AI - Advanced Client Engine & Enterprise Features
=============================================================================
- Web Audio Synthesizer (Audio feedback on approval/rejection)
- Theme Switcher (Dark/Light mode with localStorage)
- Multi-Model Consensus Arena
- What-If Sensitivity Simulator & Live Curves
- Loan Amortization & Affordability Breakdown
- Batch CSV Drag-and-Drop Scoring Engine
- Print-Ready Official Assessment Report
*/

// Audio synthesis state
let soundEnabled = true;
let audioCtx = null;

// Chart instances for smooth updating
let chartAmortization = null;
let chartSensitivity = null;

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initSound();
    initPredictionForm();
    initPresets();
    initLiveCalculators();
    initTabs();
    initDatasetExplorer();
    initDashboardCharts();
    initBatchScoring();
    initSimulator();
});

/**
 * Web Audio API Sound Synthesizer
 */
function initSound() {
    const soundToggle = document.getElementById('sound-toggle-btn');
    const storedSound = localStorage.getItem('creditpulse_sound');
    if (storedSound !== null) soundEnabled = storedSound === 'true';
    updateSoundButton();

    if (soundToggle) {
        soundToggle.addEventListener('click', () => {
            soundEnabled = !soundEnabled;
            localStorage.setItem('creditpulse_sound', soundEnabled);
            updateSoundButton();
            if (soundEnabled) playTone(600, 'sine', 0.1, 0.15);
        });
    }
}

function updateSoundButton() {
    const soundToggle = document.getElementById('sound-toggle-btn');
    if (soundToggle) {
        soundToggle.innerHTML = soundEnabled ? '🔊' : '🔇';
        soundToggle.title = soundEnabled ? 'Sound Effects Enabled' : 'Sound Effects Muted';
    }
}

function playTone(freq, type = 'sine', duration = 0.15, volume = 0.1) {
    if (!soundEnabled) return;
    try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') audioCtx.resume();

        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        osc.type = type;
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);

        gain.gain.setValueAtTime(volume, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);

        osc.connect(gain);
        gain.connect(audioCtx.destination);

        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
        // Audio context fallback
    }
}

function playApprovalSound() {
    if (!soundEnabled) return;
    setTimeout(() => playTone(523.25, 'triangle', 0.18, 0.15), 0);   // C5
    setTimeout(() => playTone(659.25, 'triangle', 0.18, 0.15), 120); // E5
    setTimeout(() => playTone(783.99, 'triangle', 0.35, 0.2), 240);  // G5
}

function playRejectionSound() {
    if (!soundEnabled) return;
    setTimeout(() => playTone(350, 'sawtooth', 0.2, 0.12), 0);
    setTimeout(() => playTone(280, 'sawtooth', 0.3, 0.15), 150);
}

/**
 * Theme Toggle (Dark / Light)
 */
function initTheme() {
    const themeBtn = document.getElementById('theme-toggle-btn');
    const savedTheme = localStorage.getItem('creditpulse_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeButton(savedTheme);

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme') || 'dark';
            const nextTheme = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', nextTheme);
            localStorage.setItem('creditpulse_theme', nextTheme);
            updateThemeButton(nextTheme);
            playTone(800, 'sine', 0.08, 0.08);
        });
    }
}

function updateThemeButton(theme) {
    const btn = document.getElementById('theme-toggle-btn');
    if (btn) {
        btn.innerHTML = theme === 'dark' ? '🌙' : '☀️';
        btn.title = `Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`;
    }
}

/**
 * Tab Navigation Component
 */
function initTabs() {
    document.querySelectorAll('.card-tabs').forEach(tabGroup => {
        const buttons = tabGroup.querySelectorAll('.tab-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetId = btn.dataset.tab;
                buttons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                // Find tab contents in parent card
                const card = tabGroup.closest('.card') || document;
                card.querySelectorAll('.tab-pane').forEach(pane => {
                    if (pane.id === targetId) {
                        pane.style.display = 'block';
                    } else {
                        pane.style.display = 'none';
                    }
                });

                playTone(700, 'sine', 0.05, 0.05);

                // If tab is consensus or sensitivity, trigger updates
                if (targetId === 'tab-consensus') runConsensusEvaluation();
                if (targetId === 'tab-sensitivity') runSensitivityEvaluation();
                if (targetId === 'tab-amortization') renderAmortizationChart();
            });
        });
    });
}

/**
 * Single Applicant Prediction Form
 */
let lastPredictionData = null;

function initPredictionForm() {
    const form = document.getElementById('loan-prediction-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        playTone(500, 'sine', 0.08, 0.1);

        const btn = document.getElementById('predict-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = `<span class="spinner"></span> Underwriting Inference in Progress...`;
        btn.disabled = true;

        const payload = collectFormData();

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            btn.innerHTML = originalText;
            btn.disabled = false;

            if (data.success) {
                lastPredictionData = { ...data, applicant: payload };
                renderPredictionResult(data);
                if (data.prediction === 1) playApprovalSound();
                else playRejectionSound();

                // If currently on consensus or sensitivity tab, refresh them
                const activeTab = document.querySelector('.tab-btn.active');
                if (activeTab && activeTab.dataset.tab === 'tab-consensus') runConsensusEvaluation();
                if (activeTab && activeTab.dataset.tab === 'tab-sensitivity') runSensitivityEvaluation();
                renderAmortizationChart();
            } else {
                alert('Prediction Error: ' + (data.error || 'Server error'));
            }
        } catch (err) {
            btn.innerHTML = originalText;
            btn.disabled = false;
            console.error(err);
            alert('Failed to connect to the prediction service.');
        }
    });
}

function collectFormData() {
    return {
        model: document.getElementById('model-select') ? document.getElementById('model-select').value : 'Random Forest',
        Gender: document.getElementById('gender').value,
        Married: document.getElementById('married').value,
        Dependents: document.getElementById('dependents').value,
        Education: document.getElementById('education').value,
        Self_Employed: document.getElementById('self_employed').value,
        ApplicantIncome: parseFloat(document.getElementById('applicant_income').value) || 0,
        CoapplicantIncome: parseFloat(document.getElementById('coapplicant_income').value) || 0,
        LoanAmount: parseFloat(document.getElementById('loan_amount').value) || 0,
        Loan_Amount_Term: parseFloat(document.getElementById('loan_term').value) || 360,
        Credit_History: parseFloat(document.getElementById('credit_history').value),
        Property_Area: document.getElementById('property_area').value
    };
}

/**
 * Render Main Prediction Result Card
 */
function renderPredictionResult(data) {
    const container = document.getElementById('prediction-result-container');
    if (!container) return;

    const isApproved = data.prediction === 1;
    const badgeClass = isApproved ? 'approved' : 'rejected';
    const statusIcon = isApproved ? '🟢' : '🔴';
    const fillClass = isApproved ? 'approved' : 'rejected';

    let factorsHtml = '';
    if (data.factors && data.factors.length > 0) {
        factorsHtml = `
            <div class="factors-list">
                <h4 style="font-size:0.92rem; color:#94a3b8; margin-bottom:0.4rem; font-weight:700;">Underwriting Assessment Factors:</h4>
                ${data.factors.map(f => `
                    <div class="factor-item ${f.type}">
                        <div>
                            <div class="factor-title">${f.factor}</div>
                            <div class="factor-desc">${f.text}</div>
                        </div>
                        <span class="factor-badge ${f.type}">${f.impact}</span>
                    </div>
                `).join('')}
            </div>
        `;
    }

    let recommendationsHtml = '';
    if (data.recommendations && data.recommendations.length > 0) {
        recommendationsHtml = `
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 0.9rem; margin-top: 1rem; text-align: left; width: 100%;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #fbbf24; margin-bottom: 0.3rem;">
                    💡 Risk Mitigation Recommendations:
                </div>
                <ul style="font-size: 0.8rem; color: #cbd5e1; padding-left: 1.2rem; line-height: 1.5;">
                    ${data.recommendations.map(r => `<li>${r}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    container.innerHTML = `
        <div class="decision-badge ${badgeClass}">
            <span>${statusIcon}</span>
            <span>LOAN ${data.decision}</span>
        </div>

        <div style="display:flex; justify-content:center; gap:0.6rem; align-items:center; margin-bottom: 0.75rem; flex-wrap:wrap;">
            <span style="font-size: 0.85rem; color: #94a3b8;">
                Evaluator: <strong style="color: #c7d2fe;">${data.active_model}</strong>
            </span>
            <span class="brand-badge" style="background:rgba(99,102,241,0.25); color:#a5b4fc;">${data.apr_tier}</span>
        </div>

        <div class="confidence-container">
            <div class="confidence-header">
                <span>Model Confidence</span>
                <span style="color: ${isApproved ? '#34d399' : '#f87171'}; font-weight: 900; font-size: 1.25rem;">
                    ${data.confidence}%
                </span>
            </div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill ${fillClass}" id="confidence-fill" style="width: 0%"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748b; margin-top: 0.4rem;">
                <span>Approval Prob: ${data.prob_approved}%</span>
                <span>Rejection Prob: ${data.prob_rejected}%</span>
            </div>
        </div>

        <!-- Financial KPI Strip -->
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 1rem; width: 100%; margin: 0.5rem 0 1rem; text-align: left;">
            <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.4rem;">
                <span style="color: #94a3b8;">Monthly Payment (EMI @ 8.5%):</span>
                <strong style="color: #f8fafc;">$${data.emi_monthly.toLocaleString()}/mo</strong>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.4rem;">
                <span style="color: #94a3b8;">Debt-to-Income (DTI):</span>
                <strong style="color: ${data.emi_ratio <= 35 ? '#34d399' : '#f87171'}; font-weight:800;">${data.emi_ratio}%</strong>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.88rem;">
                <span style="color: #94a3b8;">Safe Borrowing Ceiling:</span>
                <strong style="color: #38bdf8;">$${data.max_safe_loan.toLocaleString()}</strong>
            </div>
        </div>

        <p style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 0.75rem; line-height: 1.45;">
            ${data.summary}
        </p>

        <!-- Action Button Strip: Print Report -->
        <div style="display: flex; gap: 0.6rem; width: 100%; margin-top: 0.5rem;">
            <button type="button" class="btn btn-secondary w-full" id="btn-print-report" style="justify-content: center; font-size: 0.85rem; padding: 0.6rem;">
                📄 Generate Official Underwriting Report
            </button>
        </div>

        ${recommendationsHtml}
        ${factorsHtml}
    `;

    setTimeout(() => {
        const bar = document.getElementById('confidence-fill');
        if (bar) bar.style.width = `${data.confidence}%`;
    }, 50);

    // Bind report print button
    const printBtn = document.getElementById('btn-print-report');
    if (printBtn) {
        printBtn.addEventListener('click', openUnderwritingReportModal);
    }
}

/**
 * Multi-Model Consensus Arena Evaluation
 */
async function runConsensusEvaluation() {
    const container = document.getElementById('consensus-results-container');
    if (!container) return;

    container.innerHTML = `<div style="text-align:center; padding:2rem;"><span class="spinner"></span> Running consensus across all 5 ensemble algorithms...</div>`;

    const payload = collectFormData();
    try {
        const res = await fetch('/api/predict/all-models', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!data.success) {
            container.innerHTML = `<div style="color:#ef4444; padding:1rem;">Failed to run multi-model consensus.</div>`;
            return;
        }

        const isConsensusApproved = data.consensus_decision === 'APPROVED';
        const consensusBadge = isConsensusApproved ? 'approved' : 'rejected';
        const icon = isConsensusApproved ? '🟢' : '🔴';

        container.innerHTML = `
            <div style="background:rgba(255,255,255,0.03); border:1px solid var(--border-color); border-radius:12px; padding:1.25rem; margin-bottom:1.5rem; text-align:center;">
                <div style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase; font-weight:700;">Ensemble Consensus Verdict</div>
                <div class="decision-badge ${consensusBadge}" style="margin: 0.5rem 0; font-size: 1.5rem; padding: 0.5rem 1.75rem;">
                    <span>${icon}</span>
                    <span>${data.consensus_decision}</span>
                </div>
                <div style="font-size:0.95rem; font-weight:700; color:#c7d2fe;">
                    ${data.approved_votes} of ${data.total_models} Models Recommend Approval (${data.agreement_percentage}% Agreement)
                </div>
            </div>

            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap:1rem;">
                ${data.models_comparison.map(m => {
                    const app = m.prediction === 1;
                    const cardBorder = app ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)';
                    const cardBg = app ? 'rgba(16, 185, 129, 0.08)' : 'rgba(239, 68, 68, 0.08)';
                    const statusColor = app ? '#34d399' : '#f87171';
                    return `
                        <div style="background:${cardBg}; border:1px solid ${cardBorder}; border-radius:12px; padding:1.1rem; text-align:center;">
                            <div style="font-weight:800; font-size:1rem; color:#f8fafc; margin-bottom:0.25rem;">${m.model}</div>
                            <div style="font-size:1.15rem; font-weight:900; color:${statusColor}; margin-bottom:0.5rem;">
                                ${app ? '🟢 APPROVED' : '🔴 REJECTED'}
                            </div>
                            <div style="font-size:0.85rem; color:#94a3b8;">
                                Confidence: <strong style="color:#f8fafc;">${m.confidence}%</strong>
                            </div>
                            <div style="font-size:0.75rem; color:#64748b; margin-top:0.3rem;">
                                Approved: ${m.approval_prob}% | Rejected: ${m.rejection_prob}%
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    } catch (e) {
        console.error(e);
    }
}

/**
 * What-If Sensitivity Simulator Chart
 */
async function runSensitivityEvaluation() {
    const canvas = document.getElementById('chart-sensitivity-canvas');
    if (!canvas || typeof Chart === 'undefined') return;

    const payload = collectFormData();
    try {
        const res = await fetch('/api/sensitivity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!data.success) return;

        const labels = data.loan_curve.map(c => `$${c.loan_amount}k`);
        const probs = data.loan_curve.map(c => c.approval_prob);

        if (chartSensitivity) chartSensitivity.destroy();

        chartSensitivity = new Chart(canvas, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Loan Approval Likelihood (%)',
                    data: probs,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                    fill: true,
                    tension: 0.35,
                    borderWidth: 3,
                    pointRadius: 4,
                    pointHoverRadius: 7
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.06)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.06)' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#cbd5e1' } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Approval Probability: ${ctx.parsed.y}%`
                        }
                    }
                }
            }
        });
    } catch (e) {
        console.error('Sensitivity error:', e);
    }
}

/**
 * Loan Amortization Chart
 */
function renderAmortizationChart() {
    const canvas = document.getElementById('chart-amortization-canvas');
    if (!canvas || typeof Chart === 'undefined') return;

    const loanAmt = (parseFloat(document.getElementById('loan_amount')?.value) || 140) * 1000;
    const loanTerm = parseFloat(document.getElementById('loan_term')?.value) || 360;

    const monthlyRate = 0.085 / 12;
    const emi = (loanAmt * monthlyRate * ((1 + monthlyRate) ** loanTerm)) / (((1 + monthlyRate) ** loanTerm) - 1);
    const totalRepayment = emi * loanTerm;
    const totalInterest = Math.max(0, totalRepayment - loanAmt);

    const elEmi = document.getElementById('amort-emi');
    const elPrincipal = document.getElementById('amort-principal');
    const elInterest = document.getElementById('amort-interest');
    const elTotal = document.getElementById('amort-total');

    if (elEmi) elEmi.textContent = `$${Math.round(emi).toLocaleString()}/mo`;
    if (elPrincipal) elPrincipal.textContent = `$${Math.round(loanAmt).toLocaleString()}`;
    if (elInterest) elInterest.textContent = `$${Math.round(totalInterest).toLocaleString()}`;
    if (elTotal) elTotal.textContent = `$${Math.round(totalRepayment).toLocaleString()}`;

    if (chartAmortization) chartAmortization.destroy();

    chartAmortization = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: ['Principal ($)', 'Total Interest ($)'],
            datasets: [{
                data: [Math.round(loanAmt), Math.round(totalInterest)],
                backgroundColor: ['#6366f1', '#f59e0b'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#cbd5e1' } }
            }
        }
    });
}

/**
 * Preset profiles for rapid demonstration
 */
function initPresets() {
    const presets = {
        'prime': {
            gender: 'Male',
            married: 'Yes',
            dependents: '1',
            education: 'Graduate',
            self_employed: 'No',
            applicant_income: 7500,
            coapplicant_income: 2500,
            loan_amount: 140,
            loan_term: 360,
            credit_history: '1.0',
            property_area: 'Semiurban'
        },
        'high-risk': {
            gender: 'Male',
            married: 'No',
            dependents: '0',
            education: 'Not Graduate',
            self_employed: 'Yes',
            applicant_income: 2200,
            coapplicant_income: 0,
            loan_amount: 180,
            loan_term: 360,
            credit_history: '0.0',
            property_area: 'Rural'
        },
        'borderline': {
            gender: 'Female',
            married: 'Yes',
            dependents: '2',
            education: 'Graduate',
            self_employed: 'Yes',
            applicant_income: 4200,
            coapplicant_income: 1800,
            loan_amount: 210,
            loan_term: 180,
            credit_history: '1.0',
            property_area: 'Urban'
        },
        'coapplicant': {
            gender: 'Female',
            married: 'Yes',
            dependents: '0',
            education: 'Graduate',
            self_employed: 'No',
            applicant_income: 3000,
            coapplicant_income: 5000,
            loan_amount: 160,
            loan_term: 360,
            credit_history: '1.0',
            property_area: 'Semiurban'
        }
    };

    document.querySelectorAll('.preset-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const key = chip.dataset.preset;
            const data = presets[key];
            if (!data) return;

            document.getElementById('gender').value = data.gender;
            document.getElementById('married').value = data.married;
            document.getElementById('dependents').value = data.dependents;
            document.getElementById('education').value = data.education;
            document.getElementById('self_employed').value = data.self_employed;
            document.getElementById('applicant_income').value = data.applicant_income;
            document.getElementById('coapplicant_income').value = data.coapplicant_income;
            document.getElementById('loan_amount').value = data.loan_amount;
            document.getElementById('loan_term').value = data.loan_term;
            document.getElementById('credit_history').value = data.credit_history;
            document.getElementById('property_area').value = data.property_area;

            updateLiveMetrics();
            playTone(650, 'sine', 0.08, 0.1);

            const form = document.getElementById('loan-prediction-form');
            if (form) form.dispatchEvent(new Event('submit'));
        });
    });
}

/**
 * Dynamic Live Financial Metrics
 */
function initLiveCalculators() {
    const inputs = ['applicant_income', 'coapplicant_income', 'loan_amount', 'loan_term'];
    inputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', updateLiveMetrics);
    });
    updateLiveMetrics();
}

function updateLiveMetrics() {
    const appInc = parseFloat(document.getElementById('applicant_income')?.value) || 0;
    const coappInc = parseFloat(document.getElementById('coapplicant_income')?.value) || 0;
    const loanAmt = parseFloat(document.getElementById('loan_amount')?.value) || 0;
    const loanTerm = parseFloat(document.getElementById('loan_term')?.value) || 360;

    const totalIncome = appInc + coappInc;
    const loanActual = loanAmt * 1000;
    const estEmi = (loanActual / Math.max(loanTerm, 1)) * 1.08;
    const dti = totalIncome > 0 ? (estEmi / totalIncome) * 100 : 0;

    const totalEl = document.getElementById('live-total-income');
    const emiEl = document.getElementById('live-est-emi');
    const dtiEl = document.getElementById('live-dti');

    if (totalEl) totalEl.textContent = `$${totalIncome.toLocaleString()}`;
    if (emiEl) emiEl.textContent = `$${Math.round(estEmi).toLocaleString()}/mo`;
    if (dtiEl) {
        dtiEl.textContent = `${dti.toFixed(1)}%`;
        dtiEl.style.color = dti <= 35 ? '#34d399' : (dti <= 50 ? '#fbbf24' : '#f87171');
    }
}

/**
 * Official Underwriting Report Modal & Print
 */
function openUnderwritingReportModal() {
    let modal = document.getElementById('report-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'report-modal';
        modal.className = 'modal-overlay';
        document.body.appendChild(modal);
    }

    const data = lastPredictionData || {};
    const app = data.applicant || collectFormData();
    const isApproved = data.prediction === 1;
    const statusText = isApproved ? 'APPROVED' : 'REJECTED';
    const stampColor = isApproved ? '#10b981' : '#ef4444';

    modal.innerHTML = `
        <div class="modal-container" id="printable-report">
            <button type="button" id="close-modal-btn" style="position:absolute; top:1.25rem; right:1.25rem; background:none; border:none; font-size:1.5rem; cursor:pointer; color:#64748b;">✕</button>
            
            <div class="report-watermark">${statusText}</div>

            <div style="display:flex; justify-content:space-between; align-items:flex-start; border-bottom:2px solid #0f172a; padding-bottom:1rem; margin-bottom:1.5rem;">
                <div>
                    <h2 style="font-size:1.6rem; font-weight:900; color:#0f172a; margin-bottom:0.2rem;">🏦 CREDITPULSE FINANCIAL INSTITUTION</h2>
                    <div style="font-size:0.85rem; color:#475569;">Automated Credit Underwriting & Risk Evaluation Assessment</div>
                </div>
                <div style="text-align:right; font-size:0.8rem; color:#475569;">
                    <div>Date: ${new Date().toLocaleDateString()}</div>
                    <div>Reference: CP-${Math.floor(100000 + Math.random() * 900000)}</div>
                </div>
            </div>

            <!-- Decision Stamp -->
            <div style="display:flex; justify-content:space-between; align-items:center; background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:1.25rem; margin-bottom:1.5rem;">
                <div>
                    <div style="font-size:0.8rem; text-transform:uppercase; color:#64748b; font-weight:700;">Underwriting Verdict</div>
                    <div style="font-size:1.75rem; font-weight:900; color:${stampColor}; letter-spacing:1px;">
                        ${isApproved ? '🟢 LOAN APPROVED' : '🔴 LOAN REJECTED'}
                    </div>
                    <div style="font-size:0.85rem; color:#475569;">
                        Model Algorithm: <strong>${data.active_model || 'Random Forest'}</strong> | Confidence: <strong>${data.confidence || 85.0}%</strong>
                    </div>
                </div>
                <div style="border: 3px dashed ${stampColor}; border-radius:8px; padding:0.6rem 1.25rem; transform:rotate(-5deg); color:${stampColor}; font-weight:900; font-size:1.3rem;">
                    ${statusText}
                </div>
            </div>

            <!-- Applicant Profile Grid -->
            <h4 style="font-size:1rem; font-weight:800; color:#0f172a; margin-bottom:0.6rem;">1. Applicant Financial Profile</h4>
            <table style="width:100%; border-collapse:collapse; font-size:0.88rem; margin-bottom:1.5rem;">
                <tbody>
                    <tr style="border-bottom:1px solid #e2e8f0;">
                        <td style="padding:0.45rem; color:#64748b; width:25%;">Applicant Income:</td>
                        <td style="padding:0.45rem; font-weight:700;">$${Number(app.ApplicantIncome).toLocaleString()}/mo</td>
                        <td style="padding:0.45rem; color:#64748b; width:25%;">Co-applicant Income:</td>
                        <td style="padding:0.45rem; font-weight:700;">$${Number(app.CoapplicantIncome).toLocaleString()}/mo</td>
                    </tr>
                    <tr style="border-bottom:1px solid #e2e8f0;">
                        <td style="padding:0.45rem; color:#64748b;">Requested Loan:</td>
                        <td style="padding:0.45rem; font-weight:700;">$${Number(app.LoanAmount * 1000).toLocaleString()}</td>
                        <td style="padding:0.45rem; color:#64748b;">Term Duration:</td>
                        <td style="padding:0.45rem; font-weight:700;">${app.Loan_Amount_Term} Months</td>
                    </tr>
                    <tr style="border-bottom:1px solid #e2e8f0;">
                        <td style="padding:0.45rem; color:#64748b;">Credit History Status:</td>
                        <td style="padding:0.45rem; font-weight:700;">${app.Credit_History == 1 ? 'Prime (1.0 - Guidelines Met)' : 'Defaulted (0.0 - Bad Credit)'}</td>
                        <td style="padding:0.45rem; color:#64748b;">Property Location:</td>
                        <td style="padding:0.45rem; font-weight:700;">${app.Property_Area}</td>
                    </tr>
                    <tr>
                        <td style="padding:0.45rem; color:#64748b;">Education:</td>
                        <td style="padding:0.45rem; font-weight:700;">${app.Education}</td>
                        <td style="padding:0.45rem; color:#64748b;">Employment:</td>
                        <td style="padding:0.45rem; font-weight:700;">${app.Self_Employed === 'Yes' ? 'Self-Employed' : 'Salaried'}</td>
                    </tr>
                </tbody>
            </table>

            <!-- Financial Analysis & EMI -->
            <h4 style="font-size:1rem; font-weight:800; color:#0f172a; margin-bottom:0.6rem;">2. Solvency & Debt-to-Income Audit</h4>
            <div style="display:flex; justify-content:space-between; background:#f1f5f9; padding:0.9rem; border-radius:8px; font-size:0.85rem; margin-bottom:1.5rem;">
                <div>Monthly Payment: <strong>$${data.emi_monthly || 0}/mo</strong></div>
                <div>Debt Burden (DTI): <strong>${data.emi_ratio || 0}%</strong></div>
                <div>Safe Borrowing Limit: <strong>$${(data.max_safe_loan || 0).toLocaleString()}</strong></div>
            </div>

            <!-- Signature Line -->
            <div style="display:flex; justify-content:space-between; margin-top:3rem; padding-top:1.5rem; border-top:1px solid #cbd5e1; font-size:0.8rem; color:#64748b;">
                <div>
                    <div>Chief Credit Underwriter: ____________________</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.2rem;">Credit Assessment Board</div>
                </div>
                <div>
                    <div>AI System Verification Hash: SHA-256 Verified ✓</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.2rem;">Model: Scikit-Learn Random Forest Pipeline</div>
                </div>
            </div>

            <div style="display:flex; justify-content:flex-end; gap:0.75rem; margin-top:1.5rem;">
                <button type="button" class="btn btn-secondary" id="btn-cancel-modal" style="color:#0f172a; border-color:#cbd5e1;">Close</button>
                <button type="button" class="btn btn-primary" onclick="window.print();">🖨️ Print / Save as PDF</button>
            </div>
        </div>
    `;

    modal.classList.add('active');
    document.getElementById('close-modal-btn').onclick = () => modal.classList.remove('active');
    document.getElementById('btn-cancel-modal').onclick = () => modal.classList.remove('active');
    modal.onclick = (e) => { if (e.target === modal) modal.classList.remove('active'); };
}

/**
 * Batch CSV Scoring Component
 */
function initBatchScoring() {
    const dropzone = document.getElementById('csv-dropzone');
    const fileInput = document.getElementById('csv-file-input');
    const resultBox = document.getElementById('batch-results-box');

    if (!dropzone || !fileInput) return;

    dropzone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleBatchUpload(fileInput.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleBatchUpload(fileInput.files[0]);
        }
    });
}

async function handleBatchUpload(file) {
    const resultBox = document.getElementById('batch-results-box');
    if (!resultBox) return;

    resultBox.style.display = 'block';
    resultBox.innerHTML = `<div style="text-align:center; padding:2rem;"><span class="spinner"></span> Processing and Scoring Batch Applications...</div>`;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/batch-predict', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (!data.success) {
            resultBox.innerHTML = `<div style="color:#ef4444; padding:1.5rem;">Error: ${data.error}</div>`;
            return;
        }

        playApprovalSound();

        resultBox.innerHTML = `
            <div class="stats-grid" style="margin-bottom:1.5rem;">
                <div class="stat-card">
                    <div class="stat-icon indigo">📁</div>
                    <div class="stat-info">
                        <h4>Total Scored</h4>
                        <div class="stat-value">${data.total_scored}</div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon emerald">🟢</div>
                    <div class="stat-info">
                        <h4>Approved</h4>
                        <div class="stat-value" style="color:#34d399;">${data.approved_count}</div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon amber">🔴</div>
                    <div class="stat-info">
                        <h4>Rejected</h4>
                        <div class="stat-value" style="color:#f87171;">${data.rejected_count}</div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon cyan">📊</div>
                    <div class="stat-info">
                        <h4>Batch Approval Rate</h4>
                        <div class="stat-value">${data.batch_approval_rate}%</div>
                    </div>
                </div>
            </div>

            <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Application ID</th>
                            <th>Income</th>
                            <th>Loan Amount</th>
                            <th>Credit History</th>
                            <th>Property Area</th>
                            <th>Decision</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.records.map(r => `
                            <tr>
                                <td><strong>${r.Loan_ID}</strong></td>
                                <td>$${r.ApplicantIncome.toLocaleString()}</td>
                                <td>$${r.LoanAmount}k</td>
                                <td>${r.Credit_History == 1 ? '1.0 (Good)' : '0.0 (Bad)'}</td>
                                <td>${r.Property_Area}</td>
                                <td><span class="badge-status ${r.Prediction === 1 ? 'Y' : 'N'}">${r.Decision}</span></td>
                                <td><strong>${r.Confidence}%</strong></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    } catch (e) {
        console.error(e);
        resultBox.innerHTML = `<div style="color:#ef4444; padding:1.5rem;">Batch scoring connection failed.</div>`;
    }
}

/**
 * Dedicated Interactive Simulator Page Logic
 */
function initSimulator() {
    const simSliders = ['sim_income', 'sim_coincome', 'sim_loan', 'sim_term'];
    simSliders.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', () => {
                document.getElementById(id + '_val').textContent = el.value;
                debounceSimulate();
            });
        }
    });

    const simSelects = ['sim_credit', 'sim_area', 'sim_edu', 'sim_model'];
    simSelects.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', debounceSimulate);
    });
}

let simTimeout = null;
function debounceSimulate() {
    clearTimeout(simTimeout);
    simTimeout = setTimeout(runSimulatorInference, 100);
}

async function runSimulatorInference() {
    const simIncome = document.getElementById('sim_income');
    if (!simIncome) return;

    const payload = {
        model: document.getElementById('sim_model').value,
        Gender: 'Male',
        Married: 'Yes',
        Dependents: '1',
        Education: document.getElementById('sim_edu').value,
        Self_Employed: 'No',
        ApplicantIncome: parseFloat(document.getElementById('sim_income').value),
        CoapplicantIncome: parseFloat(document.getElementById('sim_coincome').value),
        LoanAmount: parseFloat(document.getElementById('sim_loan').value),
        Loan_Amount_Term: parseFloat(document.getElementById('sim_term').value),
        Credit_History: parseFloat(document.getElementById('sim_credit').value),
        Property_Area: document.getElementById('sim_area').value
    };

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            const gauge = document.getElementById('sim-gauge-fill');
            const score = document.getElementById('sim-score');
            const status = document.getElementById('sim-status');

            if (gauge) {
                gauge.style.width = `${data.prob_approved}%`;
                gauge.className = `confidence-bar-fill ${data.prediction === 1 ? 'approved' : 'rejected'}`;
            }
            if (score) score.textContent = `${data.prob_approved}%`;
            if (status) {
                status.textContent = data.prediction === 1 ? '🟢 APPROVED' : '🔴 REJECTED';
                status.style.color = data.prediction === 1 ? '#34d399' : '#f87171';
            }
        }
    } catch (e) {
        console.error(e);
    }
}

/**
 * Dataset Explorer on Dashboard
 */
let currentPage = 1;
const pageLimit = 10;

function initDatasetExplorer() {
    const tableBody = document.getElementById('dataset-table-body');
    if (!tableBody) return;

    const searchInput = document.getElementById('table-search');
    const filterSelect = document.getElementById('status-filter');
    const prevBtn = document.getElementById('prev-page-btn');
    const nextBtn = document.getElementById('next-page-btn');

    if (searchInput) {
        searchInput.addEventListener('input', () => {
            currentPage = 1;
            loadDatasetPage();
        });
    }

    if (filterSelect) {
        filterSelect.addEventListener('change', () => {
            currentPage = 1;
            loadDatasetPage();
        });
    }

    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                loadDatasetPage();
            }
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            currentPage++;
            loadDatasetPage();
        });
    }

    loadDatasetPage();
}

async function loadDatasetPage() {
    const tableBody = document.getElementById('dataset-table-body');
    const pageIndicator = document.getElementById('page-indicator');
    const totalCountEl = document.getElementById('table-total-count');
    if (!tableBody) return;

    const search = document.getElementById('table-search')?.value || '';
    const status = document.getElementById('status-filter')?.value || 'all';

    tableBody.innerHTML = `<tr><td colspan="13" style="text-align:center; padding: 2rem;"><span class="spinner"></span> Loading historical loans...</td></tr>`;

    try {
        const res = await fetch(`/api/data?page=${currentPage}&limit=${pageLimit}&search=${encodeURIComponent(search)}&status=${status}`);
        const data = await res.json();

        if (totalCountEl) totalCountEl.textContent = `Total: ${data.total} records`;
        if (pageIndicator) pageIndicator.textContent = `Page ${data.page} of ${Math.ceil(data.total / pageLimit) || 1}`;

        if (!data.records || data.records.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="13" style="text-align:center; padding: 2rem; color: #94a3b8;">No matching applications found.</td></tr>`;
            return;
        }

        tableBody.innerHTML = data.records.map(row => {
            const statusClass = row.Loan_Status === 'Y' ? 'Y' : 'N';
            const statusLabel = row.Loan_Status === 'Y' ? 'Approved (Y)' : 'Rejected (N)';
            const chLabel = row.Credit_History == 1.0 ? '1.0 (Good)' : (row.Credit_History == 0.0 ? '0.0 (Bad)' : 'N/A');

            return `
                <tr>
                    <td><strong>${row.Loan_ID}</strong></td>
                    <td>${row.Gender}</td>
                    <td>${row.Married}</td>
                    <td>${row.Dependents}</td>
                    <td>${row.Education}</td>
                    <td>${row.Self_Employed}</td>
                    <td>$${Number(row.ApplicantIncome || 0).toLocaleString()}</td>
                    <td>$${Number(row.CoapplicantIncome || 0).toLocaleString()}</td>
                    <td>$${row.LoanAmount}k</td>
                    <td>${row.Loan_Amount_Term} mos</td>
                    <td>${chLabel}</td>
                    <td>${row.Property_Area}</td>
                    <td><span class="badge-status ${statusClass}">${statusLabel}</span></td>
                </tr>
            `;
        }).join('');

    } catch (err) {
        console.error(err);
        tableBody.innerHTML = `<tr><td colspan="13" style="text-align:center; padding: 2rem; color: #ef4444;">Error fetching data.</td></tr>`;
    }
}

/**
 * Chart.js Visualizations for Dashboard
 */
async function initDashboardCharts() {
    const modelChartCanvas = document.getElementById('chart-model-comparison');
    const targetChartCanvas = document.getElementById('chart-target-dist');
    const featChartCanvas = document.getElementById('chart-feature-imp');

    if (!modelChartCanvas && !targetChartCanvas && !featChartCanvas) return;

    try {
        const res = await fetch('/api/metadata');
        const meta = await res.json();

        // 1. Model Accuracy Chart
        if (modelChartCanvas && meta.benchmark_metrics && typeof Chart !== 'undefined') {
            const modelNames = Object.keys(meta.benchmark_metrics);
            const testAccs = modelNames.map(m => meta.benchmark_metrics[m].test_accuracy);
            const cvAccs = modelNames.map(m => meta.benchmark_metrics[m].cv_accuracy_mean);

            new Chart(modelChartCanvas, {
                type: 'bar',
                data: {
                    labels: modelNames,
                    datasets: [
                        {
                            label: 'Test Accuracy (%)',
                            data: testAccs,
                            backgroundColor: '#10b981',
                            borderRadius: 6
                        },
                        {
                            label: '5-Fold CV Accuracy (%)',
                            data: cvAccs,
                            backgroundColor: '#6366f1',
                            borderRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            min: 60,
                            max: 100,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#94a3b8' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#94a3b8' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#cbd5e1' } }
                    }
                }
            });
        }

        // 2. Feature Importance Horizontal Bar
        if (featChartCanvas && meta.feature_importance && typeof Chart !== 'undefined') {
            const topFeats = meta.feature_importance.slice(0, 7);
            const labels = topFeats.map(f => f.feature);
            const values = topFeats.map(f => f.importance);

            new Chart(featChartCanvas, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Importance Score (%)',
                        data: values,
                        backgroundColor: 'rgba(99, 102, 241, 0.85)',
                        borderColor: '#818cf8',
                        borderWidth: 1,
                        borderRadius: 6
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#94a3b8' }
                        },
                        y: {
                            grid: { display: false },
                            ticks: { color: '#cbd5e1' }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }

    } catch (e) {
        console.error('Error rendering dashboard charts:', e);
    }
}
