// VTHPhishDet - Main JS
const $ = id => document.getElementById(id);
const $$ = s => document.querySelectorAll(s);

let currentUrl = '';
let imageFile = null;
let qrFile = null;
let emailImageFile = null;

// ── Initialization ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Health check polling
  const checkHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'ok' || data.status === 'running') {
          const loader = $('initial-loader');
          if (loader) {
            loader.style.opacity = '0';
            setTimeout(() => loader.remove(), 500);
          }
          return; // Done
        }
      }
    } catch (e) {}
    setTimeout(checkHealth, 2000);
  };
  checkHealth();
  // Tabs
  $$('.nav-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const { tab } = btn.dataset;
      $$('.nav-tab').forEach(b => b.classList.remove('active'));
      $$('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      $(`tab-${tab}`).classList.add('active');
      hideResults();
    });
  });

  // Scan buttons
  $('btn-scan-url').addEventListener('click', scanUrl);
  $('btn-scan-email').addEventListener('click', scanEmail);
  $('btn-scan-email-image').addEventListener('click', scanEmailImage);
  $('btn-scan-image').addEventListener('click', scanImage);
  $('btn-scan-qr').addEventListener('click', scanQr);

  // File uploads
  setupUpload('email-image-dropzone', 'email-image-file', 'email-image-preview-wrap', 'email-image-preview-img', f => { emailImageFile = f; });
  setupUpload('image-dropzone', 'image-file', 'image-preview-wrap', 'image-preview-img', f => { imageFile = f; });
  setupUpload('qr-dropzone', 'qr-file', 'qr-preview-wrap', 'qr-preview-img', f => { qrFile = f; });
  $('email-image-remove').addEventListener('click', () => { emailImageFile = null; resetUpload('email-image-dropzone', 'email-image-preview-wrap', 'email-image-file'); });
  $('image-remove').addEventListener('click', () => { imageFile = null; resetUpload('image-dropzone', 'image-preview-wrap', 'image-file'); });
  $('qr-remove').addEventListener('click', () => { qrFile = null; resetUpload('qr-dropzone', 'qr-preview-wrap', 'qr-file'); });

  // Enter key
  $('url-input').addEventListener('keydown', e => { if (e.key === 'Enter') scanUrl(); });
});

// ── Upload helpers ────────────────────────────────────────────────────────────
function setupUpload(zoneId, inputId, previewWrapId, previewImgId, onFile) {
  const zone = $(zoneId), input = $(inputId);
  zone.addEventListener('click', () => input.click());
  input.addEventListener('click', e => e.stopPropagation());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = 'var(--primary)'; });
  zone.addEventListener('dragleave', () => { zone.style.borderColor = 'var(--border)'; });
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.style.borderColor = 'var(--border)';
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f, zone, previewWrapId, previewImgId, onFile);
  });
  input.addEventListener('change', () => {
    if (input.files[0]) handleFile(input.files[0], zone, previewWrapId, previewImgId, onFile);
  });
}

function handleFile(file, zone, previewWrapId, previewImgId, onFile) {
  onFile(file);
  const reader = new FileReader();
  reader.onload = e => {
    $(previewImgId).src = e.target.result;
    zone.classList.add('hidden');
    $(previewWrapId).classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

function resetUpload(zoneId, previewWrapId, inputId) {
  $(zoneId).classList.remove('hidden');
  $(previewWrapId).classList.add('hidden');
  $(inputId).value = '';
}

// ── Scan functions ────────────────────────────────────────────────────────────
async function scanUrl() {
  const url = $('url-input').value.trim();
  if (!url) { shake($('url-input')); return; }
  currentUrl = url;

  setLoading(true, 'Analyzing URL with XGBoost and WHOIS...');
  try {
    const data = await postJSON('/api/scan/url', { url });
    renderResults(data, 'url');
  } catch (e) { showError(e.message); } finally { setLoading(false); }
}

async function scanEmail() {
  const text = $('email-input').value.trim();
  const model = document.querySelector('input[name="email-model"]:checked').value;
  if (!text) { shake($('email-input')); return; }
  setLoading(true, `Processing email through ${model === 'llama' ? 'Llama' : 'RoBERTa'}...`);
  try {
    const data = await postJSON('/api/scan/email', { text, model });
    renderResults(data, 'email');
  } catch (e) { showError(e.message); } finally { setLoading(false); }
}

async function scanEmailImage() {
  if (!emailImageFile) { shake($('email-image-dropzone')); return; }
  const model = document.querySelector('input[name="email-model"]:checked').value;
  setLoading(true, `Extracting text via OCR and scoring with ${model === 'llama' ? 'Llama' : 'RoBERTa'}...`);
  try {
    const fd = new FormData();
    fd.append('file', emailImageFile);
    fd.append('model', model);
    const resp = await fetch('/api/scan/email_image', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    renderResults(data, 'email_image');
  } catch (e) { showError(e.message); } finally { setLoading(false); }
}

async function scanImage() {
  if (!imageFile) { shake($('image-dropzone')); return; }
  setLoading(true, 'Sending image to the local vision model...');
  try {
    const fd = new FormData();
    fd.append('file', imageFile);
    const resp = await fetch('/api/scan/image', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    renderResults(data, 'image');
  } catch (e) { showError(e.message); } finally { setLoading(false); }
}

async function scanQr() {
  if (!qrFile) { shake($('qr-dropzone')); return; }
  setLoading(true, 'Decoding QR and scanning URL...');
  try {
    const fd = new FormData();
    fd.append('file', qrFile);
    const resp = await fetch('/api/scan/qr', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    currentUrl = data.qr_url || '';
    renderResults(data, 'qr');
  } catch (e) { showError(e.message); } finally { setLoading(false); }
}

// ── Render Results ────────────────────────────────────────────────────────────
function renderResults(data, type) {
  $('placeholder').classList.add('hidden');
  $('results-section').classList.remove('hidden');

  let score = null, verdict = 'unknown', label = '—', sub = '';
  let scores = [], whois = null, screenshotB64 = null, finalUrl = null, hasRedirect = false, visionText = null;

  if (type === 'url') {
    score = data.url_score;
    verdict = data.label || classify(score);
    label = verdict;
    whois = data.whois;
    scores = [
      { label: 'URL Score', score: data.url_score },
      { label: 'WHOIS Score', score: data.whois?.whois_score }
    ];
  } else if (type === 'email') {
    score = data.email_score;
    verdict = data.label || classify(score);
    label = verdict;
    scores = [
      { label: 'Phishing Score', score: data.email_score, clsOverride: getScoreCls(data.email_score), forceRed: true },
      { label: 'Confidence', score: data.confidence, clsOverride: 'safe', verdictOverride: 'MODEL CERTAINTY' }
    ];
  } else if (type === 'email_image') {
    score = data.email_score;
    verdict = data.label || classify(score);
    label = verdict;
    scores = [
      { label: 'Phishing Score', score: data.email_score, clsOverride: getScoreCls(data.email_score), forceRed: true },
      { label: 'Confidence', score: data.confidence, clsOverride: 'safe', verdictOverride: 'MODEL CERTAINTY' }
    ];
  } else if (type === 'image') {
    score = data.image_score;
    verdict = classify(score);
    label = verdict;
    screenshotB64 = data.screenshot_base64;
    visionText = data.vision_response?.rationale || data.vision_response?.raw_output || null;
    scores = [{ label: 'Image Score', score: data.image_score }];
  } else if (type === 'qr') {
    score = data.url_score;
    verdict = data.label || classify(score);
    label = verdict;
    whois = data.whois;
    sub = data.qr_url ? `Decoded: ${data.qr_url}` : '';
    scores = [
      { label: 'URL Score', score: data.url_score },
      { label: 'WHOIS Score', score: data.whois?.whois_score }
    ];
  }

  const cls = normVerdict(verdict);

  // Verdict card
  const card = $('verdict-card');
  card.className = `verdict-card ${cls}`;
  const icons = { safe: '✓', suspicious: '?', phishing: '⚠', unknown: '·' };
  const iconWrap = $('verdict-icon-wrap');
  iconWrap.className = `v-icon ${cls}`;
  iconWrap.textContent = icons[cls] || '?';
  const vmain = $('verdict-main');
  vmain.className = `v-main ${cls}`;
  vmain.textContent = label.toUpperCase();
  $('verdict-sub').textContent = sub;

  // Score cards
  renderScoreCards(scores.filter(s => s.score != null));

  // Details
  renderDetails(whois, data, type);

  // Screenshot
  if (screenshotB64) {
    $('results-section').classList.remove('no-sc');
    $('screenshot-card').classList.remove('hidden');
    $('sc-img').src = `data:image/png;base64,${screenshotB64}`;
    if (hasRedirect && finalUrl) {
      $('redirect-badge').classList.remove('hidden');
      $('redirect-badge').title = `Redirected to: ${finalUrl}`;
    } else {
      $('redirect-badge').classList.add('hidden');
    }
    if (visionText && visionText !== 'null') {
      $('vision-box').classList.remove('hidden');
      $('vision-text').textContent = visionText;
    } else {
      $('vision-box').classList.add('hidden');
    }
  } else {
    $('results-section').classList.add('no-sc');
    $('screenshot-card').classList.add('hidden');
  }
}

// ── Score Cards ───────────────────────────────────────────────────────────────
function renderScoreCards(scores) {
  const grid = $('scores-grid');
  grid.innerHTML = '';
  scores.forEach(({ label, score, clsOverride, verdictOverride, forceRed }) => {
    if (score == null) return;
    const pct = (score * 100).toFixed(1);
    const autoC = getScoreCls(score);
    const barCls = forceRed ? 'phishing' : (clsOverride || autoC);
    const valCls = forceRed ? 'phishing' : (clsOverride || autoC);
    const lblCls = clsOverride || autoC;

    const div = document.createElement('div');
    div.className = 'score-card';
    div.innerHTML = `
      <div class="sc-lbl">${label}</div>
      <div class="sc-val ${valCls}">${pct}%</div>
      <div class="sc-bar-bg"><div class="sc-bar ${barCls}" data-w="${pct}"></div></div>`;
    grid.appendChild(div);
    setTimeout(() => { div.querySelector('.sc-bar').style.width = pct + '%'; }, 60);
  });
}

function renderDetails(whois, data, type) {
  const grid = $('details-grid');
  grid.innerHTML = '';
  if (whois) {
    const rows = [
      { k: 'Domain', v: whois.domain || '—' },
      { k: 'Registrar', v: whois.registrar || '—' },
      { k: 'Created', v: whois.creation_date ? whois.creation_date.split('T')[0] : '—' },
      { k: 'Age (days)', v: whois.age_days != null ? whois.age_days : '—', cls: whois.is_young_domain ? 'bad' : 'good' },
      { k: 'Young Domain', v: whois.is_young_domain ? 'Yes' : 'No', cls: whois.is_young_domain ? 'bad' : 'good' },
    ];
    if (whois.error) rows.push({ k: 'WHOIS Error', v: whois.error, cls: 'warn' });
    grid.appendChild(makeDetailCard('WHOIS Intel', rows));
  }
  if (data.features_used) {
    const f = data.features_used;
    grid.appendChild(makeDetailCard('URL Features', [
      { k: 'Length', v: f.url_length },
      { k: 'HTTPS', v: f.has_https ? 'Yes' : 'No', cls: f.has_https ? 'good' : 'bad' },
      { k: 'IP Address', v: f.has_ip ? 'Yes' : 'No', cls: f.has_ip ? 'bad' : 'good' },
      { k: 'Subdomains', v: f.num_subdomains },
      { k: 'Entropy', v: f.url_entropy?.toFixed(3) },
      { k: 'Suspicious Words', v: f.has_suspicious_words ? 'Yes' : 'No', cls: f.has_suspicious_words ? 'bad' : 'good' },
    ]));
  }
  if (type === 'url+ss' && data.url_result) {
    const f = data.url_result.features_used;
    if (f) {
      grid.appendChild(makeDetailCard('URL Features', [
        { k: 'Length', v: f.url_length },
        { k: 'HTTPS', v: f.has_https ? 'Yes' : 'No', cls: f.has_https ? 'good' : 'bad' },
        { k: 'IP Address', v: f.has_ip ? 'Yes' : 'No', cls: f.has_ip ? 'bad' : 'good' },
        { k: 'Subdomains', v: f.num_subdomains },
        { k: 'Entropy', v: f.url_entropy?.toFixed(3) },
        { k: 'Suspicious Words', v: f.has_suspicious_words ? 'Yes' : 'No', cls: f.has_suspicious_words ? 'bad' : 'good' },
      ]));
    }
  }
  if (data.qr_url) {
    grid.appendChild(makeDetailCard('QR Code', [{ k: 'URL', v: data.qr_url }]));
  }
  if (type === 'email_image' && data.extracted_text) {
    grid.appendChild(makeDetailCard('OCR Text', [{ k: 'Text', v: data.extracted_text, no_trunc: true }]));
  }
}

function makeDetailCard(title, rows) {
  const card = document.createElement('div');
  card.className = 'detail-card';
  card.innerHTML = `<div class="dc-title">${title}</div>` +
    rows.map(r => `<div class="dc-row"><span class="dc-key">${r.k}</span><span class="dc-val ${r.cls || ''}">${r.no_trunc ? String(r.v || '—') : trunc(String(r.v || '—'), 30)}</span></div>`).join('');
  return card;
}

// ── UI Helpers ────────────────────────────────────────────────────────────────
function setLoading(on, msg) {
  const loadEl = $('loading');
  const phEl = $('placeholder');
  const resEl = $('results-section');

  if (on) {
    $('loading-label').textContent = msg || 'Analyzing...';
    phEl.classList.add('hidden');
    resEl.classList.add('hidden');
    loadEl.classList.remove('hidden');
    $$('.btn').forEach(b => b.disabled = true);
  } else {
    loadEl.classList.add('hidden');
    $$('.btn').forEach(b => b.disabled = false);
  }
}

function hideResults() {
  $('results-section').classList.add('hidden');
  $('loading').classList.add('hidden');
  $('placeholder').classList.remove('hidden');
}

function showError(msg) {
  $('placeholder').classList.add('hidden');
  $('results-section').classList.remove('hidden');

  const card = $('verdict-card');
  card.className = 'verdict-card phishing';
  $('verdict-icon-wrap').className = 'v-icon phishing';
  $('verdict-icon-wrap').textContent = '✗';
  $('verdict-main').className = 'v-main phishing';
  $('verdict-main').textContent = 'ERROR';
  $('verdict-sub').textContent = msg;

  $('scores-grid').innerHTML = '';
  $('details-grid').innerHTML = '';
  $('screenshot-card').classList.add('hidden');
  $('vision-box').classList.add('hidden');
  $('results-section').classList.add('no-sc');
}

function shake(el) {
  el.style.animation = 'none';
  el.offsetHeight;
  el.style.animation = 'shake 0.4s ease';
  setTimeout(() => el.style.animation = '', 500);
}

// ── Utility ───────────────────────────────────────────────────────────────────
function classify(score) {
  if (score == null) return 'unknown';
  if (score >= 0.75) return 'phishing';
  if (score >= 0.5) return 'suspicious';
  return 'safe';
}
function normVerdict(v) {
  if (!v) return 'unknown';
  v = v.toLowerCase();
  if (v === 'phishing') return 'phishing';
  if (v === 'suspicious') return 'suspicious';
  if (v === 'safe') return 'safe';
  return 'unknown';
}
function getScoreCls(s) {
  if (s == null) return 'na';
  if (s >= 0.75) return 'phishing';
  if (s >= 0.5) return 'suspicious';
  return 'safe';
}
function trunc(s, n) { return s && s.length > n ? s.slice(0, n) + '…' : (s || '—'); }

async function postJSON(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const d = await r.json();
  if (d.error) throw new Error(d.error);
  return d;
}

// Shake animation
const st = document.createElement('style');
st.textContent = '@keyframes shake{0%,100%{transform:translateX(0)}25%{transform:translateX(-8px)}75%{transform:translateX(8px)}}';
document.head.appendChild(st);