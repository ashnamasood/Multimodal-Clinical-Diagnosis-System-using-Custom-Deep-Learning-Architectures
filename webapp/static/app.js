/* FusionNet Clinical Dashboard — frontend logic (API unchanged) */
const heartFieldsContainer = document.getElementById('heart-fields');
const resultEl = document.getElementById('result');
const modelStatusEl = document.getElementById('model-status');
const resultsContent = document.getElementById('results-content');
const resultsPanel = document.getElementById('results-panel');
const emptyState = document.getElementById('empty-state');
const alertRegion = document.getElementById('alert-region');
const xrayOutput = document.getElementById('xray-output');
const skinOutput = document.getElementById('skin-output');
const heartOutput = document.getElementById('heart-output');
const fusionSummary = document.getElementById('fusion-summary');
const riskLevelEl = document.getElementById('risk-level');
const riskBarFill = document.getElementById('risk-bar-fill');
const riskBar = document.getElementById('risk-bar');
const fusionRecommendationEl = document.getElementById('fusion-recommendation');
const fusionNotesEl = document.getElementById('fusion-notes');
const pageTitleEl = document.getElementById('page-title');
const ovXray = document.getElementById('ov-xray');
const ovSkin = document.getElementById('ov-skin');
const ovHeart = document.getElementById('ov-heart');
const ovFusion = document.getElementById('ov-fusion');

const PAGE_TITLES = {
  home: 'Clinical overview',
  predict: 'Run diagnosis',
  'how-it-works': 'How it works',
  faq: 'FAQ',
};

const heartDefaults = {
  Age: 45,
  Sex: 'M',
  ChestPainType: 'ATA',
  RestingBP: 120,
  Cholesterol: 220,
  FastingBS: 0,
  RestingECG: 'Normal',
  MaxHR: 150,
  ExerciseAngina: 'N',
  Oldpeak: 1.0,
  ST_Slope: 'Up',
};

const heartLabels = {
  Age: 'Age (years)',
  RestingBP: 'Resting BP (mmHg)',
  Cholesterol: 'Cholesterol (mg/dL)',
  FastingBS: 'Fasting blood sugar (0/1)',
  MaxHR: 'Max heart rate',
  Oldpeak: 'ST depression (Oldpeak)',
  Sex: 'Sex',
  ChestPainType: 'Chest pain type',
  RestingECG: 'Resting ECG',
  ExerciseAngina: 'Exercise angina',
  ST_Slope: 'ST slope',
};

const heartSchema = {
  Sex: ['M', 'F'],
  ChestPainType: ['TA', 'ATA', 'NAP', 'ASY'],
  RestingECG: ['Normal', 'ST', 'LVH'],
  ExerciseAngina: ['Y', 'N'],
  ST_Slope: ['Up', 'Flat', 'Down'],
  numeric: ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak'],
};

function icon(name) {
  return `<svg class="icon" aria-hidden="true"><use href="/static/icons.svg#icon-${name}"></use></svg>`;
}

function createHeartFields() {
  heartFieldsContainer.innerHTML = '';
  heartSchema.numeric.forEach((field) => {
    const wrap = document.createElement('div');
    wrap.className = 'field';
    wrap.innerHTML = `<label for="heart_${field}">${heartLabels[field] || field}</label><input type="number" step="any" id="heart_${field}" value="${heartDefaults[field]}" />`;
    heartFieldsContainer.appendChild(wrap);
  });

  Object.keys(heartSchema).filter((k) => k !== 'numeric').forEach((field) => {
    const select = document.createElement('select');
    select.id = `heart_${field}`;
    heartSchema[field].forEach((o) => {
      const opt = document.createElement('option');
      opt.value = o;
      opt.textContent = o;
      if (heartDefaults[field] === o) opt.selected = true;
      select.appendChild(opt);
    });
    const wrap = document.createElement('div');
    wrap.className = 'field';
    const label = document.createElement('label');
    label.setAttribute('for', `heart_${field}`);
    label.textContent = heartLabels[field] || field;
    wrap.appendChild(label);
    wrap.appendChild(select);
    heartFieldsContainer.appendChild(wrap);
  });
}

function collectHeartPayload() {
  const payload = {};
  heartSchema.numeric.forEach((f) => {
    const raw = document.getElementById(`heart_${f}`).value;
    payload[f] = f === 'Oldpeak' ? parseFloat(raw) : parseInt(raw, 10);
  });
  Object.keys(heartSchema).filter((k) => k !== 'numeric').forEach((f) => {
    payload[f] = document.getElementById(`heart_${f}`).value;
  });
  return payload;
}

function statusBadge(label, on) {
  return `<span class="badge ${on ? 'on' : 'off'}">${icon(on ? 'check' : 'alert')}${label}</span>`;
}

function updateOverview(models) {
  if (!models) return;
  ovXray.textContent = models.xray ? 'Online' : 'Offline';
  ovSkin.textContent = models.skin ? 'Online' : 'Offline';
  ovHeart.textContent = models.heart ? 'Online' : 'Offline';
}

async function loadHealth() {
  try {
    const r = await fetch('/api/health');
    const data = await r.json();
    modelStatusEl.innerHTML = [
      statusBadge('X-ray', data.models.xray),
      statusBadge('Skin', data.models.skin),
      statusBadge('Heart', data.models.heart),
    ].join('');
    updateOverview(data.models);
  } catch {
    modelStatusEl.textContent = 'Unable to reach API';
    ovXray.textContent = ovSkin.textContent = ovHeart.textContent = 'Unknown';
  }
}

function clearAlert() {
  alertRegion.innerHTML = '';
}

function showError(message) {
  clearAlert();
  alertRegion.innerHTML = `<div class="alert-banner" role="alert">${icon('alert')}<span>${message}</span></div>`;
}

function showEmptyResults() {
  emptyState.hidden = false;
  resultsPanel.hidden = true;
}

function showResultsPanel() {
  emptyState.hidden = true;
  resultsPanel.hidden = false;
}

function showLoadingState() {
  showResultsPanel();
  xrayOutput.innerHTML = '';
  skinOutput.innerHTML = '';
  heartOutput.innerHTML = '';
  fusionRecommendationEl.textContent = '';
  fusionNotesEl.innerHTML = '';
  riskLevelEl.textContent = 'Analyzing…';
  riskBarFill.style.width = '0%';
  riskBar.setAttribute('aria-valuenow', '0');

  const loading = document.createElement('div');
  loading.className = 'loading-state';
  loading.innerHTML = `<div class="spinner" aria-hidden="true"></div><h4>Running multimodal analysis</h4><p>Processing imaging and vitals…</p>`;
  xrayOutput.appendChild(loading.cloneNode(true));
  skinOutput.appendChild(loading.cloneNode(true));
  heartOutput.appendChild(loading.cloneNode(true));
}

function renderProbabilityChart(container, title, items) {
  if (!items || !items.length) return;
  const h = document.createElement('h4');
  h.textContent = title;
  container.appendChild(h);

  const chart = document.createElement('div');
  chart.className = 'prob-chart';
  items.forEach((it) => {
    const pct = Math.round((it.confidence || 0) * 100);
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <div class="name">${it.label}</div>
      <div class="pct">${pct}%</div>
      <div class="prob-bar-track" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="${it.label} probability">
        <div class="prob-bar-fill" style="width:${pct}%"></div>
      </div>`;
    chart.appendChild(row);
  });
  container.appendChild(chart);
}

function renderFusion(fusion) {
  const score = Math.round((fusion.overall_risk_score || 0) * 100);
  const level = fusion.overall_risk_level || 'unknown';
  riskLevelEl.textContent = `${level} (${score}%)`;
  riskBarFill.style.width = `${score}%`;
  riskBar.setAttribute('aria-valuenow', String(score));
  fusionSummary.className = `fusion-summary risk-${level}`;
  fusionRecommendationEl.textContent = fusion.recommendation || '';
  fusionNotesEl.innerHTML = '';
  (fusion.notes || []).forEach((note) => {
    const li = document.createElement('li');
    li.textContent = note;
    fusionNotesEl.appendChild(li);
  });
  ovFusion.textContent = `${level} (${score}%)`;
}

function appendResultImage(container, imageUrl, altText) {
  if (!imageUrl) return;
  const wrap = document.createElement('div');
  wrap.className = 'result-image';
  const img = document.createElement('img');
  img.src = imageUrl;
  img.alt = altText;
  wrap.appendChild(img);
  container.appendChild(wrap);
}

function appendGradcam(container, gradcamUrl, title) {
  if (!gradcamUrl) return;
  const block = document.createElement('div');
  block.className = 'gradcam-block';
  const heading = document.createElement('h4');
  heading.textContent = title;
  block.appendChild(heading);
  const wrap = document.createElement('div');
  wrap.className = 'result-image gradcam-image';
  const img = document.createElement('img');
  img.src = gradcamUrl;
  img.alt = `${title} heatmap overlay`;
  wrap.appendChild(img);
  block.appendChild(wrap);
  const caption = document.createElement('p');
  caption.className = 'confidence-label muted';
  caption.textContent = 'Warmer colors highlight regions that influenced this prediction.';
  block.appendChild(caption);
  container.appendChild(block);
}

function appendHeadline(container, title, iconName, text, className) {
  const headline = document.createElement('div');
  headline.className = 'modality-headline';
  headline.innerHTML = `<h4>${icon(iconName)} ${title}</h4><p class="${className || ''}"><strong>${text}</strong></p>`;
  container.appendChild(headline);
}

function appendConfidenceBar(container, confidence) {
  const pct = Math.round((confidence || 0) * 100);
  const track = document.createElement('div');
  track.className = 'confidence-track large';
  track.setAttribute('role', 'progressbar');
  track.setAttribute('aria-valuenow', String(pct));
  track.setAttribute('aria-valuemin', '0');
  track.setAttribute('aria-valuemax', '100');
  track.innerHTML = `<div class="confidence-fill" style="width:${pct}%"></div>`;
  container.appendChild(track);
  const label = document.createElement('p');
  label.className = 'confidence-label muted';
  label.textContent = `Confidence: ${pct}%`;
  container.appendChild(label);
}

function renderXray(xray, imageUrl) {
  xrayOutput.innerHTML = '';
  if (!xray || !xray.available) {
    xrayOutput.innerHTML = `<p class="muted">${icon('xray')} ${(xray && xray.error) || 'X-ray model not available.'}</p>`;
    return;
  }
  appendResultImage(xrayOutput, imageUrl, 'Uploaded chest X-ray');
  const predicted = xray.predicted_class || 'Unknown';
  const isPneumonia = String(predicted).toUpperCase() === 'PNEUMONIA';
  appendHeadline(
    xrayOutput,
    'Chest X-ray',
    'xray',
    xray.display_text || predicted,
    isPneumonia ? 'xray-positive' : 'xray-negative',
  );
  appendConfidenceBar(xrayOutput, xray.headline_confidence ?? xray.confidence);
  appendGradcam(xrayOutput, xray.gradcam_image, 'Grad-CAM — chest X-ray');
  renderProbabilityChart(xrayOutput, 'Class probabilities', xray.top5 || []);
}

function renderSkin(skin, imageUrl) {
  skinOutput.innerHTML = '';
  if (!skin) {
    skinOutput.innerHTML = `<div class="empty-state" style="padding:20px"><p class="muted">No skin image uploaded.</p></div>`;
    return;
  }
  if (!skin.available) {
    skinOutput.innerHTML = `<p class="muted">${skin.error || 'Skin model not available.'}</p>`;
    return;
  }
  appendResultImage(skinOutput, imageUrl, 'Uploaded skin lesion');
  appendHeadline(skinOutput, 'Skin lesion', 'skin', skin.display_text || skin.predicted_class, 'skin-headline');
  appendConfidenceBar(skinOutput, skin.headline_confidence ?? skin.confidence);
  appendGradcam(skinOutput, skin.gradcam_image, 'Grad-CAM — skin lesion');
  renderProbabilityChart(skinOutput, 'Top predictions', skin.top5 || skin.top3 || []);
}

function renderHeart(heart) {
  heartOutput.innerHTML = '';
  if (!heart) {
    heartOutput.innerHTML = `<p class="muted">No cardiac data submitted.</p>`;
    return;
  }
  if (!heart.available) {
    heartOutput.innerHTML = `<p class="muted">${heart.error || 'Heart model not available.'}</p>`;
    return;
  }
  const isPositive = String(heart.predicted_class || '').toLowerCase().includes('heart disease')
    && !String(heart.predicted_class || '').toLowerCase().includes('no');
  appendHeadline(
    heartOutput,
    'Cardiac assessment',
    'heart',
    heart.display_text || heart.predicted_class,
    isPositive ? 'heart-positive' : 'heart-negative',
  );
  appendConfidenceBar(heartOutput, heart.confidence ?? heart.probability);
  renderProbabilityChart(heartOutput, 'Class probabilities', heart.top2 || []);
}

async function handleSubmit(e) {
  e.preventDefault();
  clearAlert();
  const fd = new FormData();
  const xray = document.getElementById('xray').files[0];
  const skin = document.getElementById('skin').files[0];
  if (xray) fd.append('chest_xray_image', xray);
  if (skin) fd.append('skin_image', skin);
  fd.append('heart_features', JSON.stringify(collectHeartPayload()));

  resultsContent.classList.add('loading');
  document.getElementById('predict-btn').disabled = true;
  showLoadingState();

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${res.status})`);
    }
    const data = await res.json();
    const xrayPreviewUrl = document.querySelector('#xray-preview img')?.src || null;
    const skinPreviewUrl = document.querySelector('#skin-preview img')?.src || null;
    renderXray(data.outputs.xray, xrayPreviewUrl);
    renderSkin(data.outputs.skin, skinPreviewUrl);
    renderHeart(data.outputs.heart);
    renderFusion(data.outputs.fusion);
    resultEl.hidden = true;
    resultEl.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    showError(`Prediction failed: ${err.message}`);
    showEmptyResults();
    resultEl.hidden = false;
    resultEl.textContent = String(err.message);
  } finally {
    resultsContent.classList.remove('loading');
    document.getElementById('predict-btn').disabled = false;
  }
}

function handleTabClick(e) {
  const btn = e.target.closest('.nav-btn');
  const t = btn?.dataset.tab;
  if (!t) return;
  document.querySelectorAll('.tab').forEach((el) => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach((b) => b.classList.remove('active'));
  document.getElementById(t).classList.add('active');
  document.querySelectorAll(`.nav-btn[data-tab="${t}"]`).forEach((b) => b.classList.add('active'));
  if (pageTitleEl) pageTitleEl.textContent = PAGE_TITLES[t] || 'FusionNet';
}

function bindInPageTabs() {
  document.querySelectorAll('[data-tab-link]').forEach((button) => {
    button.addEventListener('click', () => {
      const target = button.getAttribute('data-tab-link');
      document.querySelectorAll(`.nav-btn[data-tab="${target}"]`).forEach((b) => b.click());
      document.getElementById(target)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

function setupPreview(inputId, previewId) {
  const inp = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  inp.addEventListener('change', () => {
    const f = inp.files[0];
    if (preview.dataset.objectUrl) {
      URL.revokeObjectURL(preview.dataset.objectUrl);
      delete preview.dataset.objectUrl;
    }
    if (!f) {
      preview.innerHTML = '';
      preview.setAttribute('aria-hidden', 'true');
      return;
    }
    const url = URL.createObjectURL(f);
    preview.dataset.objectUrl = url;
    preview.innerHTML = `<img src="${url}" alt="Upload preview" />`;
    preview.setAttribute('aria-hidden', 'false');
  });
}

function resetForm() {
  document.getElementById('predict-form').reset();
  ['xray-preview', 'skin-preview'].forEach((id) => {
    const preview = document.getElementById(id);
    if (preview.dataset.objectUrl) URL.revokeObjectURL(preview.dataset.objectUrl);
    preview.innerHTML = '';
    preview.setAttribute('aria-hidden', 'true');
  });
  clearAlert();
  showEmptyResults();
  ovFusion.textContent = 'Awaiting run';
  resultEl.hidden = true;
  resultEl.textContent = '';
}

document.getElementById('predict-form').addEventListener('submit', handleSubmit);
document.getElementById('reset-btn').addEventListener('click', resetForm);
document.querySelectorAll('.nav-btn').forEach((b) => b.addEventListener('click', handleTabClick));
document.getElementById('toggle-json').addEventListener('click', () => { resultEl.hidden = !resultEl.hidden; });

createHeartFields();
loadHealth();
setupPreview('xray', 'xray-preview');
setupPreview('skin', 'skin-preview');
bindInPageTabs();
showEmptyResults();
