/* Frontend interaction script for FusionNet UI */
const heartFieldsContainer = document.getElementById('heart-fields');
const resultEl = document.getElementById('result');
const modelStatusEl = document.getElementById('model-status');
const resultsContent = document.getElementById('results-content');
const xrayOutput = document.getElementById('xray-output');
const skinOutput = document.getElementById('skin-output');
const heartOutput = document.getElementById('heart-output');
const riskLevelEl = document.getElementById('risk-level');
const riskBarFill = document.getElementById('risk-bar-fill');

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

const heartSchema = {
  Sex: ['M', 'F'],
  ChestPainType: ['TA', 'ATA', 'NAP', 'ASY'],
  RestingECG: ['Normal', 'ST', 'LVH'],
  ExerciseAngina: ['Y', 'N'],
  ST_Slope: ['Up', 'Flat', 'Down'],
  numeric: ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak'],
};

function createHeartFields() {
  heartFieldsContainer.innerHTML = '';
  heartSchema.numeric.forEach((field) => {
    const wrap = document.createElement('div');
    wrap.className = 'field';
    wrap.innerHTML = `<label>${field}</label><input type="number" step="any" id="heart_${field}" value="${heartDefaults[field]}" />`;
    heartFieldsContainer.appendChild(wrap);
  });

  Object.keys(heartSchema).filter((k) => k !== 'numeric').forEach((field) => {
    const options = heartSchema[field];
    const select = document.createElement('select');
    select.id = `heart_${field}`;
    options.forEach(o => {
      const opt = document.createElement('option');
      opt.value = o; opt.textContent = o;
      if (heartDefaults[field] === o) opt.selected = true;
      select.appendChild(opt);
    });
    const wrap = document.createElement('div');
    wrap.className = 'field';
    wrap.innerHTML = `<label>${field}</label>`;
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
  Object.keys(heartSchema).filter(k => k !== 'numeric').forEach((f) => {
    payload[f] = document.getElementById(`heart_${f}`).value;
  });
  return payload;
}

async function loadHealth() {
  try {
    const r = await fetch('/api/health');
    const data = await r.json();
    modelStatusEl.innerHTML = `Models: <span class="badge ${data.models.xray ? 'on' : 'off'}">X-ray</span> <span class="badge ${data.models.skin ? 'on' : 'off'}">Skin</span> <span class="badge ${data.models.heart ? 'on' : 'off'}">Heart</span>`;
  } catch (err) {
    modelStatusEl.textContent = 'Unable to reach API';
  }
}

function renderTopList(container, title, items) {
  container.innerHTML = '';
  const h = document.createElement('h4'); h.textContent = title; container.appendChild(h);
  const ul = document.createElement('ul'); ul.className = 'top-list';
  items.forEach(it => {
    const li = document.createElement('li');
    li.innerHTML = `<div class="label">${it.label}</div><div class="conf">${(it.confidence*100).toFixed(1)}%</div>`;
    ul.appendChild(li);
  });
  container.appendChild(ul);
}

function renderFusion(fusion) {
  const score = Math.round((fusion.overall_risk_score || 0) * 100);
  riskLevelEl.textContent = `${fusion.overall_risk_level || '—'} (${score}%)`;
  riskBarFill.style.width = `${score}%`;
}

function renderXray(xray) {
  if (!xray || !xray.available) {
    xrayOutput.innerHTML = '<p class="muted">X-ray model not available.</p>';
    return;
  }
  const top5 = xray.top5 || [];
  renderTopList(xrayOutput, 'Chest X-ray Top Predictions', top5);
}

function renderSkin(skin) {
  if (!skin) { skinOutput.innerHTML = '<p class="muted">No skin result.</p>'; return; }
  if (!skin.available) { skinOutput.innerHTML = '<p class="muted">Skin model not available.</p>'; return; }
  renderTopList(skinOutput, 'Skin Lesion Predictions', skin.top5 || []);
}

function renderHeart(heart) {
  if (!heart) { heartOutput.innerHTML = '<p class="muted">No heart result.</p>'; return; }
  if (!heart.available) { heartOutput.innerHTML = `<p class="muted">${heart.error || 'Heart model not available.'}</p>`; return; }
  heartOutput.innerHTML = `<h4>Cardiac Prediction</h4><p>Probability: ${(heart.probability||0).toFixed(3)}</p>`;
}

async function handleSubmit(e) {
  e.preventDefault();
  const fd = new FormData();
  const xray = document.getElementById('xray').files[0];
  const skin = document.getElementById('skin').files[0];
  if (xray) fd.append('chest_xray_image', xray);
  if (skin) fd.append('skin_image', skin);
  fd.append('heart_features', JSON.stringify(collectHeartPayload()));

  // show loading
  resultsContent.classList.add('loading');
  document.getElementById('predict-btn').disabled = true;

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    const data = await res.json();
    // structured render
    renderXray(data.outputs.xray);
    renderSkin(data.outputs.skin);
    renderHeart(data.outputs.heart);
    renderFusion(data.outputs.fusion);
    resultEl.hidden = true;
    resultEl.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    resultEl.hidden = false;
    resultEl.textContent = `Prediction failed: ${err.message}`;
  } finally {
    resultsContent.classList.remove('loading');
    document.getElementById('predict-btn').disabled = false;
  }
}

function handleTabClick(e) {
  const t = e.target.dataset.tab;
  if (!t) return;
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(t).classList.add('active');
  e.target.classList.add('active');
}

function bindInPageTabs() {
  document.querySelectorAll('[data-tab-link]').forEach((button) => {
    button.addEventListener('click', () => {
      const target = button.getAttribute('data-tab-link');
      const targetButton = document.querySelector(`.nav-btn[data-tab="${target}"]`);
      if (targetButton) targetButton.click();
      document.getElementById(target)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

function setupPreview(inputId, previewId) {
  const inp = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  inp.addEventListener('change', () => {
    const f = inp.files[0];
    if (!f) { preview.innerHTML = ''; preview.setAttribute('aria-hidden','true'); return; }
    const url = URL.createObjectURL(f);
    preview.innerHTML = `<img src="${url}" alt="preview"/>`;
    preview.setAttribute('aria-hidden','false');
  });
}

document.getElementById('predict-form').addEventListener('submit', handleSubmit);
document.getElementById('reset-btn').addEventListener('click', () => { document.getElementById('predict-form').reset(); document.getElementById('xray-preview').innerHTML=''; document.getElementById('skin-preview').innerHTML=''; });
document.querySelectorAll('.nav-btn').forEach(b => b.addEventListener('click', handleTabClick));
document.getElementById('toggle-json').addEventListener('click', () => { resultEl.hidden = !resultEl.hidden; });

// Theme switcher: apply saved theme or default
function applySavedTheme() {
  try {
    const saved = localStorage.getItem('uiTheme');
    if (saved) document.body.classList.add(saved);
  } catch (e) { /* ignore */ }
}

function initThemeSwatches() {
  document.querySelectorAll('.theme-swatch').forEach(s => {
    s.addEventListener('click', () => {
      document.querySelectorAll('.theme-swatch').forEach(x => x.classList.remove('active'));
      s.classList.add('active');
      const t = s.dataset.theme;
      document.body.classList.remove('theme-blue-a','theme-blue-b');
      if (t) document.body.classList.add(t);
      try { localStorage.setItem('uiTheme', t); } catch (e) {}
    });
  });
  // mark active swatch on load
  const current = Array.from(document.body.classList).find(c => c.startsWith('theme-'));
  if (current) {
    const el = document.querySelector(`.theme-swatch[data-theme="${current}"]`);
    if (el) el.classList.add('active');
  }
}

applySavedTheme();
initThemeSwatches();

createHeartFields();
loadHealth();
setupPreview('xray','xray-preview');
setupPreview('skin','skin-preview');
bindInPageTabs();
