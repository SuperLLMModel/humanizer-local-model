// 外部模型服务的设置面板(0.3.4,HF 讨论 #11)。真正的反代在启动器(internal/launcher/external.go):
// 网页只和启动器说话,API key 只发给启动器、存在本机数据目录,之后不会再回到网页里。
import { t } from './i18n.js';
import { getJSON, postJSON } from './api.js';

const $ = (id) => document.getElementById(id);
const E = { open: false, opener: null, st: null, keyDirty: false, clearKey: false, busy: false };
let hooks = { isRunning: () => false, onChange: () => {}, toast: () => {} };

/** 测试结果 → 一句给人看的话(中英文) */
export function describeTest(r) {
  if (r.ok) {
    const n = r.models?.length || 0;
    return { kind: 'ok', text: n ? t('ext.testOk', { n, models: r.models.slice(0, 4).join(', ') + (n > 4 ? ' …' : '') }) : t('ext.testOkNoModels') };
  }
  const key = { bad_url: 'ext.badUrl', auth: 'ext.auth', unreachable: 'ext.unreachable' }[r.code] || 'ext.httpErr';
  return { kind: 'err', text: t(key, { url: r.url || '', msg: r.message || '' }) };
}

/** 没填模型名时从列表里挑一个:名字带 humanizer 的优先,只有一个就用它。 */
export function pickModel(models) {
  if (!Array.isArray(models) || !models.length) return '';
  return models.find((m) => /humanizer/i.test(m)) || (models.length === 1 ? models[0] : '');
}

function showResult(kind, text) {
  const el = $('ext-result');
  el.hidden = !text;
  el.className = 'ext-result' + (kind ? ' ' + kind : '');
  el.textContent = text || '';
}

function render() {
  const st = E.st || {};
  $('ext-key').placeholder = st.has_key && !E.clearKey ? t('ext.keySaved') : t('ext.keyPh');
  $('ext-key-clear').hidden = !st.has_key || E.clearKey;
  const running = hooks.isRunning();
  $('ext-save').disabled = E.busy || running;
  $('ext-save').title = running ? t('ext.busyRunning') : '';
  $('ext-test').disabled = E.busy;
  $('ext-test').classList.toggle('spinning', E.busy);
}

function fill(st) {
  E.st = st;
  E.keyDirty = false;
  E.clearKey = false;
  $('ext-enabled').checked = !!st.enabled;
  $('ext-url').value = st.url || '';
  $('ext-model').value = st.model || '';
  $('ext-key').value = '';
  render();
}

function keyPayload() {
  if (E.clearKey) return '';
  if (E.keyDirty) return $('ext-key').value.trim();
  return undefined; // 不改
}

async function test() {
  const url = $('ext-url').value.trim() || 'http://localhost:1234/v1';
  if (!$('ext-url').value.trim()) $('ext-url').value = url;
  E.busy = true;
  render();
  showResult('', t('ext.testing'));
  let r;
  try {
    r = await postJSON('/app/external/test', { url, api_key: keyPayload() }, { timeout: 15000 });
  } catch (e) {
    r = { ok: false, code: 'http', message: e.message };
  }
  E.busy = false;
  const d = describeTest(r);
  showResult(d.kind, d.text);
  if (r.ok) {
    $('ext-models').replaceChildren(...(r.models || []).map((m) => Object.assign(document.createElement('option'), { value: m })));
    if (!$('ext-model').value.trim()) $('ext-model').value = pickModel(r.models);
  }
  render();
  return r;
}

async function save() {
  if (hooks.isRunning()) return;
  const enabled = $('ext-enabled').checked;
  const url = $('ext-url').value.trim();
  if (enabled && !url) { showResult('err', t('ext.badUrl')); $('ext-url').focus(); return; }
  // 开启前先测一下;连不上照样保存(服务可能还没启动),但留着面板把原因写清楚
  let probe = null;
  if (enabled) probe = await test();
  E.busy = true;
  render();
  try {
    const st = await postJSON('/app/external', { enabled, url, model: $('ext-model').value.trim(), api_key: keyPayload() }, { timeout: 30000 });
    fill(st);
    hooks.onChange(st);
    if (!enabled) { showResult('', ''); hooks.toast(t('ext.savedOff'), 'ok'); openExternal(false); }
    else if (probe?.ok) { hooks.toast(t('ext.savedOn'), 'ok'); openExternal(false); }
    else showResult('err', describeTest(probe || { ok: false, code: 'unreachable' }).text + ' ' + t('ext.savedAnyway'));
  } catch (e) {
    showResult('err', t(e.message === 'bad url' ? 'ext.badUrl' : 'ext.saveFailed', { msg: e.message }));
  } finally {
    E.busy = false;
    render();
  }
}

export async function openExternal(open = true, opener = null) {
  if (open === E.open) return;
  E.open = open;
  $('ext').hidden = !open;
  $('ext-scrim').hidden = !open;
  if (open) {
    E.opener = opener || document.activeElement;
    showResult('', '');
    try { fill(await getJSON('/app/external')); } catch { fill({}); }
    requestAnimationFrame(() => (($('ext-url').value ? $('ext-close') : $('ext-url'))).focus({ preventScroll: true }));
  } else if (E.opener && document.contains(E.opener)) {
    E.opener.focus({ preventScroll: true });
  }
}

export function bindExternal(h) {
  hooks = { ...hooks, ...h };
  $('ext-close').addEventListener('click', () => openExternal(false));
  $('ext-scrim').addEventListener('click', () => openExternal(false));
  $('ext-test').addEventListener('click', () => test());
  $('ext-save').addEventListener('click', () => save());
  $('ext-key').addEventListener('input', () => { E.keyDirty = true; E.clearKey = false; render(); });
  $('ext-key-clear').addEventListener('click', () => { E.clearKey = true; E.keyDirty = false; $('ext-key').value = ''; render(); });
  for (const id of ['ext-url', 'ext-model', 'ext-key']) {
    $(id).addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); save(); } });
  }
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && E.open) { e.stopPropagation(); openExternal(false); }
  }, true);
}

export const externalOpen = () => E.open;
export const refreshExternal = () => { if (E.open) render(); };
