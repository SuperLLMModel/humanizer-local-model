// 和启动器说话。所有写操作都带 X-Humanizer 头(启动器据此挡掉别的网站发来的请求)。
const H = { 'X-Humanizer': '1' };

// Keep the deadline active through response-body parsing, and distinguish it
// from the caller's cancellation. Always release the timer and abort listener.
async function withRequestSignal({ timeout, signal }, request) {
  const ctl = new AbortController();
  const onAbort = () => ctl.abort(signal.reason);
  let timer;
  try {
    if (signal?.aborted) throw signal.reason;
    signal?.addEventListener('abort', onAbort, { once: true });
    timer = setTimeout(() => ctl.abort(new DOMException('Request timed out. Please try again.', 'TimeoutError')), timeout);
    return await request(ctl.signal);
  } catch (error) {
    if (ctl.signal.aborted) throw ctl.signal.reason;
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }
}

export async function getJSON(path, { timeout = 6000, signal } = {}) {
  return withRequestSignal({ timeout, signal }, async (requestSignal) => {
    const r = await fetch(path, { cache: 'no-store', signal: requestSignal });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  });
}

export async function postJSON(path, body, { timeout = 10000, signal } = {}) {
  return withRequestSignal({ timeout, signal }, async (requestSignal) => {
    const r = await fetch(path, {
      method: 'POST',
      headers: { ...H, 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
      signal: requestSignal,
    });
    let data = null;
    try { data = await r.json(); } catch (error) {
      if (requestSignal.aborted) throw requestSignal.reason;
      // Preserve support for empty/non-JSON launcher responses.
    }
    if (!r.ok) throw new Error((data && (data.error?.message || data.error)) || `HTTP ${r.status}`);
    return data;
  });
}

/** Tokenization is cancellable and gets a little more time on slower machines. */
export async function countTokens(content, { timeout = 15000, signal } = {}) {
  const d = await postJSON('/api/tokenize', { content, add_special: false }, { timeout, signal });
  return Array.isArray(d.tokens) ? d.tokens.length : 0;
}

/**
 * 流式补全:逐个 yield llama-server 的 SSE 事件对象。
 * 跳过 ": ping" 这类 SSE 注释;遇到 error 事件抛异常。
 */
export async function* streamCompletion(body, signal) {
  const r = await fetch('/api/completion', {
    method: 'POST',
    headers: { ...H, 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { const j = await r.json(); msg = j.error?.message || msg; } catch { /* 不是 JSON */ }
    throw new Error(msg);
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).replace(/\r$/, '');
      buf = buf.slice(nl + 1);
      if (!line || line.startsWith(':')) continue;
      let payload = null;
      if (line.startsWith('data:')) payload = line.slice(5).trim();
      else if (line.startsWith('error:')) {
        let msg = line.slice(6).trim();
        try { msg = JSON.parse(msg).message || msg; } catch { /* 原文 */ }
        throw new Error(msg);
      }
      if (!payload || payload === '[DONE]') continue;
      let ev;
      try { ev = JSON.parse(payload); } catch { continue; }
      if (ev.error) throw new Error(ev.error.message || 'engine error');
      yield ev;
    }
  }
}

/**
 * 外部 OpenAI 兼容服务(启动器反代 /api/v1/chat/completions,见 internal/launcher/external.go)。
 * 把 chat.completion.chunk 流换成和 llama-server /completion 一样的事件:
 *   { content } …… 最后 { stop: true, stop_type: 'eos' | 'limit', tokens_predicted, timings }
 * 服务端只发 [DONE]、不给 finish_reason 时按正常结束算;连接中途断开(没有 [DONE] 也没有 finish_reason)不算完成。
 */
export async function* streamChat(body, signal) {
  const r = await fetch('/api/v1/chat/completions', {
    method: 'POST',
    headers: { ...H, 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`, type = '';
    try {
      const j = await r.json();
      const e = j.error;
      msg = (e && (e.message || (typeof e === 'string' ? e : ''))) || j.message || j.detail || msg;
      type = e?.type || '';
    } catch { /* 不是 JSON */ }
    throw Object.assign(new Error(msg), { status: r.status, type, external: true });
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  const t0 = performance.now();
  let buf = '', n = 0, finish = null;
  const final = () => {
    const secs = (performance.now() - t0) / 1000;
    return { content: '', stop: true, stop_type: finish === 'length' ? 'limit' : 'eos', tokens_predicted: n,
      timings: { predicted_per_second: secs > 0 && n ? n / secs : undefined } };
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).replace(/\r$/, '');
      buf = buf.slice(nl + 1);
      if (!line || line.startsWith(':') || !line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') { yield final(); return; }
      let ev;
      try { ev = JSON.parse(payload); } catch { continue; }
      if (ev.error) throw Object.assign(new Error(ev.error.message || String(ev.error)), { external: true });
      const ch = ev.choices?.[0];
      if (!ch) continue;
      const piece = ch.delta?.content ?? ch.text ?? '';
      if (piece) { n++; yield { content: piece }; }
      if (ch.finish_reason) {
        finish = ch.finish_reason;
        if (ev.usage?.completion_tokens) n = ev.usage.completion_tokens;
        yield final();
        return;
      }
    }
  }
  if (finish) yield final(); // 理论上走不到;没有 finish_reason 的断流不给 stop 事件
}
