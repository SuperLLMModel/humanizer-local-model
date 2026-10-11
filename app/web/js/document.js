// 长文分段改写(0.3.4,外部贡献 PR #5;集成时重写了切分,和 hz 的分段尺寸对齐)。
//
// 什么时候分段:草稿 token × 2.5 超过单次输出上限(2,048,也就是约 820 token 以上),或者提示词 + 输出放不进上下文。
// 这正是以前提示「结尾可能被截断,建议分段」的门槛;更短的草稿走原来的单篇流程,一点不变。
//
// 怎么切:先把 Create fact 的保留段换成占位符(和单篇改写同一个 protectFacts),占位符是原子,任何切点都不会落在里面,
// 所以跨段落的保留段也整个留在同一段。切点按优先级:空行 → 换行 → 句末(含中文。！？和后面的引号括号)→ 分句标点
// (，；：、 和英文 , ; :)→ 空白 → 实在没有就按字数硬切(不切开代理对)。相邻的小块合并成一段,尽量等长(hz 的
// _balanced 同一规则),每段目标约 SECTION_TOKENS(≈ hz 的 350 英文词 / 600 汉字)。
// 先用字数估算切,再用引擎真数每段的 token(外部服务不支持数 token 时就用估算);真数超过 SECTION_MAX 的段按实测比例再切。
//
// 每段之间原文里的空白(空行、缩进)原样留下,拼回去和原稿逐字一致,改写结果也按同样的空白拼接。
import { buildPrompt, nPredictFor } from './prompt.js';
import { activeFacts, protectFacts } from './facts.js';
import { estimateTokens } from './text.js';

export const SECTION_TOKENS = 560;
export const SECTION_MAX = 760;
const CTX_MARGIN = 64;

const checkAbort = (signal) => { if (signal?.aborted) throw new DOMException('stopped', 'AbortError'); };

/** 单篇改写放不下:输出可能顶到上限,或者提示词 + 输出超过上下文。 */
export function needsSections(cfg, draftTokens, promptTokens) {
  const r = cfg.n_predict || { factor: 2.5, max: 2048 };
  return draftTokens * r.factor > r.max || promptTokens + nPredictFor(cfg, draftTokens) > cfg.ctx_size;
}

/** 估算 token(偏保守):中文按字、英文按词,符号多的文本按字符数兜底。 */
export function approxTokens(s) {
  return Math.max(estimateTokens(s), Math.ceil(s.length / 4));
}

// 切点优先级。每个正则匹配的末尾就是切点(切点之后的空白归下一块,拼段时挪进 before)。
const LEVELS = [
  /\n[ \t\r　]*\n\s*/g,
  /\n\s*/g,
  /[.!?]+[)"'’”\]]*(?=\s)\s*|[。！？…]+[”’"」』)）》]*\s*/g,
  /[;:,]\s+|[；：，、]\s*/g,
  /\s+/g,
];

function cutPoints(text, from, to, re, locks) {
  const out = [];
  re.lastIndex = 0;
  const slice = text.slice(from, to);
  for (const m of slice.matchAll(re)) {
    const p = from + m.index + m[0].length;
    if (p <= from || p >= to) continue;
    if (locks.some((l) => p > l.start && p < l.end)) continue;
    out.push(p);
  }
  return out;
}

// 小块合并成段:总量超过 target 时分成 ceil(总量/target) 段,尽量等长(和 hz_text._balanced 同一规则)。
function balance(pieces, size, target) {
  if (!pieces.length) return [];
  const sizes = pieces.map(([a, b]) => size(a, b));
  const total = sizes.reduce((x, y) => x + y, 0);
  if (total <= target) return [[pieces[0][0], pieces.at(-1)[1]]];
  const goal = total / Math.ceil(total / target);
  const out = [];
  let cur = null, cw = 0;
  pieces.forEach(([a, b], i) => {
    const w = sizes[i];
    if (cur && (cw + w > target || cw + w / 2 > goal)) { out.push(cur); cur = null; cw = 0; }
    cur = cur ? [cur[0], b] : [a, b];
    cw += w;
  });
  if (cur) out.push(cur);
  return out;
}

// 实在没有可切的地方:按字数硬切,不切进占位符、不切开代理对。
function hardCut(text, from, to, size, target, locks) {
  const n = Math.max(2, Math.ceil(size(from, to) / target));
  const step = Math.max(1, Math.floor((to - from) / n));
  const out = [];
  let a = from;
  while (a < to) {
    let b = Math.min(to, a + step);
    if (to - b < step / 2) b = to; // 尾巴太短就并进来
    for (const l of locks) if (b > l.start && b < l.end) b = l.start > a ? l.start : l.end;
    if (b < to && /[\uD800-\uDBFF]/.test(text[b - 1])) b = b - 1 > a ? b - 1 : b + 1;
    out.push([a, b]);
    a = b;
  }
  return out;
}

function splitRange(text, from, to, size, target, locks, level = 0) {
  if (size(from, to) <= target || to - from < 2) return [[from, to]];
  if (level >= LEVELS.length) return hardCut(text, from, to, size, target, locks);
  const cuts = cutPoints(text, from, to, LEVELS[level], locks);
  if (!cuts.length) return splitRange(text, from, to, size, target, locks, level + 1);
  const pieces = [];
  let a = from;
  for (const c of cuts) { pieces.push([a, c]); a = c; }
  pieces.push([a, to]);
  const out = [];
  let run = [];
  for (const p of pieces) {
    if (size(p[0], p[1]) > target) {
      out.push(...balance(run, size, target));
      run = [];
      out.push(...splitRange(text, p[0], p[1], size, target, locks, level + 1));
    } else run.push(p);
  }
  out.push(...balance(run, size, target));
  return out;
}

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/**
 * 把草稿切成段。count(text, {signal}) 返回 token 数(引擎 /tokenize;外部服务用 approxTokens)。
 * 返回 [{ draft, before, after }]:draft 是这一段的原文(占位符已换回),before/after 是段前段后的空白;
 * 全部按顺序拼起来 = 原稿,一字不差。
 */
export async function planDocument(draft, facts, cfg, count, signal, { target = SECTION_TOKENS, max = SECTION_MAX } = {}) {
  const pd = protectFacts(draft, facts);
  const text = pd.text;
  const locks = [];
  if (pd.locks.length) {
    for (const m of text.matchAll(new RegExp(escapeRe('[[' + pd.prefix) + '\\d+\\]\\]', 'g'))) locks.push({ start: m.index, end: m.index + m[0].length });
  }
  const est = (a, b) => approxTokens(text.slice(a, b));
  const overhead = Math.max(0, (await count(buildPrompt(cfg, 'X'), { signal })) - (await count('X', { signal })));
  checkAbort(signal);
  const queue = splitRange(text, 0, text.length, est, target, locks).map((r) => ({ r, round: 0 }));
  const ranges = [];
  while (queue.length) {
    checkAbort(signal);
    const { r: [a, b], round } = queue.shift();
    const body = text.slice(a, b).trim();
    if (!body) { ranges.push([a, b]); continue; }
    const real = await count(body, { signal });
    checkAbort(signal);
    const fits = real + overhead + nPredictFor(cfg, real) + CTX_MARGIN <= cfg.ctx_size;
    if ((real > max || !fits) && round < 4) {
      // 估算偏小了(代码、网址、数字多的文本):按实测比例重新切这一段
      const scale = real / Math.max(1, est(a, b));
      const sub = splitRange(text, a, b, (x, y) => est(x, y) * scale, target, locks);
      if (sub.length > 1) { queue.unshift(...sub.map((r) => ({ r, round: round + 1 }))); continue; }
    }
    if (!fits) throw new Error('documentContext');
    ranges.push([a, b]);
  }
  // 换回原文,段首段尾的空白单独放;只有空白的块并进前一段(在最前面就并进下一段)
  const parts = [];
  let pending = '';
  for (const [a, b] of ranges) {
    const raw = pending + restoreSlice(text.slice(a, b), pd);
    pending = '';
    const body = raw.trim();
    if (body) {
      const lead = raw.length - raw.trimStart().length;
      const trail = raw.length - raw.trimEnd().length;
      parts.push({ draft: body, before: raw.slice(0, lead), after: trail ? raw.slice(raw.length - trail) : '' });
    } else if (parts.length) parts.at(-1).after += raw;
    else pending = raw;
  }
  if (pending) {
    if (parts.length) parts[0].before = pending + parts[0].before;
    else throw new Error('documentEmpty');
  }
  return parts;
}

// 切片里的占位符都是完整的(切点不落在占位符里),逐个精确换回。
function restoreSlice(slice, pd) {
  return slice.replace(new RegExp(escapeRe('[[' + pd.prefix) + '(\\d+)\\]\\]', 'g'), (m, i) => pd.locks[+i]?.value ?? m);
}

// ───────── 断点(浏览器 localStorage)─────────
// engine 标识(档位或外部服务地址 + 模型)、提示词、采样、上下文任何一样变了,断点就作废。
export function documentSignature(cfg, engine) {
  return JSON.stringify([engine, cfg.instr, cfg.sep, cfg.ctx_size, cfg.sampling, cfg.n_predict]);
}

export function newDocumentJob(draft, facts, signature, parts) {
  return { version: 1, draft, facts: activeFacts(facts, draft), signature, parts, outputs: [] };
}

export function readDocumentJob(raw) {
  try {
    const j = JSON.parse(raw);
    if (j?.version !== 1 || typeof j.draft !== 'string' || typeof j.signature !== 'string' ||
      !Array.isArray(j.facts) || !j.facts.every((f) => typeof f === 'string') || !Array.isArray(j.parts) || !j.parts.length ||
      !j.parts.every((p) => p && ['draft', 'before', 'after'].every((k) => typeof p[k] === 'string') && p.draft.trim()) ||
      j.parts.map((p) => p.before + p.draft + p.after).join('') !== j.draft ||
      !Array.isArray(j.outputs) || j.outputs.length > j.parts.length || !j.outputs.every((s) => typeof s === 'string' && s.trim())) return null;
    // 恢复的断点里,已完成的每段都必须还带着这段里的保留文字
    if (j.outputs.some((s, i) => activeFacts(j.facts, j.parts[i].draft).some((f) => !s.includes(f)))) return null;
    return j;
  } catch { return null; }
}

/** 断点是不是这篇草稿 + 这些保留段的(signature 不给就不比)。 */
export function jobMatches(job, draft, facts, signature = job?.signature) {
  return !!job && job.draft === draft && job.signature === signature &&
    JSON.stringify(job.facts) === JSON.stringify(activeFacts(facts, draft));
}

export const jobDone = (job) => !!job && job.outputs.length === job.parts.length;

/** 已完成的段按原来的空白拼起来(没完成的段不拼)。 */
export function combinedDocument(job) {
  return job.outputs.map((s, i) => job.parts[i].before + s + job.parts[i].after).join('');
}

/**
 * 从第一个没完成的段接着改。rewrite(段原文, 保留段, signal) 返回这一段的改写(保留段已换回)。
 * 只有完整、非空、带齐保留文字的结果才记进断点;失败或停止的那段下次重来,已完成的段不会再送进引擎。
 */
export async function runDocumentSections(job, rewrite, { signal, onCommit = () => {}, onProgress = () => {} } = {}) {
  for (let i = job.outputs.length; i < job.parts.length; i++) {
    checkAbort(signal);
    onProgress(i, job.parts.length);
    const output = await rewrite(job.parts[i].draft, job.facts, signal, i);
    checkAbort(signal);
    if (typeof output !== 'string' || !output.trim()) throw new Error('documentIncomplete');
    if (activeFacts(job.facts, job.parts[i].draft).some((f) => !output.includes(f))) throw new Error('factsLost');
    job.outputs.push(output.trim());
    onCommit(job);
  }
  onProgress(job.parts.length, job.parts.length);
  return combinedDocument(job);
}
