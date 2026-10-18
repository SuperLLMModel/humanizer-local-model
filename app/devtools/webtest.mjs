import { getJSON, postJSON, countTokens } from '../web/js/api.js';
import { planDocument, needsSections, newDocumentJob, documentSignature, readDocumentJob, jobMatches, jobDone, combinedDocument, runDocumentSections, SECTION_MAX } from '../web/js/document.js';
import { store, addHistory, loadHistory, removeHistory, clearHistory } from '../web/js/history.js';
// 网页纯逻辑的单测(不需要浏览器):node devtools/webtest.mjs 或 bun devtools/webtest.mjs
import { activeFacts, protectFacts, restoreFacts, streamView, withFactGuard, FACT_TRIES } from '../web/js/facts.js';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { pyStrip, countText, numberCheck, estimateTokens, isLongDraft } from '../web/js/text.js';
import { buildPrompt, nPredictFor, selfCheck } from '../web/js/prompt.js';
import { tokenize, diffTokens } from '../web/js/diff.js';
import { hanCount, wentChinese, wentEnglish, langDrift, withLangGuard, LANG_RETRIES } from '../web/js/guard.js';
import { customPaths, modelDirDefault, pathsHintKey, sameDir } from '../web/js/paths.js';

let fails = 0;
const ok = (cond, msg) => { if (!cond) { fails++; console.error('✗', msg); } else console.log('✓', msg); };

// 1. strip 和 Python 一致(期望值由 python3 str.strip() 生成)
const STRIP = [[" \n\t\u3000\u001c Hello world.\u00a0\n\n", "Hello world."], ["\ufeffx\ufeff", "\ufeffx\ufeff"], ["\u200bzero\u200b", "\u200bzero\u200b"], ["\u0085a\u001f", "a"], ["a\n\n b", "a\n\n b"], ["\u2028x\u2029", "x"], ["\u000bv\f", "v"]];
for (const [inp, want] of STRIP) ok(pyStrip(inp) === want, `pyStrip(${JSON.stringify(inp)})`);

// 2. 提示词指纹 = promptfmt.py 的 fingerprint()
const INSTR = 'Rewrite the text below so it reads like a person wrote it, not a language model.\n\nReorganize it as you see fit. Vary sentence length on purpose. Cut hedging,\nthroat-clearing, and any sentence that only announces what comes next.\nPrefer the concrete word over the abstract one. It is fine to sound uneven.\n\nEvery fact, number, unit, date, name and quotation must survive unchanged.';
const cfg = { instr: INSTR, sep: '\n\n### Rewritten:\n\n', n_predict: { factor: 2.5, min: 256, max: 2048 } };
const fp = createHash('sha256').update(buildPrompt(cfg, 'X')).digest('hex').slice(0, 16);
ok(fp === 'cc51d66b4c593fbe', `提示词指纹 ${fp} == promptfmt.py cc51d66b4c593fbe`);
ok(selfCheck({ ...cfg, probe: buildPrompt(cfg, 'X') }) && !selfCheck({ ...cfg, probe: 'nope' }), 'selfCheck 正反两面');
ok(!buildPrompt(cfg, 'a').includes('###\n') && buildPrompt(cfg, '  a  ').endsWith('\n\na\n\n### Rewritten:\n\n'), '草稿两端 strip、分隔符正确');

// 3. n_predict = clamp(ceil(n × 2.5), 256, 2048)
ok(nPredictFor(cfg, 10) === 256 && nPredictFor(cfg, 200) === 500 && nPredictFor(cfg, 101) === 256 && nPredictFor(cfg, 103) === 258 && nPredictFor(cfg, 5000) === 2048, 'n_predict 规则');

// 4. 字数
const c1 = countText('Hello, world! 你好世界');
ok(c1.latin === 2 && c1.cjk === 4 && c1.units === 6, `字数统计 ${JSON.stringify(c1)}`);

// 5. 差异:英文按词、中文按字
{
  const a = tokenize('The quick brown fox jumps over the lazy dog.');
  const b = tokenize('The quick red fox leaps over the lazy dog.');
  const d = diffTokens(a, b);
  const insB = b.filter((x) => !x.ws && d.b[x.ci]).map((x) => x.t);
  const delA = a.filter((x) => !x.ws && d.a[x.ci]).map((x) => x.t);
  ok(JSON.stringify(insB) === '["red","leaps"]' && JSON.stringify(delA) === '["brown","jumps"]', `英文词级差异 +${insB} -${delA}`);
}
{
  const a = tokenize('我今天去北京开会。');
  const b = tokenize('我明天去北京开会。');
  const d = diffTokens(a, b);
  const insB = b.filter((x) => !x.ws && d.b[x.ci]).map((x) => x.t).join('');
  ok(insB === '明', `中文字级差异 +${insB}`);
}
{
  // 语义清理:深度改写时不应留下孤零零的 "the"
  const a = tokenize('It was a great salary with smart colleagues and the free meals were nice.');
  const b = tokenize('Pay was huge, people were sharp, the food cost nothing.');
  const d = diffTokens(a, b);
  const kept = b.filter((x) => !x.ws && !d.b[x.ci]).map((x) => x.t);
  ok(kept.length <= 2, `深改写不留碎片,保留的 token: ${JSON.stringify(kept)}`);
}
{
  const same = tokenize('完全相同的一句话。');
  const d = diffTokens(same, tokenize('完全相同的一句话。'));
  ok(d.ratio === 0 && !d.a.some(Boolean), '相同文本无差异');
}

// 6. 数字核对
{
  const r = numberCheck('From $1,850 to 3,200 users on 6:40, 60 minutes, 2024.', 'From 1850 dollars to 3200 users at 6:40 — an hour — in 2024');
  ok(JSON.stringify(r.missing) === '["60"]' && r.total === 5, `数字核对 ${JSON.stringify(r)}`);
  const r2 = numberCheck('晚上8点到10点半', '晚上8点到10:30');
  ok(r2.missing.length === 0, '10 点半 → 10:30 不算丢');
}

// 7. 用真实夹具跑一遍差异,确认不炸、比例合理
const fx = JSON.parse(readFileSync(new URL('./fixtures/samples.json', import.meta.url), 'utf8'));
for (const s of fx.samples) {
  const d = diffTokens(tokenize(s.draft), tokenize(s.output));
  const r = numberCheck(s.draft, s.output);
  ok(d && d.ratio > 0.2 && d.ratio < 0.95, `${s.id} 改动比例 ${(d.ratio * 100).toFixed(0)}%,数字缺 ${JSON.stringify(r.missing)}`);
}

// 8. 语言保险:英文草稿写成中文 → 重采,最多 3 次;反方向只在写完后判
{
  const EN = 'The current humanizer still carries some AI style (you can probably still feel it), and it can\'t get past the most accurate AI detector, Pangram. In my internal research, though, a brand-new method can actually teach a model to write like a human, not just rewrite, and it already passes Pangram v4 as a side proof.';
  const ZH_OUT = '现在的 humanizer 还是带点 AI 味（你大概还能感觉出来），也过不了目前最准的检测器 Pangram。不过我内部研究里有个全新的方法，能真正教模型像人一样写。';
  const EN_OUT = 'Honestly, the humanizer still reads a bit like AI, and Pangram, the sharpest detector out there, still catches it. But a new method I have been testing teaches the model to write, not just rewrite.';
  const ZH = '我们在2024年第3季度完成了1,250个工单的迁移，团队一共用了6周时间，比原计划提前了两周，客户满意度也有明显提升。';
  ok(hanCount(ZH_OUT) > 15 && hanCount(EN) === 0, `汉字计数 ${hanCount(ZH_OUT)}`);
  ok(wentChinese(EN, ZH_OUT) && !wentChinese(EN, EN_OUT), '英文草稿:中文输出算跑偏,英文输出不算');
  ok(!wentChinese(EN + ' 中文字', '这是一句带了二十多个汉字的中文输出，用来确认草稿里有三个汉字时不触发。') , '草稿 ≤2 个汉字才启用(3 个汉字不启用)');
  ok(!wentChinese(EN, EN_OUT + ' 这里夹了几个汉字'), '英文输出夹几个汉字(≤15)不算跑偏');
  ok(!wentChinese(ZH, ZH_OUT), '中文草稿写中文不算跑偏');
  ok(wentEnglish(ZH, EN_OUT) && !wentEnglish(ZH, '我们2024年Q3迁移了1,250个工单，6周干完，提前两周。') && !wentEnglish(EN, EN_OUT), '中文草稿写成英文才算');
  ok(!langDrift(ZH, 'Q3 migration of 1,250 tickets finished in six weeks, two weeks early, and customers were noticeably happier', false)
     && langDrift(ZH, 'Q3 migration of 1,250 tickets finished in six weeks, two weeks early, and customers were noticeably happier', true), '中→英只在写完后判,流式途中不判');

  const run = async (draft, outs) => {
    const calls = [];
    const r = await withLangGuard(draft, async (i, isLast) => { calls.push(isLast); return { text: outs[Math.min(i, outs.length - 1)] }; });
    return { r, calls };
  };
  let x = await run(EN, [ZH_OUT, ZH_OUT, EN_OUT]);
  ok(x.r.text === EN_OUT && x.r.tries === 3 && x.calls.join() === 'false,false,false', `跑偏两次后第 3 次英文 → 交英文(共 ${x.r.tries} 次)`);
  x = await run(EN, [ZH_OUT]);
  ok(LANG_RETRIES === 3 && x.r.tries === 4 && x.calls.join() === 'false,false,false,true' && x.r.text === ZH_OUT, '一直跑偏:重采 3 次后交最后一次,最后一次标 isLast');
  x = await run(EN, [EN_OUT]);
  ok(x.r.tries === 1, '正常输出只跑 1 次');
  x = await run(ZH, [ZH_OUT]);
  ok(x.r.tries === 1, '中文草稿中文输出只跑 1 次');
  const r2 = await withLangGuard(EN, async (i) => ({ text: i === 0 ? '半截' : EN_OUT, drifted: i === 0 }));
  ok(r2.tries === 2 && r2.text === EN_OUT, '流式途中掐掉的那次(drifted)也会重采');
  let threw = false;
  try { await withLangGuard(EN, async () => { throw Object.assign(new Error('stop'), { name: 'AbortError' }); }); } catch (e) { threw = e.name === 'AbortError'; }
  ok(threw, '用户停止(AbortError)直接抛出,不重采');
}

// 9. 长文档提示:约 1,000 token(≈ 700 个英文词 / 1,100 个汉字)起显示
{
  const words = (n) => Array.from({ length: n }, () => 'word').join(' ');
  ok(isLongDraft(words(700)) && !isLongDraft(words(400)), `长文档提示:英文 700 词显示、400 词不显示(估 ${estimateTokens(words(700))} token)`);
  ok(isLongDraft('汉'.repeat(1100)) && !isLongDraft('汉'.repeat(900)), '长文档提示:1,100 个汉字显示、900 个不显示');
  ok(!isLongDraft(''), '空草稿不显示');
}

// Literal facts preserve overlaps, repetitions, Unicode, and reject invalid output.
{
  const draft = 'Nadia approved 42 files. Nadia approved 42 files. 你好世界';
  const facts = ['Nadia approved', 'approved 42 files', '你好世界'];
  const locked = protectFacts(draft, facts);
  ok(locked.locks.length === 3, 'Overlapping facts merge; repeated occurrences get individual locks');
  ok(restoreFacts(locked.text, locked) === draft, 'Locks round-trip Unicode and all occurrences');
  const output = restoreFacts(locked.text.replace('. ', '! '), locked);
  ok(facts.every((f) => output.includes(f)) && output !== draft, 'Prose changes while literal facts survive');
  const collision = protectFacts('HZ_LOCK_ literal and 42', ['42']);
  ok(collision.prefix !== 'HZ_LOCK_' && restoreFacts(collision.text, collision) === 'HZ_LOCK_ literal and 42', 'Token namespace avoids collisions');
  ok(activeFacts(['42', '42', 'gone', '', null], '42 files').join() === '42', 'Deduplicate facts and remove stale or invalid entries');
  ok(activeFacts({}, '42').length === 0, 'Invalid stored facts are ignored');
  const many = Array.from({length:60}, (_,i)=>String(i));
  ok(activeFacts(many, many.join(' ')).length === 50, 'Fact list bounded at 50');
  for (const bad of [locked.text.replace(locked.locks[0].token, ''), locked.text + locked.locks[0].token, locked.text.replace(locked.locks[0].token, '[[HZ_LOCK_changed]]')]) {
    let failed = false;
    try { restoreFacts(bad, locked); } catch (e) { failed = e.message === 'factsLost'; }
    ok(failed, 'Reject missing, duplicated, or mutated locks');
  }
  ok(restoreFacts('ordinary output', protectFacts('ordinary draft', [])) === 'ordinary output', 'No-fact rewrites keep existing behavior');
  ok(buildPrompt(cfg, locked.text).startsWith(INSTR + '\n\n'), 'Canonical model instructions remain unchanged');
}

// 10. Create fact:真模型的写法(括号层数变了)也认;流式显示;自动重试和作废
{
  const p = protectFacts('Charlotte grows from 10,000 to 20,000 sq ft. Call Nadia.', ['20,000 sq ft', 'Nadia']);
  ok(restoreFacts('Charlotte: [HZ_LOCK_0] SF. Ping HZ_LOCK_1 today.', p) === 'Charlotte: 20,000 sq ft SF. Ping Nadia today.', '括号少一层、没有括号也换回原文,不吃掉前面的空格');
  let threw = false;
  try { restoreFacts('[HZ_LOCK_0] and [[HZ_LOCK_0]], [[HZ_LOCK_1]]', p); } catch (e) { threw = e.message === 'factsLost'; }
  ok(threw, '同一个占位符出现两次(写法不同也算)判失败');
  ok(streamView('to [[HZ_LOCK_0]] and [[HZ_LO', p) === 'to 20,000 sq ft and ' && streamView('ping HZ_', p) === 'ping ', '流式途中:写完的占位符换回原文,写到一半的先不显示');
  const good = { raw: 'Grows to [[HZ_LOCK_0]]. Call [[HZ_LOCK_1]].', final: { stop_type: 'eos' } };
  const lost = { raw: 'Grows a lot. Call [[HZ_LOCK_1]].', final: { stop_type: 'eos' } };
  const r1 = await withFactGuard(p, async () => good);
  ok(r1.ok && r1.tries === 1 && r1.text === 'Grows to 20,000 sq ft. Call Nadia.', '首发通过:只跑 1 次');
  const seen = [];
  const r2 = await withFactGuard(p, async (k) => (k === 1 ? lost : good), FACT_TRIES, (n) => seen.push(n));
  ok(r2.ok && r2.tries === 2 && seen.join() === '2', '第 1 次丢了占位符 → 自动重写,第 2 次通过');
  const r3 = await withFactGuard(p, async () => lost);
  ok(!r3.ok && r3.tries === FACT_TRIES && r3.text === '', `${FACT_TRIES} 次都丢 → 作废,不交出丢了原文的结果`);
  const r4 = await withFactGuard(p, async (k) => (k === 1 ? { ...good, final: { stop_type: 'limit' } } : good));
  ok(r4.ok && r4.tries === 2, '被截断(stop_type=limit)也算没保住,重写');
  const r5 = await withFactGuard(protectFacts('no facts here', []), async () => ({ raw: 'anything', final: null }));
  ok(r5.ok && r5.tries === 1 && r5.text === 'anything', '没有保护段:原样返回');
  let stopped = false;
  try { await withFactGuard(p, async () => { throw Object.assign(new Error('stop'), { name: 'AbortError' }); }); } catch (e) { stopped = e.name === 'AbortError'; }
  ok(stopped, '用户停止直接抛出,不重试');
}

// 11. Request deadlines cover both headers and JSON bodies; cancellation is distinct.
{
  const originalFetch = globalThis.fetch;
  const stalled = (signal) => new Promise((resolve,reject) => {
    if (signal.aborted) reject(new DOMException('aborted','AbortError'));
    else signal.addEventListener('abort',()=>reject(new DOMException('aborted','AbortError')),{once:true});
  });
  try {
    let calls=0, observed;
    globalThis.fetch=async(path,options)=>{calls++;observed=options.signal;return stalled(options.signal);};
    for (const request of [()=>getJSON('/app/status',{timeout:15}),()=>postJSON('/app/setup',{}, {timeout:15}),()=>countTokens('draft',{timeout:15})]) {
      let name='';try {await request();} catch(e) {name=e.name;}
      ok(name==='TimeoutError' && observed.aborted,'Stalled request expires as TimeoutError, not user cancellation');
    }
    const ctl=new AbortController();
    const pending=countTokens('draft',{signal:ctl.signal,timeout:1000});
    ctl.abort();
    let name='';try {await pending;} catch(e) {name=e.name;}
    ok(name==='AbortError' && observed.aborted,'Stop aborts tokenization immediately');
    const before=calls;
    try {await countTokens('draft',{signal:ctl.signal});} catch(e) {name=e.name;}
    ok(name==='AbortError' && calls===before,'Already-canceled requests do not reach the engine');
    globalThis.fetch=async(path,options)=>{observed=options.signal;return {ok:true,json:()=>stalled(options.signal)};};
    for (const request of [()=>getJSON('/app/status',{timeout:15}),()=>postJSON('/app/setup',{}, {timeout:15})]) {
      name='';try {await request();} catch(e) {name=e.name;}
      ok(name==='TimeoutError','Deadline covers a response body that never finishes');
    }
    const bodyCtl=new AbortController();
    const bodyRequest=postJSON('/app/setup',{}, {signal:bodyCtl.signal,timeout:1000});
    bodyCtl.abort();
    name='';try {await bodyRequest;} catch(e) {name=e.name;}
    ok(name==='AbortError','POST body cancellation is not swallowed as an empty response');
    let sent;
    globalThis.fetch=async(path,options)=>{sent={path,...options};return {ok:true,json:async()=>({tokens:[1,2,3]})};};
    const doneCtl=new AbortController();
    const tokens=await countTokens('draft',{timeout:15,signal:doneCtl.signal});
    doneCtl.abort();
    await new Promise(resolve=>setTimeout(resolve,25));
    ok(tokens===3 && sent.path==='/api/tokenize' && sent.headers['X-Humanizer']==='1' && JSON.parse(sent.body).content==='draft','Successful requests preserve payload and headers');
    ok(!sent.signal.aborted,'Completed requests release both their caller listener and deadline timer');
    globalThis.fetch=async()=>({ok:false,status:409,json:async()=>({error:{message:'no model'}})});
    let message='';try {await postJSON('/app/setup',{});} catch(e) {message=e.message;}
    ok(message==='no model','Structured HTTP errors remain intact');
    globalThis.fetch=async()=>({ok:true,json:async()=>{throw new SyntaxError('empty');}});
    ok(await postJSON('/app/quit')===null,'Empty successful launcher responses still work');
  } finally {globalThis.fetch=originalFetch;}
}

// 12. Long-document planning and durable, section-level commits(PR #5,0.3.4 集成时重写了切分)
{
  const dc = {...cfg, ctx_size:8192, sampling:{temperature:1,top_p:0.95}};
  // 假 tokenizer:汉字 1 个 1 token,其余 4 个字符 1 token(和 approxTokens 的估算故意不一样,逼出「实测再切」)
  const HAN=/[㐀-鿿]/g;
  let calls=0;
  const count = async (s) => { calls++; return (s.match(HAN)||[]).length + Math.ceil(s.replace(HAN,'').length/4); };
  const joined = (parts) => parts.map(p=>p.before+p.draft+p.after).join('');
  const realMax = async (parts, facts=[]) => { let m=0; for (const p of parts) m=Math.max(m, await count(protectFacts(p.draft,facts).text)); return m; };

  // 触发条件:只有输出可能顶到上限(token×2.5>2048)或放不进上下文才分段,普通草稿走原流程
  ok(!needsSections(dc,500,640) && !needsSections(dc,819,960) && needsSections(dc,820,960) && needsSections({...dc,ctx_size:1024},300,900), 'needsSections:≤819 token 不分段,820 起分段,上下文放不下也分段');
  for (const s of fx.samples) {
    const n = await count(pyStrip(s.draft));
    ok(!needsSections(dc, n, n+120), `示例草稿「${s.id||s.draft.slice(0,12)}」(${n} token)走原来的单篇流程`);
  }

  const fact = 'Nadia approved 42 files.';
  const source = '  '+Array.from({length:12}, (_,i)=>`Section ${i}: ` + 'Nadia approved 42 files. The audit took three weeks. '.repeat(14)).join('\r\n\r\n')+'  \n';
  const parts = await planDocument(source,[fact],dc,count);
  ok(parts.length > 1 && joined(parts)===source, '英文:分段拼回去和原稿逐字一致(含 CRLF 空行、首尾空白)');
  ok(parts.every(p=>!p.draft.includes('HZ_LOCK_')), '段里是原文,不是占位符');
  ok(await realMax(parts,[fact]) <= SECTION_MAX, `英文:每段实测 token ≤ ${SECTION_MAX}`);
  ok(parts.slice(0,-1).every(p=>/\n\s*$/.test(p.after)), '英文:段落都不长时,只在空行处分段');
  ok(parts.reduce((n,p)=>n+p.draft.split(fact).length-1,0) === source.split(fact).length-1, '保留文字出现多少次,分段后一处不少(没有被切开)');
  const sizes = await Promise.all(parts.map(p=>count(p.draft)));
  ok(Math.max(...sizes) < 2.2*Math.min(...sizes), `各段大致等长(${Math.min(...sizes)}–${Math.max(...sizes)} token)`);

  // 中文:没有空格的一大段,按句号切;引号跟着句子走
  const zhSent = '我们在三月十二日完成了第一轮测试，“结果比预期好。”随后团队又花了两周整理数据。';
  const zh = zhSent.repeat(60);
  const zp = await planDocument(zh,[],dc,count);
  ok(zp.length > 1 && joined(zp)===zh, `中文无空格长段:拼回去逐字一致(${zp.length} 段)`);
  ok(zp.slice(0,-1).every(p=>/[。！？”]$/.test(p.draft)), '中文:只在句末切(句号后的右引号留在前一段)');
  ok(await realMax(zp) <= SECTION_MAX, '中文:每段实测 token ≤ 上限');
  // 一句话特别长:退到逗号;连逗号都没有:按字硬切,不切开代理对(emoji)
  const longSent = ('这是一个没有句号但是有很多逗号的超长句子，' ).repeat(120) + '结束。';
  const lp = await planDocument(longSent,[],dc,count);
  ok(lp.length > 1 && joined(lp)===longSent && lp.slice(0,-1).every(p=>/，$/.test(p.draft)), '超长中文句子在逗号处切');
  const unicode='你好😀'.repeat(1600);
  const up = await planDocument(unicode,[],dc,count);
  ok(up.length>1 && joined(up)===unicode && up.every(p=>!/[\uD800-\uDBFF]$/.test(p.draft) && !/^[\uDC00-\uDFFF]/.test(p.draft)), '没有任何标点的长串硬切,不切开 emoji 代理对');
  const enWords = 'word '.repeat(4000).trim();
  const ep = await planDocument(enWords,[],dc,count);
  ok(ep.length>1 && joined(ep)===enWords && ep.every(p=>!/^\s|\s$/.test(p.draft)), '只有空格的英文长串在空格处切');

  // 占位符是原子:跨段落的保留文字整段留在一起;保留文字多次出现
  const spanning='One fact\n\ncontinues across paragraphs';
  const spSrc='Intro. '+spanning+' '+'more words here. '.repeat(900);
  const sp=await planDocument(spSrc,[spanning],dc,count);
  ok(joined(sp)===spSrc && sp.some(p=>p.draft.includes(spanning)), '跨段落的保留文字留在同一段');
  // 草稿里本来就有 HZ_LOCK_:前缀会换成 _HZ_LOCK_,照样还原
  const clash = 'Literal [[HZ_LOCK_0]] text. '.repeat(300) + fact;
  const cp = await planDocument(clash,[fact],dc,count);
  ok(joined(cp)===clash, '草稿本身含 HZ_LOCK_ 时也逐字还原');
  // 保留文字在提示词里只占一个占位符:一大段保留文字也不会把段撑大,整段留在一节里
  const bigFact = 'Clause '.repeat(900).trim()+'.';
  const bp = await planDocument('Lead in. '+bigFact+' Tail sentence.',[bigFact],dc,count);
  ok(bp.length===1 && bp[0].draft.includes(bigFact), '超长保留文字不切开(模型只看到占位符)');
  let rejected=false;try { await planDocument('Some text. '.repeat(50),[],{...dc,ctx_size:200},count); } catch(e) {rejected=e.message==='documentContext';}
  ok(rejected,'上下文小到一段都放不下:报 documentContext,不交出放不下的段');
  let empty=false;try { await planDocument(' \n\n ',[],dc,count); } catch(e) {empty=e.message==='documentEmpty';}
  ok(empty,'只有空白:报错而不是生成空段');
  // 估算偏小(符号多的文本):按实测比例再切,每段仍不超上限
  const symCount = async (s) => Math.ceil(s.length/1.5);
  const sym = '{"k":[1,2,3],"v":"x"} '.repeat(800);
  const syp = await planDocument(sym,[],dc,symCount);
  ok(joined(syp)===sym && (await Promise.all(syp.map(p=>symCount(p.draft)))).every(n=>n<=SECTION_MAX), '估算偏小时按实测 token 再切');
  calls=0; await planDocument(source,[fact],dc,count);
  ok(calls <= parts.length*3+4, `数 token 的次数和段数同量级(${calls} 次 / ${parts.length} 段)`);
  // 停止:规划途中取消,抛 AbortError
  const actl=new AbortController(); actl.abort();
  let pa='';try { await planDocument(source,[],dc,count,actl.signal); } catch(e) {pa=e.name;}
  ok(pa==='AbortError','规划途中停止抛 AbortError');

  // 断点
  const sig=documentSignature(dc,'q8');
  const job=newDocumentJob(source,[fact],sig,parts);
  let n=0, saved=null;
  try {await runDocumentSections(job,async(s)=>{n++;if(n===2)throw new Error('connection failed');return s;},{onCommit:j=>{saved=JSON.stringify(j);}});} catch {}
  ok(job.outputs.length===1 && !!saved, '出错时只记下已完成的段');
  const loaded=readDocumentJob(saved);
  ok(!!loaded && jobMatches(loaded,source,[fact],sig) && !jobMatches(loaded,source+'changed',[fact],sig) && !jobMatches(loaded,source,[],sig) && !jobMatches(loaded,source,[fact],documentSignature(dc,'q4')) && !jobMatches(loaded,source,[fact],documentSignature({...dc,sampling:{temperature:0.7}},'q8')), '断点只认同一篇草稿 + 同样的保留文字 + 同样的模型和采样');
  const before=n;
  const seen=[];
  await runDocumentSections(loaded,async(s,f,sig2,i)=>{n++;seen.push(i);return s;});
  ok(n-before===parts.length-1 && seen[0]===1 && combinedDocument(loaded)===source && jobDone(loaded),'续跑从没完成的那段开始,已完成的段不再送进引擎,拼回全文');
  const bad=JSON.parse(saved);bad.outputs[0]='fact missing';
  const bad2=JSON.parse(saved);bad2.parts[0].draft='tampered';
  ok(readDocumentJob(JSON.stringify(bad))===null && readDocumentJob(JSON.stringify(bad2))===null && readDocumentJob('{broken')===null && readDocumentJob('null')===null && readDocumentJob(null)===null,'损坏的断点、丢了保留文字的断点一律不认');
  const canceled=newDocumentJob(source,[fact],sig,parts), ctl=new AbortController();
  try {await runDocumentSections(canceled,async(s)=>{ctl.abort();return s;},{signal:ctl.signal});} catch {}
  ok(canceled.outputs.length===0,'停止时正在写的那段不记进断点');
  const lost=newDocumentJob(source,[fact],sig,parts);
  let lostErr='';try {await runDocumentSections(lost,async()=> 'No protected phrase');} catch(e) {lostErr=e.message;}
  ok(lost.outputs.length===0 && lostErr==='factsLost','某段丢了保留文字:不采用、报 factsLost');
  const blank=newDocumentJob(source,[fact],sig,parts);
  let blankErr='';try {await runDocumentSections(blank,async()=> '   ');} catch(e) {blankErr=e.message;}
  ok(blank.outputs.length===0 && blankErr==='documentIncomplete','某段改写为空:不采用');
  // 每段的保留文字只看这一段里有的(protectFacts 按段做,和单篇一样)
  const pd0 = protectFacts(parts[0].draft,[fact, 'not in this section']);
  ok(pd0.facts.length===1 && pd0.text.includes('[[HZ_LOCK_0]]') && !pd0.text.includes(fact), '每段只保护本段里出现的保留文字');
}

// 13. 历史记录:删到空时写回空数组;非空记录超配额时仍丢掉老的一半
{
  const desc = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  const values = new Map();
  let maxHistory = Infinity;
  const storage = {
    getItem(k) { return values.has(k) ? values.get(k) : null; },
    setItem(k, v) {
      if (k === 'hz.history.v1' && JSON.parse(v).length > maxHistory) throw new Error('QuotaExceededError');
      values.set(k, String(v));
    },
    removeItem(k) { values.delete(k); },
  };
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: storage });
  try {
    addHistory({ id: 'only' });
    removeHistory('only');
    ok(loadHistory().length === 0 && values.get('hz.history.v1') === '[]', '删除唯一一条历史后写回空数组');

    clearHistory();
    addHistory({ id: 'first' });
    addHistory({ id: 'second' });
    removeHistory('second');
    removeHistory('first');
    ok(loadHistory().length === 0 && values.get('hz.history.v1') === '[]', '多条历史逐条删除时最后一条也能删掉');

    clearHistory();
    for (const id of ['first', 'second', 'third', 'fourth']) addHistory({ id });
    maxHistory = 2;
    addHistory({ id: 'fifth' });
    ok(loadHistory().map((x) => x.id).join() === 'fifth,fourth', '历史超配额时保留较新的一半');
  } finally {
    if (desc) Object.defineProperty(globalThis, 'localStorage', desc); else delete globalThis.localStorage;
  }
}

// localStorage 不可用(无痕 / 被禁用 / 抛异常):存取都不崩,断点读出来是 null
{
  const desc = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, get() { throw new Error('SecurityError'); } });
  try {
    let threw=false, job;
    try {
      store.set('document', '{"x":1}');
      job = readDocumentJob(store.get('document', 'null'));
      addHistory({ id:'x', out:'y' }); loadHistory(); clearHistory();
    } catch { threw=true; }
    ok(!threw && job===null, 'localStorage 不可用时存取不抛错,断点为空');
  } finally {
    if (desc) Object.defineProperty(globalThis, 'localStorage', desc); else delete globalThis.localStorage;
  }
}

// 14. 外部 OpenAI 兼容服务:streamChat 把 chat.completion.chunk 换成 /completion 那样的事件
{
  const { streamChat } = await import('../web/js/api.js');
  const originalFetch = globalThis.fetch;
  const sse = (lines, { ok = true, status = 200, json } = {}) => async (path, opts) => {
    sse.last = { path, opts };
    if (!ok) return { ok: false, status, json: async () => json };
    const enc = new TextEncoder();
    const body = new ReadableStream({ start(c) { for (const l of lines) c.enqueue(enc.encode(l)); c.close(); } });
    return { ok: true, status: 200, body };
  };
  const collect = async (gen) => { const evs = []; for await (const e of gen) evs.push(e); return evs; };
  const chunk = (content, fin = null) => `data: ${JSON.stringify({ object: 'chat.completion.chunk', choices: [{ index: 0, delta: content === null ? {} : { content }, finish_reason: fin }] })}\n\n`;
  try {
    globalThis.fetch = sse([': keep-alive\n\n', chunk(''), chunk('Hel'), 'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}\r\n\r\n', chunk(null, 'stop'), 'data: [DONE]\n\n']);
    let evs = await collect(streamChat({ messages: [{ role: 'user', content: 'x' }] }));
    ok(evs.map((e) => e.content).join('') === 'Hello' && evs.at(-1).stop === true && evs.at(-1).stop_type === 'eos' && evs.at(-1).tokens_predicted === 2, 'streamChat:内容拼对、finish_reason=stop → eos');
    ok(sse.last.path === '/api/v1/chat/completions' && sse.last.opts.headers['X-Humanizer'] === '1' && JSON.parse(sse.last.opts.body).messages.length === 1, 'streamChat:走启动器反代、带 X-Humanizer 头、请求体原样发出');
    globalThis.fetch = sse([chunk('a'), chunk('b', 'length')]);
    evs = await collect(streamChat({}));
    ok(evs.at(-1).stop_type === 'limit', 'streamChat:finish_reason=length → limit(截断,不采用)');
    globalThis.fetch = sse([chunk('a'), 'data: [DONE]\n\n']);
    evs = await collect(streamChat({}));
    ok(evs.at(-1).stop === true && evs.at(-1).stop_type === 'eos', 'streamChat:只有 [DONE] 没有 finish_reason 也算正常结束');
    globalThis.fetch = sse([chunk('a'), chunk('b')]);
    evs = await collect(streamChat({}));
    ok(!evs.some((e) => e.stop), 'streamChat:半路断流(没有 [DONE] 也没有 finish_reason)不给结束事件');
    globalThis.fetch = sse([], { ok: false, status: 502, json: { error: { code: 502, type: 'external_unreachable', message: '连不上外部服务' } } });
    let err; try { await collect(streamChat({})); } catch (e) { err = e; }
    ok(err?.external && err.status === 502 && err.type === 'external_unreachable', 'streamChat:连不上 → 带 status/type 的错误');
    globalThis.fetch = sse([], { ok: false, status: 404, json: { error: { message: "model 'x' not found" } } });
    err = null; try { await collect(streamChat({})); } catch (e) { err = e; }
    ok(err?.status === 404 && /not found/.test(err.message), 'streamChat:模型名不对 → 404 + 服务端原话');
    globalThis.fetch = sse(['data: {"error":{"message":"boom"}}\n\n']);
    err = null; try { await collect(streamChat({})); } catch (e) { err = e; }
    ok(err?.external && err.message === 'boom', 'streamChat:流里的 error 事件抛出');
  } finally { globalThis.fetch = originalFetch; }

  globalThis.document = { documentElement: { getAttribute: () => 'en' } };
  const { pickModel, describeTest } = await import('../web/js/external.js');
  ok(pickModel(['a', 'humanizer-12b-q8_0', 'b']) === 'humanizer-12b-q8_0' && pickModel(['only']) === 'only' && pickModel(['a', 'b']) === '' && pickModel([]) === '', 'pickModel:优先名字带 humanizer 的,只有一个就用它,猜不出就留空');
  const d1 = describeTest({ ok: false, code: 'auth' }), d2 = describeTest({ ok: false, code: 'unreachable', url: 'http://localhost:1234/v1' }), d3 = describeTest({ ok: true, models: ['m'] });
  ok(d1.kind === 'err' && /API key/.test(d1.text) && /localhost:1234/.test(d2.text) && d3.kind === 'ok', 'describeTest:key 错 / 连不上 / 连上了 各有一句说明');
}

// 15. 存储位置:默认目录、尾斜杠、是否自定义过
{
  const win = { data_dir: 'C:\\Users\\B\\AppData\\Local\\Humanizer', default_data_dir: 'C:\\Users\\B\\AppData\\Local\\Humanizer', model_dir: 'C:\\Users\\B\\AppData\\Local\\Humanizer\\models' };
  const mac = { data_dir: '/Users/b/Library/Application Support/Humanizer', default_data_dir: '/Users/b/Library/Application Support/Humanizer', model_dir: '/Users/b/Library/Application Support/Humanizer/models' };
  ok(modelDirDefault(win) === 'C:\\Users\\B\\AppData\\Local\\Humanizer\\models', 'Windows 默认模型目录');
  ok(modelDirDefault(mac) === '/Users/b/Library/Application Support/Humanizer/models', 'macOS 默认模型目录');
  ok(!customPaths(win) && !customPaths(mac), '没改过时不算自定义');
  ok(customPaths({ ...win, model_dir: 'D:\\models' }), '改了模型目录');
  ok(customPaths({ ...win, data_dir: 'D:\\Humanizer' }), '改了运行目录');
  // 运行目录被启动参数钉死时,那个"不一样"不算用户改的
  ok(!customPaths({ ...win, data_dir_locked: true }), '钉死的运行目录不算自定义');
  ok(modelDirDefault({ ...win, data_dir_locked: true }) === 'C:\\Users\\B\\AppData\\Local\\Humanizer\\models', '钉死时模型目录以它为基准');
  // 尾斜杠、大小写不算改动
  ok(!customPaths({ ...win, model_dir: 'c:\\users\\b\\appdata\\local\\humanizer\\models\\' }), '尾斜杠和大小写不算改动');
  ok(!customPaths({ ...mac, model_dir: '/Users/b/Library/Application Support/Humanizer/models/' }), 'macOS 尾斜杠不算改动');
  ok(sameDir('D:\\a', 'D:\\a\\') && sameDir('/x/y/', '/x/y') && !sameDir('D:\\a', 'D:\\b') && !sameDir('', 'D:\\a'),
    '目录比较忽略尾斜杠、认得出不同目录');
  // 摘要上那一句:三种状态
  ok(pathsHintKey(win) === 'paths.hintDefault', '默认 → 默认在系统盘');
  ok(pathsHintKey({ ...win, data_dir_locked: true }) === 'paths.hintLocked', '被启动参数钉死 → 由启动参数指定');
  ok(pathsHintKey({ ...win, data_dir_locked: true, model_dir: 'D:\\models' }) === 'paths.hintCustom', '钉死但改了模型目录 → 仍算自定义');
  ok(pathsHintKey({ ...win, model_dir: 'D:\\models' }) === 'paths.hintCustom', '改了模型目录 → 自定义');
}

// 16. 文案:中英两份键一一对应,页面上用到的键都有(0.3.4 合并时英文少过 menu.paths / menu.changePaths)
{
  const src = readFileSync(new URL('../web/js/i18n.js', import.meta.url), 'utf8');
  const zhAt = src.indexOf('\n  zh: {'), enAt = src.indexOf('\n  en: {');
  const keys = (part) => new Set([...part.matchAll(/^\s+'([\w.]+)':/gm)].map((m) => m[1]));
  const zh = keys(src.slice(zhAt, enAt)), en = keys(src.slice(enAt));
  const onlyZh = [...zh].filter((k) => !en.has(k)), onlyEn = [...en].filter((k) => !zh.has(k));
  ok(zh.size > 100 && !onlyZh.length && !onlyEn.length, `中英文案键一致(${zh.size} 个)${onlyZh.length ? ' 缺英文:' + onlyZh.join(',') : ''}${onlyEn.length ? ' 缺中文:' + onlyEn.join(',') : ''}`);
  const html = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  const used = [...html.matchAll(/data-i18n="([\w.]+)"/g)].map((m) => m[1])
    .concat([...html.matchAll(/data-i18n-attr="([^"]+)"/g)].flatMap((m) => m[1].split(';').map((x) => x.split(':')[1])));
  const missing = [...new Set(used)].filter((k) => !zh.has(k) || !en.has(k));
  ok(!missing.length, `index.html 用到的文案键都有中英两份${missing.length ? ':缺 ' + missing.join(',') : ''}`);
}

if (fails) { console.error(`\n${fails} 项失败`); process.exit(1); }
console.log('\n全部通过');
