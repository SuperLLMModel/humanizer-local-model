---
title: humanizer
emoji: ✍️
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.29.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Rewrite AI drafts to read like a person; facts kept. 12B
models:
  - jialinyyzz/humanizer
---

# humanizer: demo

A 12B model that rewrites AI-written drafts (emails, essays, reports, forum posts; English and Chinese)
so they read like a person wrote them. It is trained to keep every number, unit, date, name and quote,
and to add nothing. No AI detector was used anywhere in training.

This Space runs the bf16 weights from [`jialinyyzz/humanizer`](https://huggingface.co/jialinyyzz/humanizer)
on ZeroGPU, up to about 2,100 English words or 3,600 Chinese characters per run: anything over 700 words
or 1,200 characters is split at paragraph breaks and rewritten part by part, and if the GPU stops partway
the finished text stays on screen with a note. The `/humanize` API still takes one part (700 words) per call. Real
before/after pairs from the held-out evaluation set are on the page and need no GPU.

As in the desktop app, you can import the text of a `.docx` or PDF (up to 10 MB), and select any passage in the
draft and click **Create fact** to keep it word for word: the passage is swapped for a placeholder the model copies
through, then put back, and a part that loses it is rewritten (up to 3 tries) instead of being shown changed.
An English draft that comes back in Chinese is sampled again automatically.

**Run it on your own computer:** the desktop app for macOS (Apple silicon) and Windows is on
[GitHub Releases](https://github.com/sgaofen/humanizer-local-model/releases/latest); the command-line tool
`hz` and llama.cpp, MLX, transformers and vLLM steps are in
[USAGE.md](https://github.com/sgaofen/humanizer-local-model/blob/main/docs/USAGE.md)
([中文](https://github.com/sgaofen/humanizer-local-model/blob/main/docs/USAGE.zh.md)) and
[AGENTS.md](https://github.com/sgaofen/humanizer-local-model/blob/main/AGENTS.md).

**Measured, not promised:** 95% judged human. On our 210 English evaluation drafts, Originality.ai
(AI Allowance 0%, its strictest setting; bf16 weights, 2026-10-02) flagged 11 as AI (the previous
release: 26). On the Q8_0 file you download, a strict LLM judge found no factual problem in 376 of 420
English rewrites, and more than 9 in 10 of the fixes it did list are a single word or phrase. Chinese
is still catching up (no factual problem in 149 of 204). Still, read the result before you send it,
especially numbers, dates, names (and the direction of every claim).

**API:** `gradio_client` → `Client("jialinyyzz/humanizer").predict(draft, api_name="/humanize")` returns
the rewrite plus copy rate and any numbers missing from it.

**Privacy:** to improve the model, the Space stores submitted text and the rewrites anonymously, in a private
dataset that is not published or shared with third parties. Before anything is saved, emails, phone numbers, links,
card numbers, US Social Security numbers, IP addresses and account or ID numbers are replaced with placeholders such
as `[EMAIL]` or `[PHONE]`. This is automatic pattern matching: names and details written out in words are not caught,
so don't paste anything you would not want stored. Each record holds the draft, the rewrite, the hour (UTC), the
settings, lengths, language, model version and whether it was a re-run; never your IP address, Hugging Face account
or browser details. Imported files are deleted as soon as their text is read. To have your text removed, open a
thread in the [Community tab](https://huggingface.co/spaces/jialinyyzz/humanizer/discussions) with the approximate
date and a short phrase from it. To keep text off our servers entirely, run the model locally (above).

Source of this Space: [`space/`](https://github.com/sgaofen/humanizer-local-model/tree/main/space) in the GitHub repo.

---

把 AI 写的草稿改成读起来像人写的，训练目标是数字、单位、日期、人名、引语原样保留。中英文都行。训练全程没有用任何 AI 检测器。
这里可以在线试一篇，也可以导入 .docx 或 PDF（最大 10 MB，读完即删）；在草稿里选中一段文字点「原样保留」，改写时这段一字不改。想装到自己电脑上，下载 [App（macOS / Windows）](https://github.com/sgaofen/humanizer-local-model/releases/latest)，
或者看 [不用 App 怎么用](https://github.com/sgaofen/humanizer-local-model/blob/main/docs/USAGE.zh.md)。

隐私：为改进模型，我们会匿名保存提交的文字和生成结果。保存前会自动去除邮箱、电话、链接等个人信息（规则匹配，认不出人名），数据仅用于改进模型，存放在私有数据集里，不会公开或分享给第三方；不记录 IP、账号或浏览器信息。想删除自己的文字，请在 [Community 讨论区](https://huggingface.co/spaces/jialinyyzz/humanizer/discussions) 发帖，写明大概日期和文中一小段话。
