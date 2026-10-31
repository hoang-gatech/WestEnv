'use strict';

// The chat dialog. It uses $ and h() from app.js. The model's answer is untrusted text, so it is
// rendered by building DOM nodes (see markdown()), never as HTML.

const chat = {
  el: $('#chat'),
  log: $('#chat-log'),
  form: $('#chat-form'),
  input: $('#chat-input'),
  send: $('#chat-send'),
  endpoint: $('#chat-endpoint'),
  model: $('#chat-model'),
  context: $('#chat-context'),
  config: null,
  history: [], // finished question/answer pairs sent back as context: [{ role, content }]
  point: window.sitedeskPoint || null, // the point selected on the map (set by app.js)
  useContext: true,
  controller: null, // aborts the request in flight
  busy: false,
};

const fmtSeconds = (ms) => `${(ms / 1000).toFixed(1)} s`;
const fmtTokens = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));
const scrollDown = () => {
  chat.log.scrollTop = chat.log.scrollHeight;
};

// --- rendering the model's text: paragraphs, lists, **bold** and `code`, nothing else ------------------

function inline(text, parent) {
  let last = 0;
  for (const m of text.matchAll(/\*\*[^*]+\*\*|`[^`]+`/g)) {
    parent.append(text.slice(last, m.index));
    parent.append(m[0].startsWith('**') ? h('strong', {}, m[0].slice(2, -2)) : h('code', {}, m[0].slice(1, -1)));
    last = m.index + m[0].length;
  }
  parent.append(text.slice(last));
}

function markdown(source) {
  const out = document.createDocumentFragment();
  let list = null;
  for (const raw of source.split('\n')) {
    const line = raw.trimEnd();
    const item = line.match(/^\s*(?:([-*•])|\d+[.)])\s+(.*)/);
    if (item) {
      const tag = item[1] ? 'ul' : 'ol';
      if (!list || list.localName !== tag) {
        list = h(tag);
        out.append(list);
      }
      const li = h('li');
      inline(item[2], li);
      list.append(li);
      continue;
    }
    list = null;
    if (!line.trim()) continue;
    const heading = line.match(/^#{1,4}\s+(.*)/);
    const p = h('p', heading ? { style: 'font-weight: 650' } : {});
    inline(heading ? heading[1] : line, p);
    out.append(p);
  }
  return out;
}

function addMessage(kind, content) {
  const el = h('div', { class: `msg ${kind}` });
  el.append(kind === 'bot' ? markdown(content) : content);
  chat.log.append(el);
  scrollDown();
  return el;
}

function renderActivity(activity, steps) {
  activity.replaceChildren(
    spinner(steps.length ? '' : 'Thinking…'),
    ...steps.map((s) => h('span', { class: 'step' }, s.ms == null ? `${s.name}…` : `${s.name} ${fmtSeconds(s.ms)}${s.ok ? ' ✓' : ' ✗'}`)),
  );
  scrollDown();
}

function metaLine(steps, done) {
  const groups = new Map(); // one entry per tool: how many calls, the slowest, whether any failed
  for (const s of steps) {
    const g = groups.get(s.name) || { n: 0, ms: 0, failed: false };
    groups.set(s.name, { n: g.n + 1, ms: Math.max(g.ms, s.ms || 0), failed: g.failed || !s.ok });
  }
  const tools = groups.size
    ? [...groups].map(([name, g]) => `${name}${g.n > 1 ? ` ×${g.n}` : ''} ${fmtSeconds(g.ms)}${g.failed ? ' (failed)' : ''}`).join(', ')
    : 'no tool calls';
  return h('div', { class: 'meta' }, `${tools} · ${fmtTokens(done.input_tokens + done.output_tokens)} tokens · ${Number(done.seconds).toFixed(1)} s`);
}

// --- talking to the server ----------------------------------------------------------------------------

async function* serverEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = block.split('\n').find((line) => line.startsWith('data: '));
      if (data) yield JSON.parse(data.slice(6));
    }
  }
}

async function failure(response) {
  const body = await response.json().catch(() => ({}));
  const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join('; ') : body.detail;
  return detail || `HTTP ${response.status}`;
}

function setBusy(busy) {
  chat.busy = busy;
  chat.send.textContent = busy ? 'Stop' : 'Send';
  chat.log.setAttribute('aria-busy', String(busy));
}

async function ask(question) {
  setBusy(true);
  addMessage('user', question);
  const activity = h('div', { class: 'activity' });
  const steps = []; // this question's tool calls, summarised under the answer
  renderActivity(activity, steps);
  chat.log.append(activity);
  scrollDown();

  chat.controller = new AbortController();
  let answer = null;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: chat.controller.signal,
      body: JSON.stringify({
        message: question,
        endpoint: chat.endpoint.value || undefined,
        model: chat.model.value || undefined,
        history: chat.history.slice(-16),
        point: chat.useContext && chat.point ? chat.point : undefined,
      }),
    });
    if (!res.ok) throw new Error(await failure(res));
    for await (const ev of serverEvents(res)) {
      if (ev.type === 'tool_call') {
        steps.push({ id: ev.id, name: ev.name, ms: null, ok: null });
        renderActivity(activity, steps);
      } else if (ev.type === 'tool_result') {
        const step = steps.find((s) => s.id === ev.id);
        if (step) Object.assign(step, { ms: ev.ms, ok: ev.ok });
        renderActivity(activity, steps);
      } else if (ev.type === 'answer') {
        activity.remove();
        answer = addMessage('bot', ev.text);
        chat.history.push({ role: 'user', content: question }, { role: 'assistant', content: ev.text });
      } else if (ev.type === 'done') {
        if (answer) answer.append(metaLine(steps, ev));
      } else if (ev.type === 'error') {
        activity.remove();
        addMessage('err', ev.message);
      }
    }
  } catch (err) {
    activity.remove();
    if (err.name === 'AbortError') chat.log.append(h('div', { class: 'note' }, 'Stopped.'));
    else addMessage('err', err.message);
  } finally {
    activity.remove();
    chat.controller = null;
    setBusy(false);
    scrollDown();
  }
}

// --- the controls ---------------------------------------------------------------------------------------

function showContext() {
  const on = chat.point && chat.useContext;
  chat.context.hidden = !on;
  if (on) $('#chat-context-text').textContent = `Point ${chat.point.lat}, ${chat.point.lon}`;
}

function preference() {
  try {
    return JSON.parse(localStorage.getItem('sitedesk.chat') || '{}');
  } catch {
    return {};
  }
}

function remember() {
  try {
    localStorage.setItem('sitedesk.chat', JSON.stringify({ endpoint: chat.endpoint.value, model: chat.model.value }));
  } catch {
    /* private mode or storage disabled: the choice just is not remembered */
  }
}

function fillModels(preferred) {
  const endpoint = chat.config.endpoints.find((e) => e.name === chat.endpoint.value);
  chat.model.replaceChildren(...(endpoint ? endpoint.models : []).map((m) => h('option', { value: m }, m)));
  if (endpoint && endpoint.models.includes(preferred)) chat.model.value = preferred;
}

function fillEndpoints() {
  const saved = preference();
  const usable = (name) => chat.config.endpoints.some((e) => e.name === name && e.ready);
  chat.endpoint.replaceChildren(
    ...chat.config.endpoints.map((e) => h('option', { value: e.name, disabled: !e.ready }, e.ready ? e.name : `${e.name} (no key)`)),
  );
  chat.endpoint.value = usable(saved.endpoint) ? saved.endpoint : chat.config.default || '';
  fillModels(saved.model);
}

function welcome() {
  addMessage('bot', 'Ask about a site: soil, terrain, weather, nearby water or dioxin hotspots. Click the map first and I will use that point.');
}

async function start() {
  try {
    const res = await fetch('/api/chat/config');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    chat.config = await res.json();
  } catch (err) {
    chat.config = { enabled: false, problem: `The server has no chat support (${err.message}).`, endpoints: [] };
  }
  fillEndpoints();
  showContext();
  if (chat.config.enabled) {
    welcome();
    return;
  }
  for (const el of [chat.input, chat.send, chat.endpoint, chat.model]) el.disabled = true;
  addMessage(
    'hint',
    h('div', {}, h('strong', {}, "Chat isn't set up yet."), h('p', {}, chat.config.problem), h('p', {}, 'Then reload this page. Setup is described in the README, under Chat.')),
  );
}

chat.form.addEventListener('submit', (ev) => {
  ev.preventDefault();
  if (chat.busy) {
    if (ev.submitter === chat.send && chat.controller) chat.controller.abort(); // only the Stop button stops
    return;
  }
  const question = chat.input.value.trim();
  if (!question) return;
  chat.input.value = '';
  ask(question);
});

chat.input.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
    ev.preventDefault();
    chat.form.requestSubmit();
  }
});

chat.endpoint.addEventListener('change', () => {
  fillModels();
  remember();
});
chat.model.addEventListener('change', remember);

$('#chat-clear').addEventListener('click', () => {
  if (chat.controller) chat.controller.abort();
  chat.history = [];
  chat.log.replaceChildren();
  if (chat.config && chat.config.enabled) welcome();
});

$('#chat-context-clear').addEventListener('click', () => {
  chat.useContext = false;
  showContext();
});

document.addEventListener('sitedesk:point', (ev) => {
  chat.point = ev.detail;
  chat.useContext = true;
  showContext();
});

// On narrow screens the dock and the profile panel cannot both fit: opening one closes the other.
const compact = window.matchMedia('(max-width: 1080px)');
chat.el.addEventListener('toggle', () => {
  if (!chat.el.open) return;
  if (compact.matches) $('#panel').open = false;
  chat.input.focus();
});
$('#panel').addEventListener('toggle', () => {
  if ($('#panel').open && compact.matches) chat.el.open = false;
});

start();
