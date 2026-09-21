// Run after building the frontend. All HTTP/WS traffic is mocked.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
const root = path.resolve(__dirname, '../..');
const dist = process.env.REVIEW_DIST || path.join(root, 'dist');

async function scenario(browser, mode) {
  const context = await browser.newContext({permissions:['camera','microphone']});
  try {
    const page = await context.newPage();
    let created = 0, deleted = 0, frames = 0;
    await page.addInitScript(mode => {
      const NativeContext = window.AudioContext;
      window.AudioContext = class extends NativeContext {
        constructor(...args) {
          super(...args);
          Object.defineProperty(this.audioWorklet, 'addModule', {
            value: () => mode === 'failure'
              ? Promise.reject(new Error('Simulated microphone initialization failure'))
              : new Promise(() => {}),
          });
        }
      };
    }, mode);
    await page.route('https://review.invalid/**', async route => {
      const url = new URL(route.request().url());
      if(url.pathname === '/api/sessions' && route.request().method() === 'POST') {
        created++;
        return route.fulfill({json:{session_id:'test-only',ws_url:'/api/session/test-only/ws'}});
      }
      if(url.pathname.startsWith('/api/sessions/') && route.request().method() === 'DELETE') {
        deleted++;
        return route.fulfill({json:{ok:true}});
      }
      if(url.pathname === '/api/chat/stream') return route.fulfill({contentType:'text/event-stream',
        body:'data: '+JSON.stringify({type:'generation_error',message:'offline chat not supported by sglang_omni deploy'})+'\n\n'});
      if(url.pathname === '/api/status') return route.fulfill({json:{vlm:{loaded:true,capacity:4},voice:{}}});
      if(url.pathname.startsWith('/api/')) return route.fulfill({json:{items:[]}});
      const filename = path.join(dist, url.pathname === '/' ? 'index.html' : url.pathname);
      const contentType = filename.endsWith('.js') ? 'text/javascript' : filename.endsWith('.css') ? 'text/css'
        : filename.endsWith('.html') ? 'text/html' : 'application/octet-stream';
      return route.fulfill({body:fs.readFileSync(filename),contentType});
    });
    await page.routeWebSocket('wss://review.invalid/**', ws => {
      ws.onMessage(data => {
        if(typeof data !== 'string' && data[0] === 2) frames++;
        if(typeof data === 'string' && JSON.parse(data).type === 'text.input') {
          ws.send(JSON.stringify({v:1,type:'response.created',seq:2,response_id:'reply'}));
          ws.send(JSON.stringify({v:1,type:'response.text.delta',seq:3,response_id:'reply',delta:'text path works'}));
        }
      });
      ws.send(JSON.stringify({v:1,type:'session.created',seq:1,session_id:'test-only',
        audio_out:{sample_rate:48000,channels:2},audio_in:{sample_rate:16000,channels:1}}));
    });
    await page.goto('https://review.invalid/',{waitUntil:'networkidle'});
    if(mode === 'failure') {
      await page.locator('#chat-input-field').fill('test');
      await page.locator('#chat-input-field').press('Enter');
      await page.waitForFunction(() => document.body.innerText.includes('当前部署未启用离线聊天'));
      assert(!(await page.locator('body').innerText()).includes('请确认服务已启动'));
    }
    await page.getByTitle('切换为视频通话',{exact:true}).click();
    await page.locator('#btn-toggle-video').click();
    await page.getByRole('menuitem',{name:'摄像头',exact:true}).click();
    await page.locator('#btn-connect-stream').click();
    await page.waitForTimeout(mode === 'timeout' ? 11000 : 2000);
    assert.equal(created, 1);
    assert.equal(deleted, 0);
    assert(frames > 0, 'video must not wait for microphone initialization');
    if(mode !== 'cancel') {
      await page.locator('.live-text-field').fill('test',{force:true});
      await page.locator('.live-text-field').press('Enter');
      await page.waitForFunction(() => document.body.innerText.includes('text path works'));
    }
    await page.locator('#btn-connect-stream').click();
    await page.waitForTimeout(700);
    assert.equal(deleted, 1);
    assert.equal(await page.locator('#btn-connect-stream').getAttribute('title'), '连接实时通话');
    console.log(`PASS ${mode}: video/text fallback and session cleanup`);
  } finally {
    await context.close();
  }
}

(async () => {
  const browser = await chromium.launch({headless:true,
    ...(process.env.CHROMIUM_EXECUTABLE ? {executablePath:process.env.CHROMIUM_EXECUTABLE} : {}),
    args:['--no-sandbox','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
  try {
    for(const mode of ['failure','cancel','timeout']) await scenario(browser, mode);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
