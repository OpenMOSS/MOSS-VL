import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

// Voice clips land on the shared FS out-of-band (gitignored; see
// src/samp/sentences/README.md) and are discovered by import.meta.glob at
// BUILD time — but the build watcher only reacts to files already in its
// module graph, so a freshly dropped clip (or a whole new demo's dir) never
// triggered the rebuild that would bundle it. Register the sentences tree
// with the watcher explicitly: any clip add/change/removal now fires a
// rebuild and the glob re-scan does the rest. buildStart re-runs per rebuild,
// so demo dirs that appear later are re-registered on the next trigger.
const sentencesDir = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)), 'src/samp/sentences')
const watchSentenceClips = (): Plugin => ({
  name: 'watch-samp-sentence-clips',
  buildStart() {
    if (!fs.existsSync(sentencesDir)) return
    this.addWatchFile(sentencesDir) // new demo dirs
    for (const entry of fs.readdirSync(sentencesDir, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue
      const dir = path.join(sentencesDir, entry.name)
      this.addWatchFile(dir) // new clips within a demo
      for (const f of fs.readdirSync(dir)) this.addWatchFile(path.join(dir, f))
    }
  },
})

// https://vite.dev/config/
// This dev server is reached through the notebook gateway under a long,
// session-specific path that ends in /proxy/5173/. Three problems to solve:
//   1. Emit RELATIVE asset URLs so they resolve under that prefix instead of
//      hitting the domain root (which 404s). -> base: './'
//   2. Vite 8 blocks unknown Host headers (DNS-rebind guard); the gateway's
//      host must be allow-listed or the proxy request 500s. -> allowedHosts
//   3. The browser can only reach the ONE forwarded port, so the backend rides
//      the same origin: /api (HTTP + WebSocket) proxies to :8000. The client
//      resolves API URLs against document.baseURI, so requests arrive as
//      <gateway-prefix>/proxy/5173/api/... and this proxy strips nothing —
//      the path vite sees starts at /api. (backend_overhaul.md §8 decision 1)
const apiProxy = {
  '/api': {
    target: process.env.VITE_BACKEND_ORIGIN || `http://127.0.0.1:${process.env.PORT || '8000'}`,
    changeOrigin: true,
    ws: true, // the session WebSocket rides this same rule
  },
}

export default defineConfig({
  base: './',
  plugins: [react(), watchSentenceClips()],
  build: {
    // `npm run build:watch` (WATCH_POLL=1) turns builds into a rebuild-on-edit
    // loop. POLLING is load-bearing: the watcher runs on the GPU box while the
    // edits land from other machines (login node / VSCode pods) on the shared
    // GPFS — native inotify events never cross FS clients, so an event-based
    // watcher there is blind (verified live: the box stat()ed the new mtime
    // but no rebuild fired). Plain `npm run build` (no env) stays one-shot.
    //
    // vite 8 quirk: the watch RUNTIME still speaks rollup/chokidar — it maps
    // build.watch.chokidar.{usePolling,interval} to rolldown watcher options
    // and OVERWRITES any rolldown-style `watcher` key — while the TYPE is
    // already rolldown's WatcherOptions (no `chokidar` declared). Hence the
    // cast (convertToWatcherOptions in vite's buildEnvironment).
    watch: process.env.WATCH_POLL
      ? ({ chokidar: { usePolling: true, interval: 800 } } as unknown as NonNullable<
          import('vite').BuildOptions['watch']
        >)
      : null,
  },
  server: {
    host: true,        // bind 0.0.0.0 so the gateway (different netns) can reach it
    port: 5173,
    strictPort: true,  // fail loudly instead of drifting to 5174 (breaks the proxy URL)
    allowedHosts: ['.sii.edu.cn'], // un-block the gateway domain + subdomains
    proxy: apiProxy,
    watch: {
      // the python venv + backend data live IN-REPO: letting the dev watcher
      // crawl their ~100k files exhausts the kernel inotify budget and kills
      // the server with ENOSPC before the first page load
      ignored: ['**/.venv/**', '**/data/**', '**/__pycache__/**'],
    },
    hmr: {
      protocol: 'wss', // page is served over https -> HMR socket must be wss
      clientPort: 443, // browser dials back on 443 (the gateway), not 5173
    },
  },
  // `vite preview` is a plain static server (no HMR websocket), so it rides the
  // gateway proxy cleanly. It has its OWN allow-list, separate from `server`.
  // Pair with `npm run build:watch` to rebuild dist/ on every save.
  preview: {
    host: true,
    port: 5173,        // reuse the same /proxy/5173/ gateway URL as dev
    strictPort: true,
    // Exposed publicly through the 花生壳 (phddns) tunnel, whose external
    // hostname is an unpredictable oray 二级域名 (e.g. *.zicp.vip) and later a
    // CNAME'd live.mosi-ai.com. `true` accepts any Host so the DNS-rebind guard
    // doesn't 500 the tunnel; safe here because this origin is deliberately
    // internet-facing (not a private dev box).
    allowedHosts: true,
    proxy: apiProxy,
  },
})
