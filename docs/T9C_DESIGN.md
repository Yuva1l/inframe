# T9c design: publishing the page as a static site (up to the deploy step)

Items marked
[unverified] were not checked against the real service.

Decisions: the page is a public static site on Cloudflare Pages; commercial use is allowed; the repo gets an open-source license; no
accounts for visitors; the video never leaves the browser.

## 1. Model host

Cloudflare Pages allows 25 MiB per file. The models are 88 to 124 MB, so they need another host.

- **Recommended:** a public Hugging Face model repo `<hf-user>/inframe-models`. It needs a free account and no card. It holds
  `rfdetr_small_v1.onnx`, `dinov2_s_v1.onnx`, `ort-wasm-simd-threaded.asyncify.wasm`, `manifest.json`, `LICENSE-APACHE`, `LICENSE-MIT`, `NOTICE`
  and a README model card saying where each file comes from, its license and the export command. Apache-2.0 requires the LICENSE and NOTICE to
  travel with the weights.
- **Pinning works in two layers.**
  - URL: `VITE_MODEL_BASE=https://huggingface.co/<hf-user>/inframe-models/resolve/<40-hex commit>/`. The commit is read after the upload with
    `git ls-remote`, never `main`. Whether a commit-SHA resolve URL is immutable is [unverified]. Layer 2 makes that not matter.
  - Content: the sha256 of every file is in `manifest.json`, and `models.ts` checks it with `crypto.subtle` before caching. Before T9c the page
    fetched the manifest from the model base, so whoever controls the host controls the digests. T9c moves the manifest into the page: the build
    copies a committed `web/pages/manifest.json` to `pages-dist/models/manifest.json` (same origin). It is fetched through the new
    `VITE_MODEL_MANIFEST` setting, which defaults to base + `manifest.json`, so `inframe serve` does not change. The model host then serves
    bytes only, and a changed or replaced file fails its check and is dropped.
- **Caching stays as it is:** Cache Storage `inframe-models-v1`, keyed by sha256, with `persist()` requested. Each browser downloads once. The HF
  302 has `no-store` and the CDN URL is signed and expires; the page never stores a URL, only bytes under the digest key, so expiry does not
  matter. Model fetches use `credentials: 'omit'` and `referrerPolicy: 'no-referrer'` (no cookies to the host, no page URL sent). No custom
  headers are added, so no CORS preflight is sent.
- **connect-src:** `'self'`, the pinned repo path `https://huggingface.co/<hf-user>/inframe-models/resolve/<commit>/`, `https://*.cdn.hf.co` and
  `https://cas-bridge.xethub.hf.co` (deploy2 decision, narrowed in the deploy2 review: not all of `*.hf.co`). A download is answered with a redirect to a CDN host that depends on the visitor's region and changes over time
  (`us.aws.cdn.hf.co`, `cas-bridge.xethub.hf.co`, others elsewhere), so a fixed list would block some visitors. The one wildcard, `*.cdn.hf.co`, is accepted only
  when `base` is a pinned `https://huggingface.co/OWNER/REPO/resolve/<40-hex commit>/` URL (build: `pagesHost` in `web/vite.config.ts`;
  preflight: `check_host`), and it is the only wildcard either one accepts; `https://huggingface.co`, `https://hf.co`, `https://*.huggingface.co`
  and every other `*.` entry stay refused. Why it is safe: the first request is pinned to the repo and commit, and the page checks every file
  against the sha256 in its own manifest, so a host under `hf.co` can serve bytes but cannot change what the page accepts. The cost: the page may
  connect to any `*.cdn.hf.co` name, which is Hugging Face's CDN. `*.hf.co` is refused because it would also open the Xet upload host
  (`cas-server.xethub.hf.co` [unverified against HF docs]); the CSP is the backstop that footage stays in the browser, and the build and preflight refuse
  any entry that covers that host. A redirect host outside the wildcard is added by its exact name. `--remote` still follows the real redirects and prints every host.
- **Limits:** 3,000 resolver requests per 5 minutes per IP (a visitor uses 3). Free public storage is "best-effort, responsible use". HF terms
  of service not read [unverified].
- **Fallback:** an R2 public bucket on a custom domain. `r2.dev` is documented as rate-limited and for development only, so a domain is needed.
  R2 probably needs a card at checkout (not documented). CORS is set with `wrangler r2 bucket cors set`. Moving to R2 is a config change in
  `web/pages/model-host.json` (base, connect origins) and a rebuild, with no code change.
- **Not recommended:** split chunks on Pages. It fits the limits, but Cloudflare's CDN terms clause on "disproportionate" large files is a risk.
  Use it only with Cloudflare's written OK.

## 2. Pages build (part 1, implemented)

`npm --prefix web run build:pages` = `tsc --noEmit && vite build --mode pages`. In pages mode `web/vite.config.ts`:

- Sets `outDir` to `web/pages-dist` (gitignored), so the committed `src/inframe/web/dist` is never touched.
- Reads `web/pages/model-host.json` = `{base, connect: [origins]}`. `VITE_MODEL_BASE` and `INFRAME_PAGES_CONNECT` (comma separated) override it;
  the tests use them. The build fails when the base is still the placeholder (`OWNER` / `COMMIT`), has no 40-hex commit segment, is not https, or
  its origin is not covered by `connect`; each connect entry must be a plain origin (`https://host` or `https://*.host`, so a value cannot add
  CSP directives). `INFRAME_PAGES_ALLOW_LOCAL=1` lifts the https and commit rules for an `http://127.0.0.1:PORT` base (tests only).
- Defines `import.meta.env.VITE_MODEL_BASE` (the validated base) and `VITE_MODEL_MANIFEST` (`./models/manifest.json`) in the bundle.
- Copies `web/pages/manifest.json` byte for byte to `models/manifest.json`. It is a copy of `$INFRAME_MODELS/onnx/manifest.json`; a test
  compares sha256 and size when that file exists.
- Writes `_headers`: `public/_headers` with only `connect-src` changed to `'self'` plus the configured origins, plus
  `Permissions-Policy: camera=(), microphone=(), geolocation=()` and a `/assets/*` rule with
  `Cache-Control: public, max-age=31536000, immutable`. `public/_headers` stays equal to the `serve.py` CSP
  (`test_pages_headers_file_has_the_same_csp` still holds).
- No `_redirects`: the app is one `index.html` with hash state, so only `/` is ever requested.
- Mode `pages` already returns `{local: null, staticHost: true}` without calling `./api`; a test checks the built assets do not contain
  `api/session`.
- **What the page says:** the My Mac button stays disabled with the existing `STATIC_HOST_HINT`. Under the Download button (shown while files are
  still to download) the page adds: "Your video never leaves this browser. Only the three model files (about 239 MB) are downloaded, once, from
  huggingface.co." The host is the host of the build's model base ("this site" when it is the page's own origin, as under `inframe serve`).
  A link to the repo is shown only if the repo is public (question 5).
- `scan_dist` (tests/test_serve_dist.py) gains `allow_hosts`. For pages-dist it allows the hosts named in the built `_headers` connect-src, and
  the `source` URLs in `models/manifest.json`. Pages limits are checked: every file under 25 MiB and fewer than 20,000 files.

## 3. Privacy check (part 2)

`tests/browser/test_pages_privacy.py`, in the browser suite. `tests/browser/static_hosts.py` starts two `ThreadingHTTPServer`s on 127.0.0.1 with
different ports, so they are two origins:

- A, the page host: serves pages-dist and applies pages-dist/`_headers` (parsed with the same rules as Pages, so the generated CSP is what
  Chrome enforces).
- B, the model host: serves `$INFRAME_MODELS/onnx` under `/resolve/<fake 40-hex>/`. Like HF it answers with a 302 to
  `/cdn/<name>?Expires=..&Signature=..`; both hops carry ACAO, `Accept-Ranges: bytes`, a 206 answer to a Range request, and HEAD. Every request
  is logged as (method, path, query, headers, body length).

pages-dist is built into a temp folder with `VITE_MODEL_BASE` pointing at B, `INFRAME_PAGES_CONNECT=<B origin>` and `INFRAME_PAGES_ALLOW_LOCAL=1`.
Input is a synthetic clip (ffmpeg testsrc2 HEVC, 10 s) carrying a canary: a random 32-hex token in the mp4 title metadata and in a byte run
appended inside a free box. Chrome drives the real UI in a fresh context: open the file, Download models, run "any person", then a tap target on a
synthetic box (so DINOv2 embeds), then export as a download. Assertions:

- (a) The request origins over `context.on('request')` plus both server logs are exactly {A, B}. `blob:` and `data:` requests are ignored.
- (b) Every method is GET, or HEAD on B. No OPTIONS, POST or PUT, and every `post_data` is None.
- (c) On B only manifest names are fetched. On A there is no query string and only files that exist in pages-dist.
- (d) No URL, header or body contains the canary, the file name, or a 64-byte run of the clip. Every URL is under 512 characters.
- (e) No WebSocket opens, no service worker registers, and `navigator.sendBeacon` is spied and never called.
- (f) The models are read from Cache Storage on a second run (no new B requests).

A unit test of the checker (normal suite) feeds it fake logs with a POST, a third origin and the canary in a query string and expects all three
to be reported.

## 4. Deploy steps (part 3: docs/T9C_DEPLOY.md has the exact commands)

Live as a classic Cloudflare Pages project `inframe` at https://inframe.pages.dev, with `pages_build_output_dir` = `./web/pages-dist`, deployed with `wrangler pages deploy`. Workers static assets is not used: its public address contains the account name, which must not appear in a public repo. Pages applies the `_headers` file (up to 100 rules, 2,000 characters a line;
the generated CSP line is about 400); the live response headers confirm it.

- HF (done): account, `hf repo create`, `tools/hf_bundle.py` staging, `hf upload`, pin the commit in `web/pages/model-host.json`.
- Check: `uv run python tools/pages_preflight.py --no-dist --remote` checks sizes, ACAO and the redirect hosts against connect-src.
- Cloudflare: `npm --prefix web run build:pages`, `uv run python tools/pages_preflight.py`, then
  `npx wrangler@4.151.1 pages deploy web/pages-dist --project-name inframe --branch main`.
- Verify: `curl -sI https://inframe.pages.dev/` to see the CSP, then the reference-clip parity try in Chrome and Safari, which the maintainer runs personally.
- wrangler is run through npx at a pinned version and is not added as a dependency.

## 5. License for the public repo (question 1, not decided here)

Options: Apache-2.0, MIT, BSD-3-Clause, or MIT OR Apache-2.0. All allow commercial use, by the maintainer and by others. Apache-2.0 matches the model
weights and includes a patent grant. Whichever is chosen goes in `LICENSE` plus the `license` field of package.json and pyproject (package.json
says UNLICENSED today). The GPL (trackers) and LGPL (music) extras are optional and not redistributed, so README's notes stay as they are.

## No new dependency

T9c part 1 adds no dependency. The model host's files keep their licenses (Apache-2.0, Apache-2.0, MIT); wrangler and `hf` run outside the repo.
