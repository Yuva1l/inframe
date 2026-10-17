# T9c: deploying the page (steps for the maintainer)

The page is a classic Cloudflare Pages project named `inframe`, live at https://inframe.pages.dev. The Pages config is `name` inframe and `pages_build_output_dir` = `./web/pages-dist` (kept in a `wrangler.jsonc` that is not part of this repository). It is not a Workers static-assets site: a Worker's address is
`<name>.<account>.workers.dev`, which contains the account name, and that must not appear in a public repo. Use Pages only, never `wrangler deploy`.
Design and reasons: `docs/T9C_DESIGN.md`. If a wrangler flag prints something different from this document, trust `npx wrangler@4.151.1 pages deploy --help`.

## Already done

- The public Hugging Face model repo `itsUV/inframe-models`, the upload from `tools/hf_bundle.py` (rfdetr_small_v1.onnx, dinov2_s_v1.onnx,
  ort-wasm-simd-threaded.asyncify.wasm, manifest.json, LICENSE-APACHE, LICENSE-MIT, NOTICE, README.md model card).
- The commit is pinned in `web/pages/model-host.json`: `base` is `https://huggingface.co/itsUV/inframe-models/resolve/2344e3ea95c823f423763a8c1e1ebd920a662926/`.
  Never use `main`. To publish new model files: stage with `uv run python tools/hf_bundle.py --out ~/inframe-hf`, `hf upload itsUV/inframe-models ~/inframe-hf .`,
  `git ls-remote https://huggingface.co/itsUV/inframe-models refs/heads/main` for the new 40-hex commit, and put it in both entries of `model-host.json`.
- `connect` in `model-host.json` is the pinned repo path, `https://*.cdn.hf.co` and the exact host `https://cas-bridge.xethub.hf.co`. Hugging Face answers a download with a
  redirect to a CDN host that depends on the visitor's region and changes over time (`us.aws.cdn.hf.co`, `cas-bridge.xethub.hf.co`, ...). The wildcard covers
  the download CDN only; `*.hf.co` is not used because it would also open the Xet upload host (`cas-server.xethub.hf.co` [unverified against HF docs]) and the
  CSP is what keeps footage in the browser. A new redirect host outside `*.cdn.hf.co` is added by its exact name. The build and the preflight accept the
  wildcard only when `base` is a pinned `https://huggingface.co/OWNER/REPO/resolve/<40-hex commit>/` URL, refuse every other wildcard and `https://huggingface.co`
  or `https://hf.co` on their own, and refuse any entry that covers the upload host. Files are still checked against the sha256 in the page's own manifest.

## 1. Check the model host (maintainer only: contacts Hugging Face)

```
uv run python tools/pages_preflight.py --no-dist --remote --origin https://inframe.pages.dev
uv run python tools/pages_preflight.py --no-dist --remote --digest --origin https://inframe.pages.dev    # optional: downloads 239 MB once, checks the sha256
```
Every `PROBLEM` line must be gone: a hop without `Access-Control-Allow-Origin`, a size that differs from the manifest, or a redirect host that `connect`
does not cover (any `*.cdn.hf.co` host and `cas-bridge.xethub.hf.co` are covered). The default `--origin` is the live address, https://inframe.pages.dev. A "Range answered 200" note is fine.

## 2. Cloudflare (the site)

Wrangler is not a dependency of the repo. Pin it and use it through `npx`:
```
WR=wrangler@4.151.1
```

### Done once (already done)

- **Account.** A free Cloudflare account (https://dash.cloudflare.com/sign-up). Pages allows 25 MiB per file and 20,000 files per deployment;
  `tools/pages_preflight.py` checks both.
- **Login.** `npx $WR login` (opens the browser for the OAuth grant), then `npx $WR whoami`.
- **Project.** `npx $WR pages project create inframe --production-branch main --force`.

### Each release

1. **Build the page.**
   ```
   npm --prefix web run build:pages
   ```
   It fails on the placeholder, a base with no 40-hex commit, a base `connect` does not cover, or a wildcard other than `https://*.cdn.hf.co` for a pinned
   Hugging Face base. Output: `web/pages-dist` (gitignored; the committed `src/inframe/web/dist` is not touched).
2. **Preflight, local.**
   ```
   uv run python tools/pages_preflight.py
   ```
   Checks that `web/pages-dist` exists, every file is under 25 MiB, there are fewer than 20,000 files, the `_headers` connect-src is `'self'` plus exactly the
   `connect` list of `model-host.json`, the base is https with a 40-hex commit, the built manifest is `web/pages/manifest.json` byte for byte, and the built
   script carries the base. Do not deploy until it prints `local: OK`.
3. **Deploy.**
   ```
   npx $WR pages deploy web/pages-dist --project-name inframe --branch main
   ```
   This publishes the page on the internet at https://inframe.pages.dev. Only the contents of `web/pages-dist` are uploaded: the page, its scripts, `_headers`
   and `models/manifest.json`. No model file and no video is in it.
4. **The curl checks.**
   ```
   curl -sI https://inframe.pages.dev/ | grep -i -E 'content-security-policy|permissions-policy|referrer-policy|x-frame-options|x-content-type-options'
   curl -s https://inframe.pages.dev/models/manifest.json | shasum -a 256 ; shasum -a 256 web/pages/manifest.json
   uv run python tools/pages_preflight.py --no-dist --remote --origin https://inframe.pages.dev
   ```
   Expect: the CSP has `connect-src 'self'` plus the repo path, `https://*.cdn.hf.co`, `https://cas-bridge.xethub.hf.co` and nothing else new, `frame-ancestors 'none'`, the
   camera/microphone/geolocation Permissions-Policy, `x-frame-options: DENY`, `nosniff` and `no-referrer`; the two sha256 lines are equal; the preflight prints `remote: OK`.
   If the CSP header is missing, the `_headers` file was not applied: see "If something fails".
5. **The reference-clip try (by you, in Chrome and then Safari).** Open the address, open the browser's network panel, choose a video of your own, press Download models, run "any person" with the balanced preset, and export.
   - The only hosts in the network panel are the site, `huggingface.co` and a `*.cdn.hf.co` or `cas-bridge.xethub.hf.co` redirect host; no request carries the video's name or bytes
     (the same thing `tests/browser/test_pages_privacy.py` checks on synthetic media).
   - The clip times match the CLI's within the 0.3 s gate (`tools/browser_regress.py` checks the local page; the live site has not been run) [unverified].
   - A second visit downloads no model again (Cache Storage); the site's storage shows about 239 MB.

## The `_headers` file

Cloudflare Pages reads a plain text file called `_headers` in the build output directory (`web/pages-dist/_headers`) and applies it to the responses. It is not itself
served. On the live site `curl -sI https://inframe.pages.dev/` shows the content-security-policy, `x-frame-options: DENY`, `nosniff` and `no-referrer` from it.

## If something fails

- A model request is blocked by the CSP: the host in the console message is not in `connect` (it should be under `*.cdn.hf.co` or be `cas-bridge.xethub.hf.co`; if it is another host,
  decide whether to add that exact host to `connect` in `model-host.json`), rebuild, preflight, deploy (the three release steps).
- The host answers without ACAO: the page cannot read the file. Switch the host (design section 1, R2 on a custom domain) by editing `model-host.json` only.
- A file's sha256 is wrong in the browser: the host serves other bytes than the manifest says; the page drops the file by design. Check with `--remote --digest`.
- To undo: Cloudflare dashboard, Workers & Pages, the `inframe` Pages project, Deployments, roll back to an earlier deployment, or delete the project [unverified]. The model repo can be made private or deleted from its settings page. Nothing else holds a copy.
