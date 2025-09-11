SHELL := bash

.PHONY: help up down recreate-api logs-api serve-frontend prefetch-zim build-title-index api-health package-data verify-data

help:
	@echo "Targets:"
	@echo "  up                   - Start nginx + kiwix + api"
	@echo "  down                 - Stop all services"
	@echo "  recreate-api         - Force-recreate API container (no rebuild)"
	@echo "  logs-api             - Tail API logs"
	@echo "  serve-frontend       - Serve ./frontend at http://localhost:5173"
	@echo "  prefetch-zim         - Download the Wikipedia ZIM to ./data/zims/enwiki.zim"
	@echo "  build-title-index    - Build SQLite FTS title index inside API (use LIMIT=...)"
	@echo "  api-health           - GET /api/health"
	@echo "  package-data         - Bundle ZIM + title index into dist/wiki-in-a-box-data.tar.gz"
	@echo "  verify-data          - Verify data bundle integrity"

up:
	docker compose up -d nginx kiwix api

down:
	docker compose down

recreate-api:
	docker compose up -d --force-recreate api

logs-api:
	docker compose logs -f api

# --- ZIM prefetch ---
ZIM_DIR ?= ./data/zims
ZIM_FILE ?= enwiki.zim
ZIM_PATH := $(ZIM_DIR)/$(ZIM_FILE)
ZIM_URL_1 ?= https://gemmei.ftp.acc.umu.se/mirror/kiwix.org/zim/wikipedia/wikipedia_en_all_nopic_2025-08.zim
ZIM_URL_2 ?= https://ftp.fau.de/kiwix/zim/wikipedia/wikipedia_en_all_nopic_2025-08.zim

.PHONY: prefetch-zim
prefetch-zim:
	mkdir -p $(ZIM_DIR)
	@if [ -f "$(ZIM_PATH)" ]; then \
	  echo "ZIM already exists: $(ZIM_PATH)"; \
	else \
	  echo "Downloading ZIM (this is large; can take time)..."; \
	  (curl -L --fail --retry 3 --retry-connrefused -C - "$(ZIM_URL_1)" -o "$(ZIM_PATH)" \
	    || curl -L --fail --retry 3 --retry-connrefused -C - "$(ZIM_URL_2)" -o "$(ZIM_PATH)"); \
	fi
	@ls -lh $(ZIM_PATH) 2>/dev/null || true

.PHONY: prefetch-all
prefetch-all: prefetch-model prefetch-zim

api-health:
	@curl -sSf http://localhost:8000/api/health | jq . 2>/dev/null || curl -sSf http://localhost:8000/api/health

serve-frontend:
	cd frontend && python3 -m http.server 5173

.PHONY: build-title-index
build-title-index:
	@docker compose exec -T -e LIMIT=$(LIMIT) api python -m title_index

.PHONY: package-data verify-data
package-data:
		@mkdir -p dist
		@python - <<- 'PY'
			import os, json, hashlib, tarfile, time
			from pathlib import Path
			root = Path('.')
			zim = Path('data/zims/enwiki.zim')
			title = Path('data/title_index/titles.sqlite')
			out = Path('dist/wiki-in-a-box-data.tar.gz')
			manifest = {
			  'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
			  'files': []
			}
			def add_file(p: Path, arcname: str, tar: tarfile.TarFile):
			  h = hashlib.sha256()
			  with p.open('rb') as f:
			    for chunk in iter(lambda: f.read(1024*1024), b''):
			      h.update(chunk)
			  manifest['files'].append({'path': arcname, 'size': p.stat().st_size, 'sha256': h.hexdigest()})
			  tar.add(p, arcname=arcname)

			if not zim.exists():
			  raise SystemExit('missing data/zims/enwiki.zim')
			if not title.exists():
			  raise SystemExit('missing data/title_index/titles.sqlite (run make build-title-index)')

			with tarfile.open(out, 'w:gz') as tar:
			  add_file(zim, 'data/zims/enwiki.zim', tar)
			  add_file(title, 'data/title_index/titles.sqlite', tar)
			  mbytes = json.dumps(manifest, indent=2).encode()
			  import io
			  ti = tarfile.TarInfo('MANIFEST.json')
			  ti.size = len(mbytes)
			  ti.mtime = int(time.time())
			  tar.addfile(ti, io.BytesIO(mbytes))
			print(f'Wrote {out}')
		PY

verify-data:
	@python - <<- 'PY'
		import sys, tarfile, json, hashlib
		from pathlib import Path
		pkg = Path('dist/wiki-in-a-box-data.tar.gz')
		if not pkg.exists():
		    raise SystemExit('missing dist/wiki-in-a-box-data.tar.gz')
		with tarfile.open(pkg, 'r:gz') as tar:
		    m = tar.extractfile('MANIFEST.json').read()
		    manifest = json.loads(m)
		    ok = True
		    for f in manifest['files']:
		        member = tar.getmember(f['path'])
		        data = tar.extractfile(member).read()
		        h = hashlib.sha256(data).hexdigest()
		        if h != f['sha256']:
		            print('HASH MISMATCH:', f['path'])
		            ok = False
		    print('OK' if ok else 'FAILED')
	PY
