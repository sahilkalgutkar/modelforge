# Local development. CI runs the same commands; see .github/workflows/ci.yml.

# Host ports the compose stack publishes. Exported so `docker compose` picks
# them up, and so the connection strings below follow Postgres when it moves --
# an override that left DB pointing at 5432 would be a knob that breaks things
# instead of fixing them. 3000 in particular is taken on any machine running a
# Node dev server.
POSTGRES_HOST_PORT ?= 5432
PROMETHEUS_HOST_PORT ?= 9090
GRAFANA_HOST_PORT ?= 3000
export POSTGRES_HOST_PORT PROMETHEUS_HOST_PORT GRAFANA_HOST_PORT

# Both default to the compose Postgres, and both defer to the environment
# variables the server and the tests read directly, so pointing everything at
# your own instance is one export rather than an argument on every target.
DB ?= $(or $(MODELFORGE_DATABASE_URL),postgres://modelforge:modelforge@localhost:$(POSTGRES_HOST_PORT)/modelforge?sslmode=disable)
TEST_DB ?= $(or $(MODELFORGE_TEST_DATABASE_URL),postgres://modelforge:modelforge@localhost:$(POSTGRES_HOST_PORT)/modelforge_test?sslmode=disable)

.PHONY: build test cover lint fixtures db up run tokens rotate clean

build:
	CGO_ENABLED=0 go build -o bin/modelforge ./cmd/modelforge
	CGO_ENABLED=0 go build -o bin/modelforgectl ./cmd/modelforgectl

# -p 1 because several suites reset the same test database; see ci.yml.
test:
	MODELFORGE_TEST_DATABASE_URL="$(TEST_DB)" \
	  go test -p 1 -race -coverpkg=./internal/... -coverprofile=coverage.out ./...
	go tool cover -func=coverage.out | tail -1

cover: test
	go tool cover -html=coverage.out -o coverage.html
	@echo "wrote coverage.html"

lint:
	gofmt -l . | grep -v '^testdata/' && exit 1 || true
	go vet ./...

# Regenerate the XGBoost parity fixtures. Needs python with xgboost installed.
fixtures:
	python3 tools/fixtures/generate.py

db:
	docker compose -f deploy/docker-compose.yml up -d postgres
	@until docker compose -f deploy/docker-compose.yml exec -T postgres pg_isready -U modelforge >/dev/null 2>&1; do sleep 1; done
	@docker compose -f deploy/docker-compose.yml exec -T postgres \
		psql -U modelforge -d modelforge -c "SELECT 1" >/dev/null
	@docker compose -f deploy/docker-compose.yml exec -T postgres \
		psql -U modelforge -tc "SELECT 1 FROM pg_database WHERE datname='modelforge_test'" \
		| grep -q 1 || docker compose -f deploy/docker-compose.yml exec -T postgres \
		createdb -U modelforge modelforge_test
	@echo "postgres ready"

up: db tokens
	docker compose -f deploy/docker-compose.yml up -d
	@echo "prometheus http://localhost:$(PROMETHEUS_HOST_PORT)   grafana http://localhost:$(GRAFANA_HOST_PORT) (admin/admin)"

# Mint the development tokens: one admin for the CLI, one read for Prometheus.
# Uses the --env form so this does not depend on the wording of the human
# output — a tidy-up to that prose should not silently break local setup.
tokens: build
	@./bin/modelforgectl token dev admin --env > deploy/.admin.env
	@./bin/modelforgectl token prometheus read --env > deploy/.scrape.env
	@. ./deploy/.scrape.env; printf '%s' "$$MODELFORGE_TOKEN" > deploy/prometheus-token
	@. ./deploy/.admin.env; ADMIN="$$MODELFORGE_TOKEN"; ADMIN_ENTRY="$$MODELFORGE_TOKENS_ENTRY"; \
	 . ./deploy/.scrape.env; \
	 { \
	   echo "# Development tokens. Rewrite this file and SIGHUP the server to rotate;"; \
	   echo "# it is also re-read on a timer. Never commit it."; \
	   echo "$$ADMIN_ENTRY"; \
	   echo "$$MODELFORGE_TOKENS_ENTRY"; \
	 } > deploy/tokens
	@. ./deploy/.admin.env; echo "export MODELFORGE_TOKEN=$$MODELFORGE_TOKEN" > deploy/dev-tokens.env
	@rm -f deploy/.admin.env deploy/.scrape.env
	@echo "wrote deploy/tokens (server) and deploy/dev-tokens.env (client)"

# Rotate the development admin token in place, with an overlap, without
# restarting the server. This is the same procedure the README documents.
rotate: build
	@./bin/modelforgectl token dev-next admin --env > deploy/.next.env
	@. ./deploy/.next.env; \
	  echo "$$MODELFORGE_TOKENS_ENTRY" >> deploy/tokens; \
	  echo "export MODELFORGE_TOKEN=$$MODELFORGE_TOKEN" > deploy/dev-tokens.env
	@rm -f deploy/.next.env
	@pkill -HUP -f 'bin/modelforge -addr' 2>/dev/null || true
	@echo "new token added and the server signalled; both are now valid."
	@echo "source deploy/dev-tokens.env, then remove the old line from deploy/tokens and SIGHUP again."

run: build
	@test -f deploy/tokens || $(MAKE) tokens
	MODELFORGE_DATABASE_URL="$(DB)" ./bin/modelforge -token-file deploy/tokens

clean:
	rm -rf bin coverage.out coverage.html var deploy/prometheus-token deploy/dev-tokens.env deploy/tokens
