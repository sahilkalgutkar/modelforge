# Local development. CI runs the same commands; see .github/workflows/ci.yml.

DB ?= postgres://modelforge:modelforge@localhost:5432/modelforge?sslmode=disable
TEST_DB ?= postgres://modelforge:modelforge@localhost:5432/modelforge_test?sslmode=disable

.PHONY: build test cover lint fixtures db up run clean

build:
	CGO_ENABLED=0 go build -o bin/modelforge ./cmd/modelforge
	CGO_ENABLED=0 go build -o bin/modelforgectl ./cmd/modelforgectl

# -p 1 because several suites reset the same test database; see ci.yml.
test:
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

up: db
	docker compose -f deploy/docker-compose.yml up -d
	@echo "prometheus http://localhost:9090   grafana http://localhost:3000 (admin/admin)"

run: build
	MODELFORGE_DATABASE_URL="$(DB)" ./bin/modelforge

clean:
	rm -rf bin coverage.out coverage.html var
