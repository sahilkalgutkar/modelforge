package batch

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// echo scores a row by returning it unchanged, so a test can tell which
// prediction belongs to which request just by looking at it. That is what makes
// misalignment — the failure mode batching most invites — detectable.
func echo(_ context.Context, rows [][]float64) ([][]float64, error) {
	out := make([][]float64, len(rows))
	for i, r := range rows {
		out[i] = append([]float64(nil), r...)
	}
	return out, nil
}

func newBatcher(t *testing.T, cfg Config, fn ScoreFunc) *Batcher {
	t.Helper()
	b, err := New(cfg, fn)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(b.Close)
	return b
}

func TestNewRequiresAScoreFunc(t *testing.T) {
	if _, err := New(Config{}, nil); err == nil {
		t.Fatal("New with a nil score function succeeded, want an error")
	}
}

func TestSingleRequestIsFlushedByTheTimer(t *testing.T) {
	b := newBatcher(t, Config{MaxSize: 32, MaxDelay: 20 * time.Millisecond}, echo)

	start := time.Now()
	got, err := b.Submit(context.Background(), []float64{1, 2, 3})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if len(got) != 3 || got[0] != 1 {
		t.Errorf("prediction = %v, want the row back", got)
	}
	// A lone request must wait for the window, and must not wait appreciably
	// longer — the window is a ceiling on added latency, not a suggestion.
	if elapsed < 15*time.Millisecond {
		t.Errorf("returned after %v, expected to wait about the %v window", elapsed, 20*time.Millisecond)
	}
	if elapsed > 2*time.Second {
		t.Errorf("returned after %v, far beyond the window", elapsed)
	}
}

// TestFullBatchDispatchesWithoutWaiting is the other half of the latency
// contract: once MaxSize requests are available there is nothing to wait for,
// so the window must not be honoured. A fixed-ticker design fails this.
func TestFullBatchDispatchesWithoutWaiting(t *testing.T) {
	const size = 8
	b := newBatcher(t, Config{MaxSize: size, MaxDelay: 10 * time.Second, QueueSize: 64}, echo)

	start := time.Now()
	var wg sync.WaitGroup
	errs := make([]error, size)
	for i := range size {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, errs[i] = b.Submit(context.Background(), []float64{float64(i)})
		}()
	}
	wg.Wait()
	elapsed := time.Since(start)

	for i, err := range errs {
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
	}
	if elapsed > 2*time.Second {
		t.Errorf("a full batch took %v with a 10s window; it should dispatch as soon as it fills", elapsed)
	}
	if got := b.Stats().Batches; got != 1 {
		t.Errorf("formed %d batches, want 1", got)
	}
}

// TestEveryCallerGetsItsOwnPrediction is the correctness property that matters
// most. Batching puts many callers' rows into one call, and an off-by-one in
// the scatter hands a caller a prediction computed for somebody else's input —
// which looks entirely plausible and is completely wrong.
func TestEveryCallerGetsItsOwnPrediction(t *testing.T) {
	b := newBatcher(t, Config{MaxSize: 16, MaxDelay: 5 * time.Millisecond, QueueSize: 512}, echo)

	const callers = 200
	var wg sync.WaitGroup
	bad := make(chan string, callers)
	for i := range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			want := float64(i)
			got, err := b.Submit(context.Background(), []float64{want, want * 2})
			if err != nil {
				bad <- fmt.Sprintf("caller %d: %v", i, err)
				return
			}
			if len(got) != 2 || got[0] != want || got[1] != want*2 {
				bad <- fmt.Sprintf("caller %d got %v, want [%v %v]", i, got, want, want*2)
			}
		}()
	}
	wg.Wait()
	close(bad)
	for msg := range bad {
		t.Error(msg)
	}

	stats := b.Stats()
	if stats.Rows != callers {
		t.Errorf("scored %d rows, want %d", stats.Rows, callers)
	}
	// The point of the exercise: these requests were actually grouped, not
	// scored one at a time.
	if stats.Batches >= callers {
		t.Errorf("formed %d batches for %d requests; nothing was batched", stats.Batches, callers)
	}
	t.Logf("%d requests in %d batches (mean %.1f, largest %d)",
		callers, stats.Batches, stats.Mean(), stats.LargestBatch)
}

// TestCancelledRequestIsDroppedBeforeScoring covers the wasted-work path: a
// caller that has given up must not have its row scored, because the load that
// causes timeouts is exactly when the spare capacity matters.
func TestCancelledRequestIsDroppedBeforeScoring(t *testing.T) {
	release := make(chan struct{})
	var scored atomic.Int64

	// The first batch blocks, so everything behind it accumulates while the
	// test cancels it.
	fn := func(ctx context.Context, rows [][]float64) ([][]float64, error) {
		scored.Add(int64(len(rows)))
		<-release
		return echo(ctx, rows)
	}
	b := newBatcher(t, Config{MaxSize: 4, MaxDelay: time.Millisecond, QueueSize: 64, Workers: 1}, fn)

	// Occupy the single worker, and wait until its batch has actually been
	// dispatched. Without this pause the next request joins the same batch —
	// the window is 1ms — and is scored alongside the blocker rather than
	// waiting behind it, which is not the situation under test.
	blocker := make(chan struct{})
	go func() {
		defer close(blocker)
		b.Submit(context.Background(), []float64{0}) //nolint:errcheck // result unused
	}()
	time.Sleep(40 * time.Millisecond)

	// Queue a request and abandon it while the worker is busy.
	ctx, cancel := context.WithCancel(context.Background())
	abandoned := make(chan error, 1)
	go func() {
		_, err := b.Submit(ctx, []float64{99})
		abandoned <- err
	}()
	time.Sleep(30 * time.Millisecond)
	cancel()

	if err := <-abandoned; !errors.Is(err, context.Canceled) {
		t.Errorf("abandoned Submit returned %v, want context.Canceled", err)
	}

	close(release)
	<-blocker
	b.Close()

	if got := b.Stats().Dropped; got == 0 {
		t.Error("no request was recorded as dropped; the cancelled row was scored anyway")
	}
	if got := scored.Load(); got > 1 {
		t.Errorf("scored %d rows, want 1 — the cancelled row should not have reached the model", got)
	}
}

// TestOverloadIsShedRatherThanQueued checks the backpressure decision. Once the
// queue is full, admitting more requests only makes the wait worse for the ones
// already committed.
func TestOverloadIsShedRatherThanQueued(t *testing.T) {
	release := make(chan struct{})
	defer close(release)

	fn := func(ctx context.Context, rows [][]float64) ([][]float64, error) {
		<-release
		return echo(ctx, rows)
	}
	b := newBatcher(t, Config{MaxSize: 1, MaxDelay: time.Millisecond, QueueSize: 2, Workers: 1}, fn)

	var (
		wg       sync.WaitGroup
		mu       sync.Mutex
		rejected int
	)
	for range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
			defer cancel()
			_, err := b.Submit(ctx, []float64{1})
			if errors.Is(err, ErrOverloaded) {
				mu.Lock()
				rejected++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	if rejected == 0 {
		t.Fatal("nothing was rejected with a queue of 2 and a blocked worker; requests are being queued without bound")
	}
	if got := b.Stats().Rejected; int(got) != rejected {
		t.Errorf("Stats().Rejected = %d, callers saw %d", got, rejected)
	}
	t.Logf("%d of 50 requests shed", rejected)
}

func TestScoreErrorReachesEveryCallerInTheBatch(t *testing.T) {
	boom := errors.New("model exploded")
	fn := func(context.Context, [][]float64) ([][]float64, error) { return nil, boom }
	b := newBatcher(t, Config{MaxSize: 8, MaxDelay: 5 * time.Millisecond, QueueSize: 64}, fn)

	var wg sync.WaitGroup
	errs := make([]error, 8)
	for i := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, errs[i] = b.Submit(context.Background(), []float64{float64(i)})
		}()
	}
	wg.Wait()

	for i, err := range errs {
		if !errors.Is(err, boom) {
			t.Errorf("caller %d got %v, want the score error", i, err)
		}
	}
}

// TestMismatchedResultCountFailsTheBatch covers the case where a score function
// misbehaves. Returning the wrong number of predictions must not be silently
// zipped against the requests, because there is no way to know which rows the
// results belong to and a caller would receive a confident wrong answer.
func TestMismatchedResultCountFailsTheBatch(t *testing.T) {
	fn := func(_ context.Context, rows [][]float64) ([][]float64, error) {
		return [][]float64{{1}}, nil // always one, regardless of input
	}
	b := newBatcher(t, Config{MaxSize: 4, MaxDelay: 5 * time.Millisecond, QueueSize: 32}, fn)

	var wg sync.WaitGroup
	errs := make([]error, 4)
	for i := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, errs[i] = b.Submit(context.Background(), []float64{float64(i)})
		}()
	}
	wg.Wait()

	var failed int
	for _, err := range errs {
		if err != nil {
			failed++
		}
	}
	if failed == 0 {
		t.Fatal("a score function returning the wrong number of predictions was accepted")
	}
}

// TestOneCallerCancellingDoesNotCancelTheBatch is why the batch is scored under
// its own context rather than any caller's. Sharing a context would let one
// client's timeout fail every unrelated request that happened to be batched
// with it.
func TestOneCallerCancellingDoesNotCancelTheBatch(t *testing.T) {
	started := make(chan struct{})
	var once sync.Once
	fn := func(ctx context.Context, rows [][]float64) ([][]float64, error) {
		once.Do(func() { close(started) })
		time.Sleep(60 * time.Millisecond)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return echo(ctx, rows)
	}
	b := newBatcher(t, Config{MaxSize: 4, MaxDelay: 10 * time.Millisecond, QueueSize: 32}, fn)

	doomed, cancel := context.WithCancel(context.Background())
	results := make(chan error, 2)

	go func() {
		_, err := b.Submit(doomed, []float64{1})
		results <- err
	}()
	go func() {
		_, err := b.Submit(context.Background(), []float64{2})
		results <- err
	}()

	<-started
	cancel()

	var survivors, cancelled int
	for range 2 {
		if err := <-results; err == nil {
			survivors++
		} else if errors.Is(err, context.Canceled) {
			cancelled++
		} else {
			t.Errorf("unexpected error: %v", err)
		}
	}
	if cancelled != 1 {
		t.Errorf("%d callers were cancelled, want 1", cancelled)
	}
	if survivors != 1 {
		t.Errorf("%d callers survived, want 1 — one client's cancellation took down its batch mates", survivors)
	}
}

func TestSubmitAfterCloseIsRefused(t *testing.T) {
	b, err := New(Config{MaxSize: 4, MaxDelay: time.Millisecond}, echo)
	if err != nil {
		t.Fatal(err)
	}
	b.Close()
	b.Close() // idempotent

	if _, err := b.Submit(context.Background(), []float64{1}); !errors.Is(err, ErrClosed) {
		t.Errorf("Submit after Close = %v, want ErrClosed", err)
	}
}

func TestSubmitWithAlreadyCancelledContext(t *testing.T) {
	b := newBatcher(t, Config{MaxSize: 4, MaxDelay: time.Millisecond}, echo)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := b.Submit(ctx, []float64{1}); !errors.Is(err, context.Canceled) {
		t.Errorf("Submit with a cancelled context = %v, want context.Canceled", err)
	}
}

// TestCloseDrainsQueuedRequests checks that shutdown answers requests already
// accepted instead of abandoning callers who are still waiting on them.
func TestCloseDrainsQueuedRequests(t *testing.T) {
	b, err := New(Config{MaxSize: 64, MaxDelay: time.Hour, QueueSize: 64}, echo)
	if err != nil {
		t.Fatal(err)
	}

	// MaxDelay is an hour, so nothing dispatches on its own; only the drain
	// on Close can answer these.
	const n = 10
	var wg sync.WaitGroup
	errs := make([]error, n)
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, errs[i] = b.Submit(context.Background(), []float64{float64(i)})
		}()
	}

	// Give the submissions time to reach the queue before shutting down.
	time.Sleep(50 * time.Millisecond)
	b.Close()
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("queued request %d was abandoned at shutdown: %v", i, err)
		}
	}
}

func TestDefaultsAreApplied(t *testing.T) {
	var cfg Config
	cfg.setDefaults()
	if cfg.MaxSize <= 0 || cfg.MaxDelay <= 0 || cfg.QueueSize <= 0 || cfg.Workers <= 0 {
		t.Fatalf("setDefaults left a zero value: %+v", cfg)
	}
}

func TestStatsMean(t *testing.T) {
	if got := (Stats{}).Mean(); got != 0 {
		t.Errorf("Mean of no batches = %v, want 0", got)
	}
	if got := (Stats{Batches: 4, Rows: 10}).Mean(); got != 2.5 {
		t.Errorf("Mean = %v, want 2.5", got)
	}
}

// BenchmarkBatchedVsSerial is the measurement that justifies the package. It
// compares a fixed amount of concurrent work with batching on and with the
// batch size pinned to 1, so the difference is the batching rather than the
// model.
func BenchmarkBatchedVsSerial(b *testing.B) {
	// A stand-in for per-call overhead — decoding, routing, metrics — that
	// batching amortises but per-row arithmetic does not.
	fn := func(_ context.Context, rows [][]float64) ([][]float64, error) {
		time.Sleep(200 * time.Microsecond)
		out := make([][]float64, len(rows))
		for i := range rows {
			out[i] = []float64{0.5}
		}
		return out, nil
	}

	run := func(b *testing.B, maxSize int) {
		bt, err := New(Config{MaxSize: maxSize, MaxDelay: time.Millisecond, QueueSize: 8192, Workers: 4}, fn)
		if err != nil {
			b.Fatal(err)
		}
		defer bt.Close()

		// Enough concurrent callers to actually fill a batch. The default
		// parallelism is GOMAXPROCS, which on this machine is fewer clients
		// than MaxSize — every batch would then time out half-empty and the
		// benchmark would measure the window rather than the batching. Serving
		// hundreds of concurrent requests is the regime batching is for, so
		// that is the regime to measure.
		b.SetParallelism(32)

		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				if _, err := bt.Submit(context.Background(), []float64{1, 2}); err != nil {
					b.Error(err)
					return
				}
			}
		})
	}

	b.Run("batch-size-1", func(b *testing.B) { run(b, 1) })
	b.Run("batch-size-64", func(b *testing.B) { run(b, 64) })
}
