// Package batch turns many concurrent single-row requests into few multi-row
// calls into a model.
//
// Scoring one row of a gradient-boosted model is on the order of a microsecond,
// while everything wrapped around it — the HTTP handler, JSON decoding, the
// routing decision, metrics — is considerably more. Under concurrent load the
// per-call overhead, not the arithmetic, is what sets throughput. Batching
// amortises that overhead across a batch at the cost of a bounded amount of
// added latency, and the bound is the whole design: a request waits at most
// maxDelay, and only if the batch has not already filled up.
//
// The alternative shape — a fixed ticker that scores whatever has accumulated —
// looks equivalent and is not. A request arriving just after a tick waits a
// full interval, the tick fires on an empty queue when traffic is light, and
// the added latency is a property of the clock rather than of the request. Here
// the timer starts when the *first* request of a batch arrives, so an idle
// system dispatches a lone request after exactly maxDelay and a busy one
// dispatches full batches immediately.
package batch

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Errors callers are expected to handle.
var (
	// ErrOverloaded means the queue is full. It is deliberately a fast
	// failure: once the queue is longer than the workers can drain, every
	// further request that joins it makes the wait worse for the ones already
	// there. Shedding at the edge keeps latency bounded for the requests that
	// are already committed.
	ErrOverloaded = errors.New("batch: overloaded")

	// ErrClosed means the batcher is shutting down.
	ErrClosed = errors.New("batch: closed")
)

// ScoreFunc scores a batch of rows. It must return one prediction per input row,
// in the same order.
type ScoreFunc func(ctx context.Context, rows [][]float64) ([][]float64, error)

// Config configures a Batcher.
type Config struct {
	// MaxSize is the largest batch that will be dispatched.
	MaxSize int

	// MaxDelay is the longest a request will wait for other requests to join
	// its batch. This is the added latency budget, and it is a ceiling rather
	// than an average.
	MaxDelay time.Duration

	// QueueSize bounds how many requests can be waiting to be batched.
	QueueSize int

	// Workers is the number of batches that may be in flight at once.
	Workers int
}

func (c *Config) setDefaults() {
	if c.MaxSize <= 0 {
		c.MaxSize = 32
	}
	if c.MaxDelay <= 0 {
		c.MaxDelay = 5 * time.Millisecond
	}
	if c.QueueSize <= 0 {
		c.QueueSize = 1024
	}
	if c.Workers <= 0 {
		c.Workers = 4
	}
}

// Stats is a snapshot of what the batcher has done, for metrics and tests.
type Stats struct {
	Batches      int64
	Rows         int64
	Dropped      int64 // requests abandoned by their caller before dispatch
	Rejected     int64 // requests refused because the queue was full
	LargestBatch int
}

// Mean returns the average batch size, or 0 if nothing has been dispatched.
// It is the number that says whether batching is doing anything: a mean near 1
// means requests are arriving too slowly to batch and MaxDelay is only adding
// latency.
func (s Stats) Mean() float64 {
	if s.Batches == 0 {
		return 0
	}
	return float64(s.Rows) / float64(s.Batches)
}

type request struct {
	ctx context.Context
	row []float64
	// out is buffered so the dispatcher never blocks delivering a result. A
	// caller that has already given up on its context is not around to
	// receive, and an unbuffered send would park the dispatcher goroutine on
	// it forever.
	out chan outcome
}

type outcome struct {
	pred []float64
	err  error
}

// Batcher groups requests into batches and scores them.
type Batcher struct {
	cfg   Config
	score ScoreFunc

	queue chan *request
	stop  chan struct{}

	// slots caps how many batches are in flight, which is what stops a burst
	// from spawning an unbounded number of goroutines all contending for CPU.
	slots chan struct{}

	closeOnce sync.Once
	collector sync.WaitGroup
	inflight  sync.WaitGroup

	mu    sync.Mutex
	stats Stats
}

// New starts a Batcher. Close must be called to release its goroutines.
func New(cfg Config, score ScoreFunc) (*Batcher, error) {
	if score == nil {
		return nil, errors.New("batch: a score function is required")
	}
	cfg.setDefaults()

	b := &Batcher{
		cfg:   cfg,
		score: score,
		queue: make(chan *request, cfg.QueueSize),
		stop:  make(chan struct{}),
		slots: make(chan struct{}, cfg.Workers),
	}
	b.collector.Add(1)
	go b.collect()
	return b, nil
}

// Submit scores one row, batching it with whatever else arrives in time.
//
// It returns ErrOverloaded rather than blocking when the queue is full, and
// honours ctx both while queued and while its batch is being scored.
func (b *Batcher) Submit(ctx context.Context, row []float64) ([]float64, error) {
	req := &request{ctx: ctx, row: row, out: make(chan outcome, 1)}

	select {
	case <-b.stop:
		return nil, ErrClosed
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	select {
	case b.queue <- req:
	default:
		b.record(func(s *Stats) { s.Rejected++ })
		return nil, fmt.Errorf("%w: %d requests already queued", ErrOverloaded, b.cfg.QueueSize)
	}

	select {
	case out := <-req.out:
		return out.pred, out.err
	case <-ctx.Done():
		// The caller is gone. The request may still be in a batch; the
		// dispatcher's send is buffered, so it will not block on the fact that
		// nobody is reading this channel any more.
		return nil, ctx.Err()
	}
}

// collect assembles batches and hands them to workers.
func (b *Batcher) collect() {
	defer b.collector.Done()

	for {
		var first *request
		select {
		case first = <-b.queue:
		case <-b.stop:
			// Drain whatever is queued so requests already accepted are
			// answered rather than abandoned. Submit has stopped accepting new
			// ones by this point.
			b.drain()
			return
		}

		batch := make([]*request, 0, b.cfg.MaxSize)
		batch = append(batch, first)

		// The window opens with the first request of this batch, so the
		// latency it adds is measured from that request's arrival.
		timer := time.NewTimer(b.cfg.MaxDelay)
	fill:
		for len(batch) < b.cfg.MaxSize {
			select {
			case req := <-b.queue:
				batch = append(batch, req)
			case <-timer.C:
				break fill
			case <-b.stop:
				break fill
			}
		}
		timer.Stop()

		b.dispatch(batch)
	}
}

// drain empties the queue into best-effort batches during shutdown.
func (b *Batcher) drain() {
	for {
		batch := make([]*request, 0, b.cfg.MaxSize)
		for len(batch) < b.cfg.MaxSize {
			select {
			case req := <-b.queue:
				batch = append(batch, req)
				continue
			default:
			}
			break
		}
		if len(batch) == 0 {
			return
		}
		b.dispatch(batch)
	}
}

// dispatch scores one batch, on a worker slot.
func (b *Batcher) dispatch(batch []*request) {
	if len(batch) == 0 {
		return
	}
	b.slots <- struct{}{}
	b.inflight.Add(1)
	go func() {
		defer func() {
			<-b.slots
			b.inflight.Done()
		}()
		b.run(batch)
	}()
}

// run scores a batch and scatters the results back to their callers.
func (b *Batcher) run(batch []*request) {
	// Requests whose caller has already given up are dropped before scoring.
	// They would otherwise take a slot in the batch and consume work whose
	// result nobody will read — and under the load that makes clients time
	// out, that is exactly when the wasted work hurts most.
	live := make([]*request, 0, len(batch))
	rows := make([][]float64, 0, len(batch))
	var dropped int64
	for _, req := range batch {
		if req.ctx.Err() != nil {
			dropped++
			continue
		}
		live = append(live, req)
		rows = append(rows, req.row)
	}
	if len(live) == 0 {
		b.record(func(s *Stats) { s.Dropped += dropped })
		return
	}

	// The batch's context is deliberately not any single caller's: one client
	// cancelling must not cancel the batch its row happens to share. Callers
	// are released individually by Submit's own select on ctx.Done().
	preds, err := b.score(context.Background(), rows)

	switch {
	case err != nil:
		for _, req := range live {
			req.out <- outcome{err: err}
		}
	case len(preds) != len(live):
		// A score function that returns the wrong number of rows would
		// otherwise misalign predictions with requests, handing callers
		// someone else's answer. Failing the batch is the only safe response,
		// because there is no way to tell which rows the results belong to.
		e := fmt.Errorf("batch: score returned %d predictions for %d rows", len(preds), len(live))
		for _, req := range live {
			req.out <- outcome{err: e}
		}
	default:
		for i, req := range live {
			req.out <- outcome{pred: preds[i]}
		}
	}

	n := len(live)
	b.record(func(s *Stats) {
		s.Batches++
		s.Rows += int64(n)
		s.Dropped += dropped
		if n > s.LargestBatch {
			s.LargestBatch = n
		}
	})
}

func (b *Batcher) record(f func(*Stats)) {
	b.mu.Lock()
	defer b.mu.Unlock()
	f(&b.stats)
}

// Stats returns a snapshot of the batcher's counters.
func (b *Batcher) Stats() Stats {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.stats
}

// Close stops accepting requests, finishes what is already queued, and waits
// for in-flight batches. It is safe to call more than once.
func (b *Batcher) Close() {
	b.closeOnce.Do(func() { close(b.stop) })
	b.collector.Wait()
	b.inflight.Wait()
}
