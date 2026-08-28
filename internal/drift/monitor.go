package drift

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// Reading is one feature's drift measurement over the current window.
type Reading struct {
	Feature     string   `json:"feature"`
	PSI         float64  `json:"psi"`
	Severity    Severity `json:"severity"`
	Samples     int64    `json:"samples"`
	MissingRate float64  `json:"missing_rate"`
}

// Report is a full drift measurement for a model version.
type Report struct {
	Model    string    `json:"model"`
	Version  int       `json:"version"`
	Window   string    `json:"window"`
	Samples  int64     `json:"samples"`
	Features []Reading `json:"features"`
	// Prediction is drift in the model's own output distribution. It is
	// reported separately from the inputs because the two answer different
	// questions: input drift says the world changed, output drift says the
	// model's behaviour changed. Either can happen without the other — a
	// model can absorb a shifted input and keep predicting the same
	// distribution, and a model can start predicting differently because a
	// feature it depends on heavily moved while the others held still.
	Prediction *Reading `json:"prediction,omitempty"`
}

// Worst returns the highest-PSI feature reading, which is the one an alert
// should name.
func (r Report) Worst() (Reading, bool) {
	var (
		worst Reading
		found bool
	)
	for _, f := range r.Features {
		if !found || f.PSI > worst.PSI {
			worst, found = f, true
		}
	}
	return worst, found
}

// MinSamples is the number of observations a window needs before its PSI is
// reported.
//
// PSI over a handful of rows is dominated by which bins happened to be hit, and
// an alerting system that fires on that trains people to ignore it. The
// threshold is on the window, not on the total, because a window that has just
// rolled legitimately starts empty.
const MinSamples = 200

// Monitor accumulates live observations for one model version and compares them
// against the baselines.
//
// It is a fixed-size ring of time buckets rather than a list of retained
// samples. Keeping the raw values would make memory grow with traffic, which is
// exactly backwards: the busier the service, the more it would cost to watch
// it. Bin counts are all PSI needs, and the ring's memory is a function of the
// window and the bin count alone — the same whether it sees ten requests a
// second or ten thousand.
type Monitor struct {
	model   string
	version int

	baselines []Baseline
	// predBaseline is optional; a model registered without one still gets
	// input drift.
	predBaseline *Baseline

	window    time.Duration
	bucketDur time.Duration

	now func() time.Time

	mu      sync.Mutex
	buckets []*window
	head    int
}

// window holds the counts accumulated in one slice of time.
type window struct {
	start time.Time
	// counts[f][b] is how many observations of feature f fell in bin b.
	counts [][]int64
	// missing[f] counts observations where feature f was NaN. These are
	// excluded from the PSI bins — a missing value has no place on a numeric
	// axis — but tracked separately, because a feature that suddenly becomes
	// 40% missing is a serious problem that binning alone would hide.
	missing []int64
	predict []int64
	samples int64
}

// Config configures a Monitor.
type Config struct {
	Model   string
	Version int
	// Window is how much history a reading covers.
	Window time.Duration
	// Buckets is how many slices the window is divided into. More buckets mean
	// the window slides more smoothly at the cost of memory.
	Buckets int
	// Now is injectable so tests can drive the clock rather than sleep.
	Now func() time.Time
}

// NewMonitor creates a Monitor for one model version.
func NewMonitor(cfg Config, baselines []Baseline, predBaseline *Baseline) (*Monitor, error) {
	if len(baselines) == 0 {
		return nil, fmt.Errorf("drift: no baselines for %s:%d", cfg.Model, cfg.Version)
	}
	if cfg.Window <= 0 {
		cfg.Window = 15 * time.Minute
	}
	if cfg.Buckets < 2 {
		cfg.Buckets = 6
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}

	m := &Monitor{
		model:        cfg.Model,
		version:      cfg.Version,
		baselines:    baselines,
		predBaseline: predBaseline,
		window:       cfg.Window,
		bucketDur:    cfg.Window / time.Duration(cfg.Buckets),
		now:          cfg.Now,
		buckets:      make([]*window, cfg.Buckets),
	}
	return m, nil
}

// Features returns the feature names the monitor watches, in order.
func (m *Monitor) Features() []string {
	out := make([]string, len(m.baselines))
	for i, b := range m.baselines {
		out[i] = b.Feature
	}
	return out
}

func (m *Monitor) newWindow(start time.Time) *window {
	w := &window{
		start:   start,
		counts:  make([][]int64, len(m.baselines)),
		missing: make([]int64, len(m.baselines)),
	}
	for i, b := range m.baselines {
		w.counts[i] = make([]int64, b.Bins())
	}
	if m.predBaseline != nil {
		w.predict = make([]int64, m.predBaseline.Bins())
	}
	return w
}

// current returns the bucket for now, rolling the ring forward if needed.
func (m *Monitor) current(now time.Time) *window {
	slot := now.Truncate(m.bucketDur)

	if w := m.buckets[m.head]; w != nil && w.start.Equal(slot) {
		return w
	}
	// A new slice of time. Advance and overwrite the oldest bucket, which is
	// what makes the window slide without any allocation growth over time.
	m.head = (m.head + 1) % len(m.buckets)
	w := m.newWindow(slot)
	m.buckets[m.head] = w
	return w
}

// Observe records one scored request: its input row and the prediction it
// produced.
func (m *Monitor) Observe(row []float64, prediction []float64) {
	if len(row) != len(m.baselines) {
		// A row of the wrong width is a caller bug, but drift monitoring must
		// never be the thing that fails a request, so it is dropped rather
		// than reported.
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	w := m.current(m.now())
	w.samples++
	for i, v := range row {
		if math.IsNaN(v) {
			w.missing[i]++
			continue
		}
		w.counts[i][m.baselines[i].Bucket(v)]++
	}
	if m.predBaseline != nil && len(prediction) > 0 {
		// For a multi-class model the first output is monitored; comparing a
		// whole probability vector against one baseline would need a joint
		// distribution, which is a different statistic.
		if v := prediction[0]; !math.IsNaN(v) {
			w.predict[m.predBaseline.Bucket(v)]++
		}
	}
}

// Report computes drift across the current window.
//
// It returns ok=false until the window holds MinSamples observations, so a
// caller cannot accidentally alert on a reading that is mostly noise.
func (m *Monitor) Report() (Report, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	cutoff := m.now().Add(-m.window)

	totals := make([][]int64, len(m.baselines))
	for i, b := range m.baselines {
		totals[i] = make([]int64, b.Bins())
	}
	missing := make([]int64, len(m.baselines))
	var predTotals []int64
	if m.predBaseline != nil {
		predTotals = make([]int64, m.predBaseline.Bins())
	}
	var samples int64

	for _, w := range m.buckets {
		// Buckets older than the window are ignored rather than cleared. They
		// are overwritten when the ring comes round to them again, so there is
		// nothing to free and no separate expiry pass to get wrong.
		if w == nil || w.start.Before(cutoff) {
			continue
		}
		samples += w.samples
		for i := range totals {
			for b := range totals[i] {
				totals[i][b] += w.counts[i][b]
			}
			missing[i] += w.missing[i]
		}
		for b := range predTotals {
			predTotals[b] += w.predict[b]
		}
	}

	rep := Report{
		Model:   m.model,
		Version: m.version,
		Window:  m.window.String(),
		Samples: samples,
	}
	if samples < MinSamples {
		return rep, false
	}

	for i, b := range m.baselines {
		r := Reading{Feature: b.Feature, Samples: samples}
		if samples > 0 {
			r.MissingRate = float64(missing[i]) / float64(samples)
		}
		// PSI is computed over the observations that had a value. Folding
		// missing rows into a bin would move the whole distribution as the
		// missing rate changed, and report that as a shift in the values
		// themselves.
		if psi, err := b.PSI(totals[i]); err == nil {
			r.PSI = psi
			r.Severity = Classify(psi)
		} else {
			r.Severity = SeverityStable
		}
		rep.Features = append(rep.Features, r)
	}

	if m.predBaseline != nil {
		r := Reading{Feature: "prediction", Samples: samples}
		if psi, err := m.predBaseline.PSI(predTotals); err == nil {
			r.PSI = psi
			r.Severity = Classify(psi)
			rep.Prediction = &r
		}
	}
	return rep, true
}
