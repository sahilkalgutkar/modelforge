// Package cli implements modelforgectl, the client for driving a modelforge
// server from a terminal or a deploy script.
//
// The logic lives here rather than in main so it can be tested against an
// httptest server: a CLI whose only test is "it compiles" is exactly the tool
// that turns out to send the wrong verb during a rollback.
package cli

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/sahilkalgutkar/modelforge/internal/auth"
	"github.com/sahilkalgutkar/modelforge/internal/routing"
)

// Client talks to a modelforge server.
type Client struct {
	BaseURL string
	HTTP    *http.Client
	Out     io.Writer
	// Token is sent as a bearer credential. Empty means send nothing, which is
	// what talking to a server started with -auth-disabled looks like.
	Token string
}

// NewClient creates a Client with sensible defaults.
func NewClient(baseURL string, out io.Writer) *Client {
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		Token:   os.Getenv("MODELFORGE_TOKEN"),
		// The timeout is generous because pushing a model uploads its
		// artifact, which is the slowest thing this client does by far.
		HTTP: &http.Client{Timeout: 2 * time.Minute},
		Out:  out,
	}
}

// Usage is the help text.
const Usage = `modelforgectl — client for a modelforge server

Usage:
  modelforgectl [-addr URL] <command> [arguments]

Commands:
  models                                  list registered models
  create <model> [description]            register a new model
  versions <model>                        list a model's versions
  push <model> <file.json> <features...>  upload a trained model as a new version
  deploy <model> <version>                send all traffic to one version
  canary <model> <stable> <candidate> <percent>
                                          split traffic between two versions
  shadow <model> <version>                mirror traffic to a version without serving it
  rollback <model> <version>              send all traffic back to a known-good version
  policy <model>                          show the current traffic policy
  baseline <model> <version> <file.json>  attach training distributions for drift
  drift <model> <version>                 show drift readings
  stats <model> <version>                 show batching and health counters
  token <name> <scope>[+<scope>] [--env]  mint an API token and print its config entry

Flags:
  -addr URL   server address (default http://localhost:8080, or MODELFORGE_ADDR)

Environment:
  MODELFORGE_TOKEN   bearer token sent with every request

Scopes: predict (call the serving endpoint), read (inspect models, policies,
drift and metrics), admin (everything, including changing what serves traffic).
`

// Run executes one command and returns a process exit code.
func Run(args []string, addr string, out io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(out, Usage)
		return 2
	}
	c := NewClient(addr, out)

	var err error
	switch args[0] {
	case "models":
		err = c.ListModels()
	case "create":
		if len(args) < 2 {
			err = fmt.Errorf("create needs a model name")
			break
		}
		err = c.CreateModel(args[1], strings.Join(args[2:], " "))
	case "versions":
		if len(args) < 2 {
			err = fmt.Errorf("versions needs a model name")
			break
		}
		err = c.ListVersions(args[1])
	case "push":
		if len(args) < 4 {
			err = fmt.Errorf("push needs a model, a file and at least one feature name")
			break
		}
		err = c.Push(args[1], args[2], args[3:])
	case "deploy":
		var v int
		if v, err = versionArg(args, "deploy"); err == nil {
			err = c.setRoutes(args[1], []routing.Route{{Version: v, Weight: 100}})
		}
	case "rollback":
		// Deliberately the same operation as deploy, under a name that says
		// what it is for. A rollback that needs different mechanics from a
		// deploy is a rollback nobody trusts at 3am, and the point of
		// immutable versions is that going back is just going forward to an
		// older number.
		var v int
		if v, err = versionArg(args, "rollback"); err == nil {
			err = c.setRoutes(args[1], []routing.Route{{Version: v, Weight: 100}})
		}
	case "canary":
		err = c.canary(args)
	case "shadow":
		err = c.shadow(args)
	case "policy":
		if len(args) < 2 {
			err = fmt.Errorf("policy needs a model name")
			break
		}
		err = c.ShowPolicy(args[1])
	case "baseline":
		if len(args) < 4 {
			err = fmt.Errorf("baseline needs a model, a version and a JSON file")
			break
		}
		var v int
		if v, err = versionArg(args, "baseline"); err == nil {
			err = c.PushBaseline(args[1], v, args[3])
		}
	case "drift":
		var v int
		if v, err = versionArg(args, "drift"); err == nil {
			err = c.ShowDrift(args[1], v)
		}
	case "stats":
		var v int
		if v, err = versionArg(args, "stats"); err == nil {
			err = c.ShowStats(args[1], v)
		}
	case "token":
		if len(args) < 3 {
			err = fmt.Errorf("token needs a name and at least one scope")
			break
		}
		envOnly := len(args) > 3 && args[3] == "--env"
		err = MintToken(args[1], args[2], out, envOnly)
	default:
		fmt.Fprintf(out, "unknown command %q\n\n%s", args[0], Usage)
		return 2
	}

	if err != nil {
		fmt.Fprintf(out, "error: %v\n", err)
		return 1
	}
	return 0
}

func versionArg(args []string, cmd string) (int, error) {
	if len(args) < 3 {
		return 0, fmt.Errorf("%s needs a model and a version", cmd)
	}
	v, err := strconv.Atoi(args[2])
	if err != nil {
		return 0, fmt.Errorf("version must be a number, got %q", args[2])
	}
	return v, nil
}

func (c *Client) canary(args []string) error {
	if len(args) < 5 {
		return fmt.Errorf("canary needs a model, a stable version, a candidate version and a percentage")
	}
	stable, err := strconv.Atoi(args[2])
	if err != nil {
		return fmt.Errorf("stable version must be a number, got %q", args[2])
	}
	candidate, err := strconv.Atoi(args[3])
	if err != nil {
		return fmt.Errorf("candidate version must be a number, got %q", args[3])
	}
	pct, err := strconv.Atoi(args[4])
	if err != nil || pct < 0 || pct > 100 {
		return fmt.Errorf("percentage must be a number between 0 and 100, got %q", args[4])
	}
	return c.setRoutes(args[1], []routing.Route{
		{Version: stable, Weight: 100 - pct},
		{Version: candidate, Weight: pct},
	})
}

// setRoutes changes what is serving while leaving an existing shadow in place.
//
// Carrying the shadow across is what makes `deploy` and `canary` consistent
// with `shadow`, which already preserves the routes. Without it, promoting a
// candidate would silently stop the observation of whatever else was being
// watched, and the operator would find out by noticing an empty graph.
//
// The one case where the shadow is dropped is when it is now serving. The
// server rejects a policy where a version is both, and rightly: shadowing a
// version against itself for part of the traffic produces a divergence rate
// that mixes two different comparisons and means nothing. Promoting a shadow
// to a canary is the normal way a rollout proceeds, so this clears it rather
// than failing.
func (c *Client) setRoutes(model string, routes []routing.Route) error {
	p := routing.Policy{Routes: routes}

	// A model with no policy yet is the common first deploy, not an error.
	if current, err := c.getPolicy(model); err == nil && current.Shadow != nil {
		serving := false
		for _, r := range routes {
			if r.Version == *current.Shadow {
				serving = true
				break
			}
		}
		if serving {
			fmt.Fprintf(c.Out, "note: version %d is now serving, so it is no longer shadowed\n", *current.Shadow)
		} else {
			p.Shadow = current.Shadow
		}
		p.Guard = current.Guard
	}
	return c.SetPolicy(model, p)
}

func (c *Client) shadow(args []string) error {
	v, err := versionArg(args, "shadow")
	if err != nil {
		return err
	}
	// Shadowing preserves whatever is currently serving. Replacing the routes
	// as well would turn "watch this candidate" into an unannounced deploy.
	current, err := c.getPolicy(args[1])
	if err != nil {
		return fmt.Errorf("cannot add a shadow before a model is serving: %w", err)
	}
	current.Shadow = &v
	return c.SetPolicy(args[1], current)
}

// --- operations ---

// ListModels prints the registered models.
func (c *Client) ListModels() error {
	var body struct {
		Models []struct {
			Name        string    `json:"name"`
			Description string    `json:"description"`
			CreatedAt   time.Time `json:"created_at"`
		} `json:"models"`
	}
	if err := c.get("/v1/models", &body); err != nil {
		return err
	}
	if len(body.Models) == 0 {
		fmt.Fprintln(c.Out, "no models registered")
		return nil
	}
	tw := tabwriter.NewWriter(c.Out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "MODEL\tCREATED\tDESCRIPTION")
	for _, m := range body.Models {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", m.Name, m.CreatedAt.Format(time.DateOnly), m.Description)
	}
	return tw.Flush()
}

// CreateModel registers a model name.
func (c *Client) CreateModel(name, description string) error {
	if err := c.post("/v1/models", map[string]string{
		"name": name, "description": description,
	}, nil); err != nil {
		return err
	}
	fmt.Fprintf(c.Out, "created model %s\n", name)
	return nil
}

// ListVersions prints a model's versions.
func (c *Client) ListVersions(model string) error {
	var body struct {
		Versions []struct {
			Version   int       `json:"version"`
			Digest    string    `json:"digest"`
			SizeBytes int64     `json:"size_bytes"`
			Features  []string  `json:"features"`
			CreatedAt time.Time `json:"created_at"`
			Notes     string    `json:"notes"`
		} `json:"versions"`
	}
	if err := c.get("/v1/models/"+model+"/versions", &body); err != nil {
		return err
	}
	if len(body.Versions) == 0 {
		fmt.Fprintf(c.Out, "%s has no versions\n", model)
		return nil
	}
	tw := tabwriter.NewWriter(c.Out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "VERSION\tDIGEST\tSIZE\tFEATURES\tCREATED\tNOTES")
	for _, v := range body.Versions {
		digest := v.Digest
		if len(digest) > 12 {
			digest = digest[:12]
		}
		fmt.Fprintf(tw, "%d\t%s\t%s\t%d\t%s\t%s\n",
			v.Version, digest, humanBytes(v.SizeBytes), len(v.Features),
			v.CreatedAt.Format(time.DateOnly), v.Notes)
	}
	return tw.Flush()
}

// Push uploads a trained model file as a new version.
func (c *Client) Push(model, path string, features []string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open model file: %w", err)
	}
	defer f.Close()

	q := make([]string, 0, len(features))
	for _, name := range features {
		q = append(q, "feature="+name)
	}
	url := fmt.Sprintf("%s/v1/models/%s/versions?%s", c.BaseURL, model, strings.Join(q, "&"))

	// The file is streamed rather than read into memory: a model artifact can
	// be hundreds of megabytes and this is a CLI people run on laptops.
	req, err := http.NewRequest("POST", url, f)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	c.authorize(req)

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return serverError(resp.StatusCode, raw)
	}

	var out struct {
		Version struct {
			Version int    `json:"version"`
			Digest  string `json:"digest"`
		} `json:"version"`
		Objective string `json:"objective"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return err
	}
	digest := out.Version.Digest
	if len(digest) > 12 {
		digest = digest[:12]
	}
	fmt.Fprintf(c.Out, "pushed %s version %d (%s, objective %s, %d features)\n",
		model, out.Version.Version, digest, out.Objective, len(features))
	fmt.Fprintf(c.Out, "it is registered but not serving; use `deploy` or `canary` to send it traffic\n")
	return nil
}

// SetPolicy installs a traffic policy.
func (c *Client) SetPolicy(model string, p routing.Policy) error {
	p.Model = model
	if err := c.put("/v1/models/"+model+"/policy", p, nil); err != nil {
		return err
	}
	fmt.Fprintf(c.Out, "%s now serving: %s\n", model, p.String())
	return nil
}

func (c *Client) getPolicy(model string) (routing.Policy, error) {
	var p routing.Policy
	if err := c.get("/v1/models/"+model+"/policy", &p); err != nil {
		return routing.Policy{}, err
	}
	return p, nil
}

// ShowPolicy prints a model's traffic policy.
func (c *Client) ShowPolicy(model string) error {
	p, err := c.getPolicy(model)
	if err != nil {
		return err
	}
	fmt.Fprintf(c.Out, "%s: %s\n", model, p.String())
	if p.Guard != nil {
		fmt.Fprintf(c.Out, "  guard: remove a version above %.0f%% errors after %d requests\n",
			p.Guard.MaxErrorRate*100, p.Guard.MinRequests)
	}
	return nil
}

// PushBaseline uploads training-set distributions so drift can be measured.
//
// The file holds raw samples per feature, not pre-computed bins: the binning
// rules are the server's, and a client computing its own would silently
// disagree with the next server version.
func (c *Client) PushBaseline(model string, version int, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open baseline file: %w", err)
	}
	defer f.Close()

	url := fmt.Sprintf("%s/v1/models/%s/versions/%d/baseline", c.BaseURL, model, version)
	req, err := http.NewRequest("PUT", url, f)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	c.authorize(req)

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return serverError(resp.StatusCode, raw)
	}

	var out struct {
		Features           int  `json:"features"`
		Bins               int  `json:"bins"`
		PredictionBaseline bool `json:"prediction_baseline"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return err
	}
	fmt.Fprintf(c.Out, "%s:%d now monitored: %d features in %d bins",
		model, version, out.Features, out.Bins)
	if out.PredictionBaseline {
		fmt.Fprint(c.Out, ", plus prediction drift")
	}
	fmt.Fprintln(c.Out)
	return nil
}

// ShowDrift prints a version's drift readings.
func (c *Client) ShowDrift(model string, version int) error {
	var body struct {
		Ready  bool `json:"ready"`
		Report struct {
			Samples  int64  `json:"samples"`
			Window   string `json:"window"`
			Features []struct {
				Feature     string  `json:"feature"`
				PSI         float64 `json:"psi"`
				Severity    string  `json:"severity"`
				MissingRate float64 `json:"missing_rate"`
			} `json:"features"`
			Prediction *struct {
				PSI      float64 `json:"psi"`
				Severity string  `json:"severity"`
			} `json:"prediction"`
		} `json:"report"`
		MinSamples int `json:"min_samples"`
	}
	if err := c.get(fmt.Sprintf("/v1/models/%s/versions/%d/drift", model, version), &body); err != nil {
		return err
	}

	if !body.Ready {
		// "Not enough traffic to say" is a different answer from "no drift",
		// and conflating them is how a dashboard reports everything is fine
		// about a model nobody is calling.
		fmt.Fprintf(c.Out, "%s:%d — not enough traffic yet (%d of %d samples in the window)\n",
			model, version, body.Report.Samples, body.MinSamples)
		return nil
	}

	fmt.Fprintf(c.Out, "%s:%d over %s (%d samples)\n", model, version, body.Report.Window, body.Report.Samples)
	features := body.Report.Features
	sort.Slice(features, func(i, j int) bool { return features[i].PSI > features[j].PSI })

	tw := tabwriter.NewWriter(c.Out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "FEATURE\tPSI\tSEVERITY\tMISSING")
	for _, f := range features {
		fmt.Fprintf(tw, "%s\t%.4f\t%s\t%.1f%%\n", f.Feature, f.PSI, f.Severity, f.MissingRate*100)
	}
	if p := body.Report.Prediction; p != nil {
		fmt.Fprintf(tw, "%s\t%.4f\t%s\t-\n", "(prediction)", p.PSI, p.Severity)
	}
	return tw.Flush()
}

// ShowStats prints batching and health counters.
func (c *Client) ShowStats(model string, version int) error {
	var body struct {
		Batches        int64   `json:"batches"`
		Rows           int64   `json:"rows"`
		MeanBatch      float64 `json:"mean_batch"`
		LargestBatch   int     `json:"largest_batch"`
		Dropped        int64   `json:"dropped"`
		Rejected       int64   `json:"rejected"`
		Requests       int64   `json:"requests"`
		Failures       int64   `json:"failures"`
		RemovedByGuard bool    `json:"removed_by_guard"`
	}
	if err := c.get(fmt.Sprintf("/v1/models/%s/versions/%d/stats", model, version), &body); err != nil {
		return err
	}
	fmt.Fprintf(c.Out, "%s:%d\n", model, version)
	fmt.Fprintf(c.Out, "  requests   %d (%d failed)\n", body.Requests, body.Failures)
	fmt.Fprintf(c.Out, "  batching   %d rows in %d batches, mean %.1f, largest %d\n",
		body.Rows, body.Batches, body.MeanBatch, body.LargestBatch)
	fmt.Fprintf(c.Out, "  shed       %d rejected, %d abandoned before scoring\n", body.Rejected, body.Dropped)
	if body.RemovedByGuard {
		fmt.Fprintf(c.Out, "  GUARD      this version was removed from the split by the rollout guard\n")
	}
	return nil
}

// MintToken generates a token and prints it alongside the entry to configure.
//
// The token is shown once and never stored: the server only ever holds its
// digest, so there is nowhere to look it up later. That is the point — a
// credential store that can show you the secret again is a credential store
// that can leak it — but it does mean losing one means minting a replacement,
// so the output says so rather than letting somebody discover it.
// The --env form exists so that scripts and the Makefile can consume the
// result without parsing prose. Machine output and human output being the same
// text is how a tidy-up to the wording silently breaks somebody's setup script.
func MintToken(name, scopes string, out io.Writer, envOnly bool) error {
	for _, s := range strings.Split(scopes, "+") {
		if !slices.Contains(auth.AllScopes, auth.Scope(strings.TrimSpace(s))) {
			return fmt.Errorf("unknown scope %q", s)
		}
	}
	if strings.ContainsAny(name, ":+ ") || name == "" {
		// The config format is colon-delimited with plus-separated scopes, so a
		// name carrying either would produce an entry that parses as something
		// else entirely.
		return fmt.Errorf("token name %q must not be empty or contain ':', '+' or spaces", name)
	}

	token, err := auth.GenerateToken()
	if err != nil {
		return err
	}

	if envOnly {
		fmt.Fprintf(out, "MODELFORGE_TOKEN=%s\n", token)
		fmt.Fprintf(out, "MODELFORGE_TOKENS_ENTRY=%s:%s:%s\n", name, scopes, auth.Digest(token))
		return nil
	}

	fmt.Fprintf(out, "token (shown once, store it now):\n\n  %s\n\n", token)
	fmt.Fprintf(out, "add this to the server's MODELFORGE_TOKENS:\n\n  %s:%s:%s\n\n",
		name, scopes, auth.Digest(token))
	fmt.Fprintf(out, "the server stores only the digest, so this token cannot be recovered — "+
		"mint a new one if it is lost.\n")
	return nil
}

// --- transport ---

func (c *Client) get(path string, out any) error        { return c.do("GET", path, nil, out) }
func (c *Client) post(path string, body, out any) error { return c.do("POST", path, body, out) }
func (c *Client) put(path string, body, out any) error  { return c.do("PUT", path, body, out) }

func (c *Client) do(method, path string, body, out any) error {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, c.BaseURL+path, r)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	c.authorize(req)
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode >= 300 {
		return serverError(resp.StatusCode, raw)
	}
	if out != nil {
		return json.Unmarshal(raw, out)
	}
	return nil
}

// serverError turns an error response into something readable. The server
// sends {"error": "..."}, and printing that message beats printing a status
// code the operator then has to go and look up.
// authorize attaches the bearer credential, if one is configured.
func (c *Client) authorize(req *http.Request) {
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
}

func serverError(status int, raw []byte) error {
	var e struct {
		Error string `json:"error"`
	}
	msg := strings.TrimSpace(string(raw))
	if json.Unmarshal(raw, &e) == nil && e.Error != "" {
		msg = e.Error
	}

	// 401 and 403 have different fixes, and saying which is which here saves
	// the operator from reading the server's logs to find out.
	switch status {
	case http.StatusUnauthorized:
		return fmt.Errorf("server returned 401: %s — set MODELFORGE_TOKEN to a valid token "+
			"(mint one with `modelforgectl token <name> <scope>`)", msg)
	case http.StatusForbidden:
		return fmt.Errorf("server returned 403: %s — the token is valid but not permitted; "+
			"it needs a broader scope", msg)
	}
	return fmt.Errorf("server returned %d: %s", status, msg)
}

func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%dB", n)
	}
	div, exp := int64(unit), 0
	for x := n / unit; x >= unit; x /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f%cB", float64(n)/float64(div), "KMGT"[exp])
}
