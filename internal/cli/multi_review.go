package cli

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"sync"

	"reasonix/internal/agent"
	"reasonix/internal/boot"
	"reasonix/internal/config"
	"reasonix/internal/event"
	"reasonix/internal/skill"
)

// persona defines a review dimension. The Body is used as the subagent's
// system/session prompt; the shared review skill's read-only tool registry is
// reused so every persona stays safely read-only.
type persona struct {
	Name        string
	Description string
	Body        string
}

// built-in personas for multi-review. Each focuses on a different risk surface.
// These are intentionally minimal; projects can add custom personas via the
// --team flag or future skill overrides.
var builtinPersonas = map[string]persona{
	"quality": {
		Name:        "quality",
		Description: "general code quality and correctness",
		Body: `You are a quality-focused code reviewer. Inspect the changes and flag correctness issues, maintainability problems, missing tests, unclear naming, and anything that would make future changes harder. Be concise; cite file:line.`,
	},
	"security": {
		Name:        "security",
		Description: "security-focused review",
		Body: `You are a security-focused code reviewer. Inspect the changes for injection, auth/authz, secrets, path traversal, deserialization of untrusted input, crypto mistakes, and unsafe defaults. Cite file:line and severity.`,
	},
	"performance": {
		Name:        "performance",
		Description: "performance and scalability",
		Body: `You are a performance-focused code reviewer. Look for O(n^2) loops, unnecessary allocations, blocking I/O, missing concurrency limits, N+1 queries, hot-path inefficiencies, and resource leaks. Cite file:line.`,
	},
	"architecture": {
		Name:        "architecture",
		Description: "architecture and API design",
		Body: `You are an architecture-focused code reviewer. Evaluate layering, API surface, coupling, abstraction boundaries, backwards compatibility, and whether the change fits the existing codebase structure. Cite file:line.`,
	},
}

const coordinatorBody = `You are a review coordinator. Several specialist reviewers have independently inspected the same diff. Synthesize their findings into one coherent, structured PR review comment.

Rules:
- Remove duplicates and merge related points.
- Group by theme (Blocking, Should-fix, Nit, Architecture, Security, Performance).
- Lead with a one-sentence verdict.
- Keep it concise and actionable.
- Preserve file:line citations when available.`

type reviewResult struct {
	Name   string `json:"name"`
	Output string `json:"output"`
	Err    string `json:"error,omitempty"`
}

func multiReviewCommand(args []string) int {
	fs := flag.NewFlagSet("multi-review", flag.ContinueOnError)
	base := fs.String("base", "", "base branch/commit to diff against (defaults to HEAD — reviews uncommitted working-tree changes)")
	commit := fs.String("commit", "", "review a specific commit (shows changes introduced by that commit)")
	model := fs.String("model", "", "provider name override (default: config default_model)")
	instructions := fs.String("instructions", "", "extra review instructions appended to every reviewer prompt")
	team := fs.String("team", "quality,security,performance", "comma-separated reviewer personas to run")
	outputFormat := fs.String("output-format", "text", "output format: text | json")
	language := fs.String("language", "", "response language: zh | en (empty = follow config)")
	maxSteps := fs.Int("max-steps", 12, "max tool-call rounds per reviewer")
	maxConcurrency := fs.Int("max-concurrency", 3, "max parallel reviewers")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	diff, err := getReviewDiff(*base, *commit)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		return 1
	}
	if diff == "" {
		fmt.Println("No changes to review.")
		return 0
	}

	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, "error: failed to load config:", err)
		return 1
	}
	modelName := *model
	if modelName == "" {
		modelName = cfg.DefaultModel
	}
	entry, ok := cfg.ResolveModel(modelName)
	if !ok {
		fmt.Fprintf(os.Stderr, "error: unknown model %q — check your config\n", modelName)
		return 1
	}
	if err := cfg.Validate(modelName); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		return 1
	}

	prov, err := boot.NewProviderWithProxy(entry, cfg.NetworkProxySpec())
	if err != nil {
		fmt.Fprintln(os.Stderr, "error: failed to create provider:", err)
		return 1
	}

	root, _ := os.Getwd()
	skillStore := skill.New(skill.Options{ProjectRoot: root, Stderr: os.Stderr})
	reviewSk, ok := skillStore.Read("review")
	if !ok {
		fmt.Fprintln(os.Stderr, "error: built-in review skill not found")
		return 1
	}
	if reviewSk.RunAs != skill.RunSubagent {
		fmt.Fprintln(os.Stderr, "error: review skill is not a subagent skill")
		return 1
	}
	reg := buildReviewSubagentRegistry(reviewSk, cfg, root)

	personas := resolvePersonas(*team)
	if len(personas) == 0 {
		fmt.Fprintln(os.Stderr, "error: no valid personas in --team")
		return 2
	}

	baseTask := buildReviewTask(diff, *instructions)
	lang := strings.TrimSpace(*language)

	results := make([]reviewResult, len(personas))

	sem := make(chan struct{}, *maxConcurrency)
	var wg sync.WaitGroup
	ctx := context.Background()

	for i, p := range personas {
		wg.Add(1)
		go func(idx int, p persona) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			sess := agent.NewSession(p.Body)
			opts := agent.Options{
				MaxSteps:      *maxSteps,
				Temperature:   cfg.Agent.Temperature,
				Pricing:       entry.Price,
				ContextWindow: entry.ContextWindow,
			}
			if lang != "" {
				opts.ResponseLanguage = lang
			}

			out, runErr := agent.RunSubAgentWithSession(ctx, prov, reg, sess, baseTask, opts, event.Discard)
			results[idx].Name = p.Name
			if runErr != nil {
				results[idx].Err = runErr.Error()
				return
			}
			results[idx].Output = out
		}(i, p)
	}
	wg.Wait()

	// Surface any reviewer errors on stderr but keep going; the coordinator can
	// still produce a partial summary.
	for _, r := range results {
		if r.Err != "" {
			fmt.Fprintf(os.Stderr, "error: %s reviewer failed: %s\n", r.Name, r.Err)
		}
	}

	coordinatorTask := buildCoordinatorTask(results)
	coordinatorSess := agent.NewSession(coordinatorBody)
	coordinatorOpts := agent.Options{
		MaxSteps:      *maxSteps,
		Temperature:   cfg.Agent.Temperature,
		Pricing:       entry.Price,
		ContextWindow: entry.ContextWindow,
	}
	if lang != "" {
		coordinatorOpts.ResponseLanguage = lang
	}
	final, err := agent.RunSubAgentWithSession(ctx, prov, reg, coordinatorSess, coordinatorTask, coordinatorOpts, event.Discard)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error: coordinator failed:", err)
		return 1
	}

	switch strings.ToLower(*outputFormat) {
	case "json":
		payload := map[string]interface{}{
			"result":   final,
			"reviews":  results,
			"personas": personaNames(personas),
		}
		b, err := json.MarshalIndent(payload, "", "  ")
		if err != nil {
			fmt.Fprintln(os.Stderr, "error: failed to marshal json:", err)
			return 1
		}
		fmt.Println(string(b))
	default:
		fmt.Print(final)
	}
	return 0
}

func resolvePersonas(team string) []persona {
	var out []persona
	for _, raw := range strings.Split(team, ",") {
		name := strings.TrimSpace(strings.ToLower(raw))
		if name == "" {
			continue
		}
		p, ok := builtinPersonas[name]
		if !ok {
			// Unknown personas are ignored silently for now; in the future this
			// could be a custom skill lookup.
			fmt.Fprintf(os.Stderr, "warning: unknown persona %q, skipping\n", name)
			continue
		}
		out = append(out, p)
	}
	return out
}

func personaNames(ps []persona) []string {
	names := make([]string, len(ps))
	for i, p := range ps {
		names[i] = p.Name
	}
	return names
}

func buildCoordinatorTask(results []reviewResult) string {
	var b strings.Builder
	b.WriteString("Synthesize the following reviewer outputs into a single PR review comment.\n\n")
	for _, r := range results {
		if r.Err != "" {
			fmt.Fprintf(&b, "## %s reviewer (failed: %s)\n\n", r.Name, r.Err)
			continue
		}
		fmt.Fprintf(&b, "## %s reviewer\n\n%s\n\n", r.Name, r.Output)
	}
	return b.String()
}
