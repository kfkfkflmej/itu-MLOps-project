package main

import (
	"context"
	"testing"
)

// Check if artifacts are generated correctly
func TestBuild(t *testing.T) {
	ctx := context.Background()
	if err := Build(ctx, true); err != nil {
		t.Fatalf("Pipeline execution failed: %v", err)
	}
}
