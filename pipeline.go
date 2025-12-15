package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	// Create a shared context
	ctx := context.Background()

	// Run the stages of the pipeline
	if err := Build(ctx); err != nil {
		fmt.Println("Error:", err)
		panic(err)
	}
}

func Build(ctx context.Context) error {
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	// 1. Host Directory: Exclude output folders to prevent cache invalidation loop
	src := client.Host().Directory(".")

	pipCache := client.CacheVolume("pip_cache")

	// 2. Setup Container Base & Install Dependencies
	python := client.Container().
		From("python:3.12").
		// Mount cache first
		WithMountedCache("/root/.cache/pip", pipCache).
		// Mount source code
		WithDirectory("/app", src).
		WithWorkdir("/app").
		// Install dependencies
		WithExec([]string{"pip", "install", "."})

	// 3. Setup DVC (Git init + Pull)
	python = pull_data(python)

	// 4. Run ML Pipeline Stages
	python = run_script(python, "new_customers_classifier/preprocessing.py", "/app/data/raw/raw_data.csv")
	python = run_script(python, "new_customers_classifier/model_dev.py", "/app/data/processed/train_data_gold.csv")
	python = run_script(python, "new_customers_classifier/model_selection.py")
	python = run_script(python, "new_customers_classifier/deploy.py")

	// 5. Export Artifacts
	return export_artifacts(python, ctx, "models")
}

// --- Helper Functions ---
func pull_data(container *dagger.Container) *dagger.Container {
	return container.
		WithExec([]string{"git", "init"}).
		WithExec([]string{"dvc", "update", "data/raw/raw_data.csv.dvc"})
}

func run_script(container *dagger.Container, scriptPath string, args ...string) *dagger.Container {
	// Helper to handle appending arguments cleanly
	cmd := append([]string{"python", scriptPath}, args...)
	return container.WithExec(cmd)
}

func export_artifacts(container *dagger.Container, ctx context.Context, exportPath string) error {
	// Exports the 'models' directory from container to host
	_, err := container.
		Directory("models").
		Export(ctx, exportPath)
	return err
}
