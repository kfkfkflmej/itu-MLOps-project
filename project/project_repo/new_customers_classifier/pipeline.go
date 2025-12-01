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

	src := client.Host().Directory(".")

	pipCache := client.CacheVolume("pip_cache")
	dataFile := client.Host().File("/Users/sunechristiansen/sune/MLops_project/itu-MLOps-project/project/dvc_remote_storage/raw_data.csv")

	python := client.Container().From("python:3.12").
		WithDirectory("/app", src).
		WithWorkdir("/app").
		WithMountedCache("/root/.cache/pip", pipCache).
		WithExec([]string{"mkdir", "-p", "new_customers_classifier"}).
		WithMountedFile("/app/dvc_remote_storage/raw_data.csv", dataFile).
		WithExec([]string{"touch", "new_customers_classifier/__init__.py"}).
		WithExec([]string{"touch", "README.md", "LICENSE"}).
		WithExec([]string{"pip", "install", "."}).
		WithExec([]string{"python", "preprocessing.py"}).
		WithExec([]string{"python", "model_dev.py", "artifacts/train_data_gold.csv"}).
		WithExec([]string{"python", "model_selection.py"}).
		WithExec([]string{"python", "deploy.py"}).
		WithExec([]string{"mkdir", "-p", "output"}).
		WithExec([]string{"cp", "-r", "artifacts", "output/"}).
		WithExec([]string{"cp", "-r", "new_customers_classifier", "output/"}).
		WithExec([]string{"cp", "README.md", "LICENSE", "output/"})

	_, err = python.
		Directory("output").
		Export(ctx, "output")
	if err != nil {
		return err
	}

	return nil
}
