package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func renameInFolder(folderPath string) error {
	entries, err := os.ReadDir(folderPath)
	if err != nil {
		return fmt.Errorf("cannot read %s: %w", folderPath, err)
	}

	var jpegs []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		lower := strings.ToLower(name)
		if strings.HasSuffix(lower, ".jpeg") || strings.HasSuffix(lower, ".jpg") {
			jpegs = append(jpegs, name)
		}
	}

	sort.Strings(jpegs)

	for i, name := range jpegs {
		oldPath := filepath.Join(folderPath, name)
		newName := fmt.Sprintf("img_%04d.jpeg", i+1)
		newPath := filepath.Join(folderPath, newName)

		if err := os.Rename(oldPath, newPath); err != nil {
			return fmt.Errorf("rename %s -> %s: %w", oldPath, newPath, err)
		}
		fmt.Printf("  %s -> %s\n", name, newName)
	}

	fmt.Printf("  [%s] %d files renamed\n", filepath.Base(folderPath), len(jpegs))
	return nil
}

func main() {
	base := "dataset/raw"
	if len(os.Args) > 1 {
		base = os.Args[1]
	}

	folders := []string{
		filepath.Join(base, "clase_0"),
		filepath.Join(base, "clase_1"),
	}

	for _, folder := range folders {
		fmt.Printf("\nProcessing %s ...\n", folder)
		if err := renameInFolder(folder); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	}
}
