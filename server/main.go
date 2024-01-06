package main

import (
	"io"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"os/exec"
)

type Hello struct {
	message string
}

const (
	uploadDir = "./upload"
)


func uploadHandler(w http.ResponseWriter, r *http.Request) {
    // Parse the multipart form data
    err := r.ParseMultipartForm(10 << 20) // 10 MB maximum file size
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    // Get the file from the "file" field
    file, handler, err := r.FormFile("file")
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    defer file.Close()

    // Save the file to disk or perform any other desired operations
    // Here, we simply print the file name
    fmt.Fprintf(w, "Uploaded file: %s\n", handler.Filename)

	// Create a destination file path
	destFilePath := filepath.Join(uploadDir, handler.Filename)

	// Create the destination directory if it doesn't exist
	err = os.MkdirAll(uploadDir, os.ModePerm)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Create the destination file
	destFile, err := os.Create(destFilePath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer destFile.Close()

	// Copy the contents of the uploaded file to the destination file
	_, err = io.Copy(destFile, file)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Respond with a success message
	fmt.Fprintf(w, "File uploaded successfully")
}

func homeHandle(w http.ResponseWriter, r *http.Request) {

	fmt.Println("Hello world")
	
	hello := Hello{
		message: "Hello world",
	};

	response, err := json.Marshal(hello)

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Set the Content-Type header to application/json
	w.Header().Set("Content-Type", "application/json")

	// Write the JSON response
	w.Write(response)
}

func handleDeploy(w http.ResponseWriter, r *http.Request) {
	// Command to run the bash script
	cmd := exec.Command("/bin/bash", "./deploy.sh")

	// Set the environment variables for the command if needed
	cmd.Env = os.Environ()

	// Capture the output and error from the command
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatalf("Error running script: %v", err)
	}

	// Print the output of the command
	log.Printf("Script output:\n%s", string(output))
}

func main() {
	http.HandleFunc("/", homeHandle)
    http.HandleFunc("/upload", uploadHandler)
	http.HandleFunc("/deploy", handleDeploy)

	// deploying the server 
    err := http.ListenAndServe(":8080", nil)

	// checking for errors in starting the server
	if err != nil {
		log.Fatal("ListenAndServe: ", err)
	}

	// printing when the server is successful 
	log.Println("The server is running on PORT 8080")
}
