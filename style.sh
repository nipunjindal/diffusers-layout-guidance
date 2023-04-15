#!/bin/bash

# Define the required tools
tools=("isort" "flake8" "mypy" "black")

# Check if each tool is installed
for tool in "${tools[@]}"
do
    if ! command -v "$tool" &> /dev/null
    then
        echo "$tool could not be found. Please install it using 'pip install $tool'."
        exit 1
    fi
done

# Change to the root directory of your project
cd layoutguidance

# Run isort to sort imports
isort .

# Run flake8 to check for syntax errors and style issues
flake8 .

# Run mypy to check for type errors
mypy .

# Run black to format the code
black .
