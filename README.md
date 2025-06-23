# Gradio Hello World App

A simple hello world application built with Gradio and managed with Poetry.

## Features

- Interactive web interface with Gradio
- Modern dependency management with Poetry
- Beautiful and responsive UI
- Personalized greetings

## Installation

1. Make sure you have Poetry installed:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

## Usage

Run the application:

```bash
poetry run python app.py
```

The app will start and provide both a local URL (http://127.0.0.1:7860) and a public URL for sharing.

## How it works

1. Enter your name in the text box
2. Click "Greet Me!" or press Enter
3. Receive a personalized greeting message!

## Development

This project uses Poetry for dependency management. To add new dependencies:

```bash
poetry add <package-name>
```

To update dependencies:

```bash
poetry update
```
