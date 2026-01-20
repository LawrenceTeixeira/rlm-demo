# 🧠 RLM Streamlit Demo

An interactive web application to demonstrate how **Reasoning Language Models (RLM)** work using a REPL environment to recursively process large contexts.

## Features

- **Interactive Input**: Enter your own context and queries
- **Live Execution Log**: Watch the RLM process in real-time with collapsible expanders
- **Code Visualization**: See the Python code the model generates with syntax highlighting
- **Sub-LLM Calls**: Observe how the model delegates to sub-LLMs for chunk analysis
- **Example Loader**: One-click "needle in haystack" demo

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## How It Works

1. **Root LLM** receives your query and context
2. It writes **Python code** in a REPL environment
3. The REPL can call **sub-LLMs** to analyze chunks of the context
4. Results are combined to form the **final answer**

## Configuration

Use the sidebar to configure:

| Option | Description |
|--------|-------------|
| **Root Model** | Main LLM (gpt-5, gpt-4o, gpt-4o-mini) |
| **Sub-LLM Model** | Model for chunk analysis (gpt-5-nano, gpt-4o-mini) |
| **Max Iterations** | Maximum REPL interaction cycles (1-20) |

## Example: Needle in Haystack

Click **📋 Load Example** to generate a context with 1000 lines of random text and a hidden "magic number". The RLM will:

1. Analyze the context structure
2. Chunk the text into manageable pieces
3. Query sub-LLMs to search each chunk
4. Return the found magic number

## Environment Variables

Make sure to set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Or on Windows:
```powershell
$env:OPENAI_API_KEY="your-api-key"
```

## Docker

Build and run the application using Docker:

```bash
# Build the image
docker build -t rlm-demo .

# Run the container
docker run -p 8501:8501 --env-file .env rlm-demo
```

The app will be available at `http://localhost:8501`

## Screenshot

```
┌─────────────────────────────────────────────────────────────────┐
│  🧠 RLM Demo                                                    │
├───────────────────────┬─────────────────────────────────────────┤
│  📝 Input             │  📊 Execution Log                       │
│  ┌─────────────────┐  │  ┌─────────────────────────────────────┐│
│  │ Context...      │  │  │ 📋 System Prompt          [expand] ││
│  │                 │  │  │ 🤖 Step 1: Model Response [expand] ││
│  │                 │  │  │ 💻 Code Execution         [expand] ││
│  └─────────────────┘  │  │ 🤖 Step 2: Model Response [expand] ││
│  Query: ___________   │  │ 🎯 FINAL ANSWER                    ││
│  [Load Example][Run]  │  │    The magic number is 1234567     ││
│                       │  └─────────────────────────────────────┘│
└───────────────────────┴─────────────────────────────────────────┘
```

## Files

- `app.py` - Main Streamlit application
- `rlm/` - Core RLM implementation
  - `rlm_repl.py` - RLM with REPL environment
  - `repl.py` - REPL execution environment
  - `utils/prompts.py` - System prompts and templates

## Author

Made by [Lawrence Teixeira](https://www.linkedin.com/in/lawrenceteixeira/)

## Links

- [Lawrence's Blog](https://lawrence.eti.br/)

## License

See [LICENSE](LICENSE) for details.
