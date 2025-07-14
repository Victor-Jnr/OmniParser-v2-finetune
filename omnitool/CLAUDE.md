# CLAUDE.md - OmniTool

This file provides guidance to Claude Code when working with the OmniTool component of the OmniParser repository.

## Overview

OmniTool is a comprehensive AI agent framework that controls a Windows 11 VM using OmniParser's structured screen parsing capabilities combined with various Vision Language Models (VLMs). It provides a complete ecosystem for automated computer interaction and testing.

## Architecture Components

### 1. gradio/ - UI and Agent Framework

**Purpose**: Primary user interface and agent orchestration system

#### Key Files:
- `app.py`: Main Gradio application with multi-LLM support
  - Supports OpenAI (GPT-4o, O1, O3-mini), Anthropic (Claude), Groq, DeepSeek (R1), Qwen (2.5VL)
  - Integrates Windows VM via iframe (localhost:8006)
  - Real-time agent conversation interface

- `agent/anthropic_agent.py`: Claude Computer Use implementation
  - Native Anthropic computer use API integration
  - Direct screenshot analysis and action execution

- `agent/vlm_agent.py`: Multi-VLM agent implementation
  - Converts OmniParser structured output to actionable commands
  - Step-by-step reasoning and action planning

- `agent/vlm_agent_with_orchestrator.py`: Enhanced orchestration capabilities

- `tools/computer.py`: Windows VM control interface
  - HTTP API communication with VM (localhost:5000)
  - Mouse, keyboard, and screenshot operations
  - Coordinate scaling and action execution

- `tools/screen_capture.py`: Screenshot capture utility

#### Starting the Gradio Interface:
```bash
cd omnitool/gradio
python app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000
```

### 2. omni-cli/ - Command Line Interface

**Purpose**: Direct CLI access to OmniParser server functionality

#### Key Files:
- `omni_cli.py`: Comprehensive CLI tool
  - Supports HTTP URLs, local files, base64 strings
  - Chunked transfer for large files (>50MB)
  - Health checking and error handling
  - Multiple transfer methods (JSON, chunked, file upload)

- `CLI-README.md`: Complete usage documentation
- `omni-cli.bat` / `omni-cli.sh`: Platform-specific convenience scripts

#### Usage Examples:
```bash
# Basic usage
python omni_cli.py /path/to/screenshot.png

# URL input
python omni_cli.py https://example.com/image.jpg

# Health check
python omni_cli.py --health
```

### 3. omnibox/ - Windows VM Container

**Purpose**: Dockerized Windows 11 environment for safe agent execution

#### Key Components:

**Container Configuration**:
- `Dockerfile`: QEMU-based Windows VM container
- `compose.yml`: 8GB RAM, 4 CPU cores, 20GB disk allocation
- KVM acceleration support (Linux/Windows hosts)

**VM Setup Automation**:
- `vm/win11setup/setupscripts/setup.ps1`: Complete Windows software installation
  - Python 3.10, Git, Chrome, LibreOffice, VS Code, development tools
  - Firewall configuration and scheduled tasks
  - HTTP server setup on port 5000

- `vm/win11setup/setupscripts/server/main.py`: Flask control server
  - `/execute` endpoint for command execution
  - `/screenshot` endpoint with cursor overlay
  - Thread-safe command synchronization

**Management Scripts**:
- `scripts/manage_vm.sh` / `scripts/manage_vm.ps1`: VM lifecycle management

#### VM Management Commands:
```bash
cd omnitool/omnibox/scripts

# Initial setup (requires Windows 11 ISO)
./manage_vm.sh create

# Start/stop VM
./manage_vm.sh start
./manage_vm.sh stop

# Full reset
./manage_vm.sh delete
rm -rf vm/win11storage
```

#### Prerequisites:
- 30GB free disk space
- Docker Desktop with KVM support
- Windows 11 Enterprise Evaluation ISO (renamed to `custom.iso` in `vm/win11iso/`)

### 4. omniparserserver/ - FastAPI Server

**Purpose**: HTTP API server for OmniParser screen parsing

#### Key Features:
- `omniparserserver.py`: FastAPI-based parsing server
  - Standard and chunked upload endpoints
  - GZIP compression middleware
  - Configurable CPU/GPU device support

#### API Endpoints:
- `POST /parse/`: Standard base64 image parsing
- `POST /parse/chunk/init/`: Initialize chunked upload
- `POST /parse/chunk/upload/`: Upload image chunks
- `POST /parse/chunk/process/`: Process assembled image
- `POST /parse/file/`: Direct file upload
- `GET /probe/`: Health check

#### Starting the Server:
```bash
cd omnitool/omniparserserver
python -m omniparserserver
# Server runs on localhost:8000
```

## System Integration Flow

```
User Input → Gradio UI → OmniParser Server → Structured Output → 
VLM Agent → Action Commands → Windows VM → Execution Results → UI Feedback
```

### Port Configuration:
- **OmniParser Server**: `localhost:8000`
- **Windows VM Web Viewer**: `localhost:8006` (NoVNC)
- **Windows VM Control API**: `localhost:5000` (Flask)
- **Gradio Interface**: `localhost:7860` (default)

## Development Workflow

### Complete Setup Sequence:
1. **Start OmniParser Server** (requires conda environment with weights)
2. **Create/Start Windows VM** (one-time setup: 60-90 minutes)
3. **Launch Gradio Interface** (connects to both server and VM)

### Testing and Development:
- Use `omni-cli/` for direct API testing
- Monitor VM through NoVNC viewer at `localhost:8006`
- Check server health with `/probe` endpoints

## Agent Capabilities

### Supported Models:
- **OpenAI**: GPT-4o, O1, O3-mini
- **Anthropic**: Claude Sonnet (Computer Use)
- **DeepSeek**: R1
- **Qwen**: 2.5VL
- **Groq**: Various models

### Action Types:
- Mouse clicks and movements
- Keyboard input and shortcuts
- Screenshot analysis
- Application navigation
- File system operations

## Configuration Files

### LLM Client Configuration:
- `gradio/agent/llm_utils/oaiclient.py`: OpenAI client
- `gradio/agent/llm_utils/groqclient.py`: Groq client
- `gradio/agent/llm_utils/omniparserclient.py`: OmniParser integration

### VM Configuration:
- `omnibox/vm/win11setup/setupscripts/tools_config.json`: Installed software configuration
- `omnibox/compose.yml`: Docker container specifications

## Security Considerations

- VM runs in isolated Docker container
- Firewall rules configured for necessary ports only
- Responsible AI practices implemented in caption model
- Human oversight recommended for all automated actions

## Troubleshooting

### Common Issues:
1. **"Windows Host not responding"**: VM setup incomplete or server down
   - Check VM status with `curl localhost:5000/probe`
   - Restart VM with management scripts

2. **OmniParser server errors**: Missing weights or environment issues
   - Verify weights in `../weights/icon_caption_florence/`
   - Check conda environment activation

3. **VM creation timeout**: Slow internet or insufficient resources
   - Monitor progress via NoVNC viewer
   - Ensure 30GB free space and KVM support

### Performance Optimization:
- Run OmniParser server on GPU for faster parsing
- Use CPU machine for VM and Gradio interface
- Chunked uploads for large screenshots (>50MB)

## Development Notes

- All components designed for cross-platform development
- Modular architecture allows independent component testing
- Extensive error handling and validation throughout
- Support for both local and remote model deployments