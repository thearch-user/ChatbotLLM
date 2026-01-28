# ChatbotLLM Project

## Overview
This project aims to develop a comprehensive system for building and deploying a Large Language Model (LLM) based chatbot. It includes components for neural network core operations, various machine learning algorithms, data handling, model architecture, tokenization, and a web-based interface for interaction.

## Features
-   **Core Neural Network Operations**: Custom implementations for autograd, tensor operations, and optimization algorithms.
-   **Neural Network Layers**: Includes attention mechanisms and linear layers.
-   **Algorithms**: Implementations for reinforcement learning (reward functions, RL trainer) and semi-supervised learning (pseudo-labeling, SSL trainer).
-   **Dataset Management**: Tools for downloading and preparing datasets (e.g., Wikipedia).
-   **LLM Model**: Definition and architecture for the Large Language Model.
-   **Tokenizer**: Custom tokenizer training and implementation for text processing.
-   **Web Interface**: A modern web application built with React, providing a user-friendly interface to interact with the chatbot.

## 🔹 Quick start in 5 mins

Get the ChatbotLLM up and running locally in just a few steps.

1.  **Install Backend Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Start the Backend API**:
    ```bash
    # From the root directory
    python backend/main.py
    ```
3.  **Setup and Run the Web Interface**:
    ```bash
    cd Web
    npm install
    npm run dev
    ```
4.  **Interact**: Open your browser to `http://localhost:5173` and start chatting!

## 🔹 Architecture overview

ChatbotLLM is built on a custom-designed neural network framework, prioritizing transparency and educational clarity.

-   **Core Engine (`/core`)**: A from-scratch implementation of a Tensor library with automatic differentiation (`autograd.py`), fundamental operations (`ops.py`), and optimizers (`optim.py`).
-   **Neural Network Layers (`/core/nn`)**: Modular building blocks including Multi-Head Attention, Embeddings, and Linear layers.
-   **Transformer Architecture (`/model`)**: A standard Transformer implementation using the custom core, featuring `TransformerEncoderBlock` and configurable `ModelArgs`.
-   **Algorithms (`/algorithms`)**: Extends basic training with Reinforcement Learning (RLHF-ready) and Semi-Supervised Learning (SSL) capabilities.
-   **Full-Stack Integration**: A FastAPI backend serving the model and a React/Vite frontend for a modern user experience.

## 🔹 What models/weights it supports

-   **GPT-2 Family**: Fully compatible with GPT-2, GPT-2 Medium, Large, and XL weights from the [OpenAI GPT-2 Repository](https://github.com/openai/gpt-2) (via Hugging Face).
-   **Automated Loader**: Use the provided script to download and convert weights into the custom Tensor format:
    ```bash
    python scripts/load_gpt2_weights.py gpt2
    ```
-   **Custom Architectures**: Primarily designed for the internal `Transformer` implementation, prioritizing educational transparency.
-   **Weight Formats**: Supports standard PyTorch `.pth` or `.bin` state dictionaries (mapped to custom Tensor objects via the loader).
-   **Training Data**: Built-in support for Wikipedia, Project Gutenberg, and custom dialogue datasets.

## 🔹 Sample output interactions

The current implementation provides a foundation for interaction. Here is what the current API-driven conversation looks like:

**User:**
> "Hello! Can you tell me about the architecture of this model?"

**ChatbotLLM:**
> "Received text for prediction: Hello! Can you tell me about the architecture of this model?
> 
> Dummy prediction for: 'Hello! Can you tell me about the architecture of this model?'"

*(Note: The model currently returns placeholder responses until weights are loaded and the tokenizer is fully integrated into the backend.)*

## Project Structure
```
.
├── .gitignore
├── requirements.txt
├── algorithms/
│   ├── reinforcement_learning/
│   │   ├── reward_functions.py
│   │   └── rl_trainer.py
│   └── semi_supervised_learning/
│       ├── pseudo_labeling.py
│       └── ssl_trainer.py
├── core/
│   ├── autograd.py
│   ├── ops.py
│   ├── optim.py
│   ├── tensor.py
│   └── nn/
│       ├── attention.py
│       └── linear.py
├── datasets/
│   └── wikipedia_download.py
├── model/
│   ├── model_args.py
│   └── model.py
├── tokenizor/
│   ├── tokenizor_train.py
│   └── tokenizor.py
└── Web/
    ├── public/
    └── src/
        ├── App.jsx
        └── main.jsx
```

## Installation

### Backend (Python)
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/ChatbotLLM.git
    cd ChatbotLLM
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Frontend (Web)
1.  Navigate to the `Web` directory:
    ```bash
    cd Web
    ```
2.  Install Node.js dependencies:
    ```bash
    npm install
    ```

## Usage

### Backend (Python)
Instructions for training models, running algorithms, or using the tokenizer will go here. E.g.:
```bash
# Example: Run an RL trainer
python algorithms/reinforcement_learning/rl_trainer.py
# Example: Train the tokenizer
python tokenizor/tokenizor_train.py
```

### Frontend (Web)
To start the development server for the web interface:
1.  Navigate to the `Web` directory:
    ```bash
    cd Web
    ```
2.  Start the development server:
    ```bash
    npm run dev
    ```
    The application will typically be available at `http://localhost:5173` (or another port as indicated in your terminal).

## Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details (if applicable).